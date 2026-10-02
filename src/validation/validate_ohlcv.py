"""OHLCV data-quality validation before DuckDB ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")
OHLCV_VALUE_COLUMNS: tuple[str, ...] = (*PRICE_COLUMNS, "volume")
ISSUE_COLUMNS: tuple[str, ...] = ("symbol", "date", "issue_type", "severity", "details")
EXTREME_MOVE_THRESHOLD: float = 0.20


def _empty_issues() -> pd.DataFrame:
    """Return an empty issues DataFrame with the expected schema."""
    return pd.DataFrame(columns=list(ISSUE_COLUMNS))


def _extract_dates(df: pd.DataFrame) -> pd.Series:
    """Extract normalized dates from ``timestamp`` or ``date`` without mutating ``df``."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return pd.Series([pd.NaT] * len(df), index=df.index)


def _build_issues(
    symbol: str,
    dates: pd.Series,
    mask: pd.Series,
    issue_type: str,
    severity: str,
    details: str | pd.Series,
) -> pd.DataFrame:
    """Build issue rows for all index positions where ``mask`` is True."""
    if not mask.any():
        return _empty_issues()

    selected_dates = dates.loc[mask]
    if isinstance(details, str):
        selected_details = pd.Series([details] * int(mask.sum()), index=selected_dates.index)
    else:
        selected_details = details.loc[mask]

    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": selected_dates.values,
            "issue_type": issue_type,
            "severity": severity,
            "details": selected_details.values,
        }
    )


def check_missing_values(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect missing or NaN values in OHLCV price and volume columns.

    Missing candles break indicator calculations, backtests, and position sizing
    because downstream logic assumes a complete bar exists for every session.
    """
    dates = _extract_dates(df)
    issues: list[pd.DataFrame] = []

    for column in OHLCV_VALUE_COLUMNS:
        if column not in df.columns:
            continue

        mask = df[column].isna()
        issues.append(
            _build_issues(
                symbol=symbol,
                dates=dates,
                mask=mask,
                issue_type="missing_value",
                severity="error",
                details=f"Missing or NaN value in column '{column}'",
            )
        )

    if not issues:
        return _empty_issues()

    return pd.concat(issues, ignore_index=True)


def check_ohlc_violations(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect rows where OHLC relationships are logically impossible.

    Violations usually indicate vendor corruption, bad merges, or unit errors.
    Trading on such bars can produce false breakouts and invalid stop-loss levels.
    """
    required = ("open", "high", "low", "close")
    if any(column not in df.columns for column in required):
        return _empty_issues()

    dates = _extract_dates(df)
    valid = df[list(required)].notna().all(axis=1)

    violation_checks: tuple[tuple[str, pd.Series], ...] = (
        ("low_above_high", valid & (df["low"] > df["high"])),
        ("close_above_high", valid & (df["close"] > df["high"])),
        ("close_below_low", valid & (df["close"] < df["low"])),
        ("open_below_low", valid & (df["open"] < df["low"])),
        ("open_above_high", valid & (df["open"] > df["high"])),
    )

    issues: list[pd.DataFrame] = []
    for violation_name, mask in violation_checks:
        issues.append(
            _build_issues(
                symbol=symbol,
                dates=dates,
                mask=mask,
                issue_type="ohlc_violation",
                severity="error",
                details=f"Logical OHLC violation: {violation_name}",
            )
        )

    if not issues:
        return _empty_issues()

    return pd.concat(issues, ignore_index=True)


def check_zero_negative_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect zero or negative prices in OHLC and adjusted-close columns.

    Non-positive prices make returns, volatility, and risk metrics undefined and
    can cause divide-by-zero failures in strategy code.
    """
    dates = _extract_dates(df)
    issues: list[pd.DataFrame] = []

    for column in PRICE_COLUMNS:
        if column not in df.columns:
            continue

        mask = df[column].notna() & (df[column] <= 0)
        issues.append(
            _build_issues(
                symbol=symbol,
                dates=dates,
                mask=mask,
                issue_type="zero_negative_price",
                severity="error",
                details=f"Non-positive price in column '{column}'",
            )
        )

    if not issues:
        return _empty_issues()

    return pd.concat(issues, ignore_index=True)


def check_zero_volume(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flag rows with zero volume without dropping them.

    Zero-volume days can be legitimate for illiquid names or exchange holidays,
    but they still reduce signal quality for volume-based filters and liquidity checks.
    """
    if "volume" not in df.columns:
        return _empty_issues()

    dates = _extract_dates(df)
    mask = df["volume"].notna() & (df["volume"] == 0)

    return _build_issues(
        symbol=symbol,
        dates=dates,
        mask=mask,
        issue_type="zero_volume",
        severity="warning",
        details="Zero volume bar flagged for review",
    )


def check_flat_days(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flag suspicious flat sessions where open, high, low, and close are equal.

    Flat bars can reflect exchange circuit limits or stale vendor data. They often
    carry little informational value and can distort volatility estimates.
    """
    required = ("open", "high", "low", "close")
    if any(column not in df.columns for column in required):
        return _empty_issues()

    dates = _extract_dates(df)
    valid = df[list(required)].notna().all(axis=1)
    mask = valid & (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])

    return _build_issues(
        symbol=symbol,
        dates=dates,
        mask=mask,
        issue_type="flat_day",
        severity="warning",
        details="Flat session where open == high == low == close",
    )


def check_duplicate_rows(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect duplicate (symbol, date) rows within the same fetch batch.

    Duplicate timestamps can double-count PnL, inflate signal counts, and violate
    the database primary key contract on ``(symbol, timestamp, timeframe)``.
    """
    dates = _extract_dates(df)
    valid_dates = dates.notna()
    duplicate_mask = valid_dates & dates.duplicated(keep=False)

    if not duplicate_mask.any():
        return _empty_issues()

    duplicate_counts = dates.groupby(dates).transform("size")
    details = duplicate_counts.loc[duplicate_mask].map(
        lambda count: f"Duplicate (symbol, date) row appears {int(count)} times in fetch"
    )

    return _build_issues(
        symbol=symbol,
        dates=dates,
        mask=duplicate_mask,
        issue_type="duplicate_row",
        severity="error",
        details=details,
    )


def check_extreme_price_jumps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flag large single-day close-to-close moves above the configured threshold.

    Large jumps can be legitimate around splits, bonuses, or news, but unverified
    spikes often indicate bad vendor adjustments or stale reference prices.
    """
    if "close" not in df.columns:
        return _empty_issues()

    dates = _extract_dates(df)
    ordered = df.assign(_date=dates).sort_values("_date", kind="mergesort")
    previous_close = ordered["close"].shift(1)
    valid = ordered["close"].notna() & previous_close.notna() & (previous_close != 0)
    pct_change = (ordered["close"] - previous_close).abs() / previous_close.abs()
    mask = valid & (pct_change > EXTREME_MOVE_THRESHOLD)

    if not mask.any():
        return _empty_issues()

    move_pct = (pct_change.loc[mask] * 100).round(2).astype(str) + "%"
    details = "Single-day close move of " + move_pct + " exceeds 20% threshold"

    return _build_issues(
        symbol=symbol,
        dates=ordered["_date"],
        mask=mask,
        issue_type="extreme_price_jump",
        severity="warning",
        details=details,
    )


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Run all OHLCV quality checks and return detected issues.

    The input DataFrame is never modified. No exceptions are raised for data
    problems; callers receive a structured issues report instead.

    Args:
        df: OHLCV DataFrame using ``timestamp`` (or ``date``) plus OHLCV columns.
        symbol: Symbol associated with the fetch batch.

    Returns:
        Issues DataFrame with columns ``symbol``, ``date``, ``issue_type``,
        ``severity`` (``error`` or ``warning``), and ``details``.
    """
    if df.empty:
        return _empty_issues()

    checks = (
        check_missing_values,
        check_ohlc_violations,
        check_zero_negative_prices,
        check_zero_volume,
        check_flat_days,
        check_duplicate_rows,
        check_extreme_price_jumps,
    )

    issue_frames = [check(df, symbol) for check in checks]
    non_empty = [frame for frame in issue_frames if not frame.empty]

    if not non_empty:
        return _empty_issues()

    issues = pd.concat(non_empty, ignore_index=True)
    return issues.loc[:, ISSUE_COLUMNS]


def init_data_quality_log(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``data_quality_log`` table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_log (
            symbol VARCHAR NOT NULL,
            date TIMESTAMP NOT NULL,
            issue_type VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,
            details VARCHAR NOT NULL,
            logged_at TIMESTAMP NOT NULL
        )
        """
    )


def log_issues(conn: duckdb.DuckDBPyConnection, issues_df: pd.DataFrame) -> None:
    """Append validation issues to the ``data_quality_log`` table.

    Args:
        conn: Open DuckDB connection.
        issues_df: Issues DataFrame returned by ``validate_ohlcv``.
    """
    if issues_df.empty:
        return

    init_data_quality_log(conn)

    logged_at = datetime.now(timezone.utc).replace(tzinfo=None)
    payload = issues_df.loc[:, ISSUE_COLUMNS].copy()
    payload["logged_at"] = logged_at

    conn.register("_issues_df", payload)
    try:
        conn.execute(
            """
            INSERT INTO data_quality_log
            SELECT symbol, date, issue_type, severity, details, logged_at
            FROM _issues_df
            """
        )
    finally:
        conn.unregister("_issues_df")


def _build_sample_dataframe() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame with injected quality problems."""
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "adj_close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000, 1200, 900, 1100, 950],
            "timeframe": ["1d"] * 5,
        }
    )

    bad_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                ]
            ),
            "open": [108.0, 50.0, 200.0, 210.0, 210.0, float("nan"), 300.0],
            "high": [107.0, 51.0, 200.0, 211.0, 211.0, 301.0, 300.0],
            "low": [106.0, 49.0, 200.0, 209.0, 209.0, 299.0, 300.0],
            "close": [108.0, 50.0, 200.0, 210.0, 210.0, 300.0, 300.0],
            "adj_close": [108.0, 50.0, 200.0, 210.0, 210.0, 300.0, -1.0],
            "volume": [0, 800, 700, 650, 650, 500, 600],
            "timeframe": ["1d"] * 7,
        }
    )

    return pd.concat([base, bad_rows], ignore_index=True)


if __name__ == "__main__":
    sample_df = _build_sample_dataframe()
    issues_df = validate_ohlcv(sample_df, symbol="RELIANCE")

    print("Detected OHLCV data-quality issues:")
    print(issues_df.to_string(index=False))

    with duckdb.connect(":memory:") as conn:
        log_issues(conn, issues_df)
        logged = conn.execute(
            "SELECT symbol, date, issue_type, severity, details, logged_at FROM data_quality_log"
        ).df()
        print("\nLogged issues in DuckDB:")
        print(logged.to_string(index=False))
