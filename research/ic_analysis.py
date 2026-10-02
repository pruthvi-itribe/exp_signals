"""Information Coefficient (IC) analysis: daily cross-sectional correlation between
a candidate signal and its forward returns.

IC is the standard first screen in quantitative equity research for whether a
signal has any monotonic relationship with subsequent returns — the
cross-sectional setup (correlate across stocks on the same date, not across
time for one stock) is exactly what a "buy the top-ranked, avoid the
bottom-ranked" strategy would need to be true to work.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs from a CLI, never a notebook/display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_ic(
    signal_series: pd.Series,
    forward_return_series: pd.Series,
    dates: pd.Series,
    symbols: pd.Series,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute daily cross-sectional IC between a signal and forward returns.

    For each date, correlates the signal value across all symbols with
    their forward return on that same date. Rows with a NaN signal or NaN
    forward return are dropped before correlating (common near the start
    of a symbol's history, before an indicator has warmed up, or near the
    end of the available range, where the forward return isn't resolvable
    yet).

    Args:
        signal_series: Signal values, one per (symbol, date) row.
        forward_return_series: Forward returns, aligned to the same rows.
        dates: Date for each row, aligned to the same rows.
        symbols: Symbol for each row, aligned to the same rows (not used in
            the correlation itself; kept so callers can pass a DataFrame's
            columns directly without pre-filtering, and for future
            per-symbol diagnostics).
        method: ``'spearman'`` (rank correlation — robust to outliers and
            nonlinear-but-monotonic relationships; the conventional choice
            for IC) or ``'pearson'`` (linear correlation). Spearman is
            computed as the Pearson correlation of the two series' ranks
            rather than via ``pandas``' ``method='spearman'``, which calls
            ``scipy.stats.spearmanr`` under the hood — not a project
            dependency.

    Returns:
        DataFrame with one row per date: ``date``, ``ic`` (NaN if fewer
        than 3 stocks have valid data that day — too few to correlate
        meaningfully), ``n_stocks`` (valid-data stock count used for that
        date's IC).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates).reset_index(drop=True)).dt.normalize(),
            "symbol": pd.Series(symbols).reset_index(drop=True),
            "signal": pd.Series(signal_series).reset_index(drop=True),
            "fwd_return": pd.Series(forward_return_series).reset_index(drop=True),
        }
    )
    frame = frame.dropna(subset=["signal", "fwd_return"])

    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date", sort=True):
        n_stocks = len(group)
        if n_stocks < 3:
            rows.append({"date": date, "ic": np.nan, "n_stocks": n_stocks})
            continue
        if method == "spearman":
            # pandas' method='spearman' calls scipy.stats.spearmanr, which isn't
            # a project dependency. Spearman's rho is exactly the Pearson
            # correlation of the two series' ranks, so compute it that way.
            ic = group["signal"].rank().corr(group["fwd_return"].rank(), method="pearson")
        else:
            ic = group["signal"].corr(group["fwd_return"], method="pearson")
        rows.append({"date": date, "ic": ic, "n_stocks": n_stocks})

    return pd.DataFrame(rows, columns=["date", "ic", "n_stocks"])


def summarize_ic(ic_df: pd.DataFrame) -> dict[str, float]:
    """Summarize a daily IC series into diagnostic statistics, with realistic thresholds.

    How to read these numbers (equity cross-sectional signal research —
    rule-of-thumb bands, not universal law):

      - ``|mean_ic|`` around 0.02-0.05: a WEAK but potentially real signal.
        This is *normal* for a single, simple equity signal — most signals
        that eventually prove useful live in this range, not higher.
      - ``|mean_ic|`` above 0.05: stronger by equity standards, worth real
        attention — though still no guarantee it survives transaction costs
        or capacity constraints once traded.
      - ``mean_ic`` near zero, or ``ic_ir`` near zero, or a non-significant
        ``t_stat`` (roughly ``|t_stat| < 2``): no real edge. Don't read a
        small positive mean alone as success — a small positive mean on a
        noisy daily series is exactly what pure chance looks like.
      - Do NOT expect IC of 0.1 or higher as a normal outcome. That's rare
        even for well-documented systematic signals, and more often signals
        a short/lucky sample or a data leak (e.g. look-ahead bias) worth
        re-checking before treating it as a discovery.

    Args:
        ic_df: Output of ``calculate_ic``.

    Returns:
        Dict with ``mean_ic``, ``std_ic``, ``ic_ir`` (``mean_ic / std_ic``,
        annualized by ``sqrt(252)`` assuming daily IC observations — the
        same annualization convention ``backtest.calculate_metrics`` uses
        for its Sharpe ratio), ``pct_positive_days``, ``t_stat`` (one-sample
        t-test of whether mean IC differs from zero:
        ``mean_ic / (std_ic / sqrt(n))`` — computed directly rather than via
        ``scipy.stats``, which isn't a project dependency), and ``n_days``
        (valid IC observations used).
    """
    valid = ic_df["ic"].dropna()
    n_days = len(valid)

    if n_days == 0:
        return {
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "ic_ir": float("nan"),
            "pct_positive_days": float("nan"),
            "t_stat": float("nan"),
            "n_days": 0,
        }

    mean_ic = float(valid.mean())
    std_ic = float(valid.std())
    pct_positive_days = float((valid > 0).mean())

    if std_ic > 0:
        ic_ir = (mean_ic / std_ic) * np.sqrt(252)
        t_stat = mean_ic / (std_ic / np.sqrt(n_days))
    else:
        ic_ir = float("nan")
        t_stat = float("nan")

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "pct_positive_days": pct_positive_days,
        "t_stat": t_stat,
        "n_days": n_days,
    }


def plot_ic_over_time(ic_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot daily IC (bars) with a rolling 60-day mean IC overlay, saved to a PNG.

    Args:
        ic_df: Output of ``calculate_ic``.
        output_path: File to save to (parent directories created if needed).

    Returns:
        The resolved ``Path`` written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = ic_df.dropna(subset=["ic"]).sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 5))
    if not plot_df.empty:
        rolling_mean = plot_df["ic"].rolling(window=60, min_periods=10).mean()
        ax.bar(plot_df["date"], plot_df["ic"], width=1.0, color="steelblue", alpha=0.4, label="Daily IC")
        ax.plot(plot_df["date"], rolling_mean, color="firebrick", linewidth=1.5, label="60-day rolling mean IC")
        ax.legend(loc="upper right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Cross-Sectional IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
