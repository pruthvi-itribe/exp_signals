"""Nifty 50 index-level market-regime state, shared by strategies that gate
entries or force an exit on the broader market's own trend (currently
``trend_ladder`` and ``precision_pullback`` -- see their module docstrings
and ``strategies/README.md`` for why this was previously omitted).

THE GAP THIS BRIDGES: ``Strategy.generate_signals(df)`` is handed a merged
panel built by ``src.strategy.load_strategy_input`` (and, for
``validate_strategy.py``, by its own ``_load_ohlcv_history``) -- and until
now, neither loader ever included the Nifty 50 index's own price history in
that panel, so no strategy could see market-wide regime state at all,
regardless of how many *stock* symbols happened to be in the panel at once.
The fix doesn't touch ``Strategy``'s interface or ``backtest.py``'s
simulation engine: the index's own OHLCV is fetched and stored in
``ohlcv_data`` exactly like any other symbol (under ``NIFTY_INDEX_SYMBOL``),
and ``attach_market_regime`` broadcasts its derived regime columns onto
every OTHER symbol's row *by date*, not by symbol, before a strategy ever
sees the data -- so each strategy just gets two more ordinary input columns
to read (or ignore, if it doesn't declare them in ``required_columns``).

INTERPRETATION NOTE: neither this repo's existing docs nor the source-spec
PDFs (not present in this repo) pin down the "exit everything on a specific
index-level breakdown pattern" rule's exact technical definition -- only
that the entry gate is "no new entries while the index is below its own 100
EMA." This module defines "breakdown" as the day the index's own close
crosses from at/above its 100-EMA to below it: the same condition that
gates entries, mirrored as a one-time exit trigger the day it first turns
bearish. This is a reasonable, clearly-flagged interpretation, not a
verified transcription of the original spec's exact wording -- revisit this
if the exact source rule ever surfaces.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.universe import bulk_fetch_and_store

NIFTY_INDEX_SYMBOL: str = "^NSEI"
REGIME_EMA_PERIOD: int = 100

REGIME_COLUMNS: tuple[str, ...] = ("nifty_regime_bullish", "nifty_regime_breakdown")


def fetch_nifty_index_history(
    conn: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
) -> dict[str, list[str]]:
    """Fetch and store the Nifty 50 index's own OHLCV for ``[start_date, end_date)``.

    Reuses the exact same fetch/validate/upsert pipeline as every tradeable
    symbol (``bulk_fetch_and_store``) -- the index lands in ``ohlcv_data``
    under ``NIFTY_INDEX_SYMBOL`` like any other symbol, but is never added to
    the ``universe`` table, so ``get_active_universe`` (and anything built on
    it) never mistakes it for a tradeable stock.
    """
    return bulk_fetch_and_store(
        conn, tickers=[NIFTY_INDEX_SYMBOL], start_date=start_date, end_date=end_date
    )


def load_market_regime(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load the Nifty 50 index's full stored history and compute its regime state.

    Returns a DataFrame with one row per date the index has data for:
    ``date``, ``nifty_close``, ``nifty_ema_100``, ``nifty_regime_bullish``
    (bool: close >= its own 100-EMA, fail-open ``True`` during the EMA's own
    warm-up), ``nifty_regime_breakdown`` (bool: the single day the index's
    close first crosses from at/above its 100-EMA to below it -- see this
    module's docstring for why "breakdown" is defined this way).

    Empty (with the same columns) if the index has no stored history at all
    -- call ``fetch_nifty_index_history`` first. Callers should treat an
    empty result as "regime data unavailable" and fail open (see
    ``attach_market_regime``), never as "the market is always bearish."
    """
    empty_columns = ["date", "nifty_close", "nifty_ema_100", *REGIME_COLUMNS]
    index_df = conn.execute(
        """
        SELECT timestamp::DATE AS date, close AS nifty_close
        FROM ohlcv_data
        WHERE timeframe = '1d' AND symbol = ?
        ORDER BY date
        """,
        [NIFTY_INDEX_SYMBOL],
    ).df()
    if index_df.empty:
        return pd.DataFrame(columns=empty_columns)

    index_df["date"] = pd.to_datetime(index_df["date"]).dt.normalize()
    index_df["nifty_ema_100"] = (
        index_df["nifty_close"]
        .ewm(span=REGIME_EMA_PERIOD, adjust=False, min_periods=REGIME_EMA_PERIOD)
        .mean()
    )

    ema_ready = index_df["nifty_ema_100"].notna()
    bullish_raw = index_df["nifty_close"] >= index_df["nifty_ema_100"]
    bullish = bullish_raw.where(ema_ready, True)  # fail open during the EMA's own warm-up
    prev_bullish = bullish.shift(1).fillna(True)  # no prior day -> cannot be a breakdown

    index_df["nifty_regime_bullish"] = bullish
    index_df["nifty_regime_breakdown"] = (~bullish) & prev_bullish

    return index_df.loc[:, empty_columns]


def attach_market_regime(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the index's regime columns onto every row of ``df``, by date.

    ``df`` may hold many symbols; every row for a given date gets the SAME
    regime values (a market-wide condition, not a per-symbol one) via a left
    join on ``date`` alone -- never on ``symbol``.

    Fails open wherever regime data doesn't cover a date (``df``'s own date
    range predates the index's stored history, ``regime_df`` is empty
    because the index was never fetched, or a date falls inside the index's
    own 100-EMA warm-up): ``nifty_regime_bullish`` defaults to ``True`` and
    ``nifty_regime_breakdown`` to ``False`` -- exactly the behavior every
    strategy had before this feature existed. This is deliberate: silently
    blocking every entry because the index simply hasn't been fetched yet
    would be a much worse failure mode than silently not gating at all.
    """
    if df.empty:
        return df

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()

    if regime_df.empty:
        working["nifty_regime_bullish"] = True
        working["nifty_regime_breakdown"] = False
        return working

    merged = working.merge(regime_df.loc[:, ["date", *REGIME_COLUMNS]], on="date", how="left")
    merged["nifty_regime_bullish"] = merged["nifty_regime_bullish"].fillna(True).astype(bool)
    merged["nifty_regime_breakdown"] = merged["nifty_regime_breakdown"].fillna(False).astype(bool)
    return merged
