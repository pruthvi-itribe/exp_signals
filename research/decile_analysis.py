"""Bucket (decile/quintile) analysis: average forward return by signal-value bucket.

A complementary view to IC: IC tests for an overall monotonic/linear
relationship in one number, while bucket analysis shows the actual *shape*
of the relationship — whether it's a clean monotonic staircase across every
bucket, or is actually driven by just one extreme bucket, which a summary IC
statistic alone can hide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs from a CLI, never a notebook/display
import matplotlib.pyplot as plt
import pandas as pd


def bucket_by_decile(
    signal_series: pd.Series,
    forward_return_series: pd.Series,
    dates: pd.Series,
    symbols: pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Rank stocks into ``n_buckets`` by signal value, per date, and average forward returns.

    Uses ``pd.qcut`` independently for each date, so bucket boundaries
    adapt to that day's own signal distribution (equal-*count* buckets, not
    equal-value ranges). ``duplicates='drop'`` handles dates where many
    stocks share the same signal value (common for, say, an
    integer-valued RSI) by merging buckets rather than raising — a given
    date may then yield fewer than ``n_buckets`` distinct buckets, which is
    expected and handled downstream by aggregating on bucket *label*
    (1 = lowest signal value that day, ``n_buckets`` = highest), not bucket
    position. Dates with fewer valid stocks than ``n_buckets`` are skipped
    (too few to form that many groups).

    Args:
        signal_series: Signal values, one per (symbol, date) row.
        forward_return_series: Forward returns, aligned to the same rows.
        dates: Date for each row, aligned to the same rows.
        symbols: Symbol for each row, aligned to the same rows.
        n_buckets: Number of buckets to rank into (default 5 = quintiles).

    Returns:
        DataFrame with one row per (date, bucket): ``date``, ``bucket``
        (1..n_buckets), ``mean_fwd_return``, ``n_stocks``.
    """
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
        if len(group) < n_buckets:
            continue
        try:
            bucket = pd.qcut(group["signal"], q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            # All signal values identical that day -> qcut can't form any bins.
            continue

        labeled = group.assign(bucket=bucket + 1)  # 1-indexed: 1 = lowest signal value that day
        for bucket_id, bucket_group in labeled.groupby("bucket"):
            rows.append(
                {
                    "date": date,
                    "bucket": int(bucket_id),
                    "mean_fwd_return": bucket_group["fwd_return"].mean(),
                    "n_stocks": len(bucket_group),
                }
            )

    return pd.DataFrame(rows, columns=["date", "bucket", "mean_fwd_return", "n_stocks"])


def summarize_deciles(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-date bucket returns into one mean forward return per bucket.

    A real, useful signal should show mean forward return moving steadily
    (roughly monotonically) from bucket 1 to bucket N. A flat or
    non-monotonic pattern — even with a large gap between buckets 1 and N —
    means that gap is being driven by noise in one or two buckets rather
    than a real, consistent ordering, and should be read as no meaningful
    signal regardless of what the IC statistics say.

    Args:
        bucket_df: Output of ``bucket_by_decile``.

    Returns:
        DataFrame indexed by bucket (1..n_buckets) with columns
        ``mean_fwd_return`` (equal-weighted average across all dates) and
        ``n_observations``, plus one final row labeled ``'spread'`` whose
        ``mean_fwd_return`` is (top bucket mean - bottom bucket mean).
    """
    if bucket_df.empty:
        return pd.DataFrame(columns=["mean_fwd_return", "n_observations"])

    per_bucket = (
        bucket_df.groupby("bucket")
        .agg(mean_fwd_return=("mean_fwd_return", "mean"), n_observations=("n_stocks", "sum"))
        .sort_index()
    )

    top_bucket = per_bucket.index.max()
    bottom_bucket = per_bucket.index.min()
    spread = per_bucket.loc[top_bucket, "mean_fwd_return"] - per_bucket.loc[bottom_bucket, "mean_fwd_return"]

    spread_row = pd.DataFrame(
        {"mean_fwd_return": [spread], "n_observations": [int(per_bucket["n_observations"].sum())]},
        index=["spread"],
    )
    return pd.concat([per_bucket, spread_row])


def plot_decile_returns(summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Bar chart of mean forward return per bucket, saved to a PNG.

    Args:
        summary_df: Output of ``summarize_deciles``. The trailing
            ``'spread'`` row is excluded from the chart — it's a derived
            diagnostic, not a bucket.
        output_path: File to save to (parent directories created if needed).

    Returns:
        The resolved ``Path`` written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.drop(index="spread", errors="ignore")

    fig, ax = plt.subplots(figsize=(7, 5))
    if not plot_df.empty:
        colors = ["firebrick" if v < 0 else "steelblue" for v in plot_df["mean_fwd_return"]]
        ax.bar(plot_df.index.astype(str), plot_df["mean_fwd_return"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Forward Return by Signal Bucket")
    ax.set_xlabel("Bucket (1 = lowest signal value)")
    ax.set_ylabel("Mean Forward Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
