"""Abstract base class and config mechanism for trading strategies.

Every concrete strategy declares:
  - ``base_name``: stable identity used as the registry lookup key (see
    ``strategies.registry``) and, by convention, the stem of ``name``.
  - ``config_cls``: the ``StrategyConfig`` dataclass holding its tunable
    parameters. Instantiating a strategy with keyword arguments builds this
    config, e.g. ``SmaCrossoverStrategy(fast_window=10, slow_window=30)``.
  - ``required_columns``: the input DataFrame columns ``generate_signals``
    needs. Generic callers (``backtest.py``, ``validate_strategy.py``) never
    hardcode a strategy's feature columns — they load a superset of raw
    OHLCV data and let each strategy compute whatever derived indicators it
    needs, at whatever parameters its config specifies, internally.
  - ``generate_signals``: pure function from merged market data to signal
    events, matching the ``signals`` table's row shape.

Indicators are computed by the strategy itself rather than read from
``indicators_daily`` on purpose: that table holds exactly one fixed
parameterization per indicator (e.g. SMA at 20/50), which can't serve a
parameter sweep over arbitrary windows. A strategy that needs history before
its own input window to warm up (e.g. a 100-day SMA) should compute over
whatever history it's given and let the caller worry about how much of that
history to load — see ``validate_strategy.py`` for the pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd

SIGNAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "strategy",
    "signal_type",
    "price",
    "reason",
)


class ParamInfo(NamedTuple):
    """One tunable parameter's default value and human-readable description."""

    name: str
    default: object
    description: str


@dataclass(frozen=True)
class StrategyConfig:
    """Base class for a strategy's tunable parameters.

    Subclass per strategy with whatever fields it needs, using
    ``dataclasses.field(default=..., metadata={"description": "..."})`` so
    the description is introspectable (see ``param_info``), e.g.::

        @dataclass(frozen=True)
        class SmaCrossoverConfig(StrategyConfig):
            fast_window: int = field(
                default=20,
                metadata={"description": "Lookback window for the fast moving average."},
            )

    Frozen so a config is hashable and safe to reuse as a grid-search row
    key. The base class has no fields — a strategy with no tunable
    parameters can use it directly.
    """

    @classmethod
    def param_info(cls) -> dict[str, ParamInfo]:
        """Return each tunable parameter's name, default value, and description.

        Descriptions come from each field's ``metadata={"description": ...}``
        (empty string if a field doesn't set one). Used by
        ``backtest_cli.py``'s ``list-params`` command and its
        unknown-parameter validation — add a field to a ``StrategyConfig``
        subclass and it shows up here automatically, no separate
        registration needed.
        """
        return {
            f.name: ParamInfo(name=f.name, default=f.default, description=f.metadata.get("description", ""))
            for f in fields(cls)
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if this config's values don't make sense together.

        No-op by default — override in a subclass to add cross-field checks
        (e.g. ``SmaCrossoverConfig`` requires ``fast_window < slow_window``).
        Called automatically by ``Strategy.__init__`` right after the config
        is built, so an invalid combination fails fast at construction time
        rather than surfacing later as a confusing error inside
        ``generate_signals`` or as a silently-empty signal set.
        """
        return None


class Strategy(ABC):
    """Base class for rule-based trading strategies.

    Subclasses implement ``generate_signals`` to transform merged OHLCV (and
    optionally pre-computed indicator) rows into discrete BUY/SELL signal
    events, matching ``SIGNAL_OUTPUT_COLUMNS``.
    """

    base_name: str
    config_cls: type[StrategyConfig] = StrategyConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    def __init__(self, **config_kwargs: object) -> None:
        """Build this strategy's config from keyword arguments and validate it.

        Args:
            **config_kwargs: Fields of ``config_cls``. Fields omitted here
                fall back to that dataclass's own defaults.

        Raises:
            TypeError: If ``config_kwargs`` contains a name that isn't a
                field of ``config_cls`` (raised by the dataclass constructor
                itself).
            ValueError: If ``config_cls.validate()`` rejects the resulting
                combination of values.
        """
        self.config: StrategyConfig = self.config_cls(**config_kwargs)
        self.config.validate()

    @property
    def name(self) -> str:
        """Identity stored in ``signals.strategy`` / ``backtest_runs.strategy_name``.

        Defaults to ``base_name``. Override to fold the config into the name
        (as ``SmaCrossoverStrategy`` does with its windows) so that different
        parameter combinations of the same strategy don't collide under one
        name in the ``signals`` table's ``(symbol, date, strategy)`` primary
        key — each distinct configuration needs a distinct stored identity.
        """
        return self.base_name

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if ``df`` is missing any ``required_columns``."""
        missing = [column for column in self.required_columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} input is missing required columns: {', '.join(missing)}"
            )

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signal rows from merged market data.

        Args:
            df: DataFrame containing at least ``required_columns`` for one or
                more symbols, sorted by ``symbol`` and ``date``. May contain
                more history than the caller ultimately cares about — that's
                intentional, since indicators need warm-up history before the
                window of interest.

        Returns:
            DataFrame matching ``SIGNAL_OUTPUT_COLUMNS``. Only signal
            *events* should be emitted, not a HOLD row for every day.
        """


def first_exit_after_each_buy(
    buy: Sequence[bool], exit_condition: Sequence[bool], disarm: Sequence[bool]
) -> list[bool]:
    """Mark the first ``exit_condition`` day after each BUY.

    All three sequences are one symbol's rows in date order; mixing symbols
    would carry one symbol's pending exit into another's rows.

    For "exit on the first day X happens while holding" rules: emits one exit
    per entry instead of a row for every day X holds, without missing the
    exit when X first happens later than the day a level was crossed. BUYs
    are never suppressed, so whenever the engine holds a position, the last
    BUY since the previous exit has armed exactly one exit.

    A ``disarm`` day (e.g. a market-wide exit-all, which the caller emits
    separately) clears the pending exit and is never itself marked.
    """
    marks = [False] * len(buy)
    armed = False
    for i in range(len(buy)):
        if disarm[i]:
            armed = False
        elif armed and exit_condition[i]:
            marks[i] = True
            armed = False
        if buy[i]:
            armed = True
    return marks
