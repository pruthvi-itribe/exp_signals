"""Strategy package: base class, config, registry, and concrete strategies.

Importing this package registers every strategy module below with
``strategies.registry`` as a side effect — callers only need
``get_strategy(name)``; they never need to know which module a given
strategy lives in.
"""

from strategies.base import Strategy, StrategyConfig
from strategies.registry import available_strategies, get_strategy, register_strategy

# Import concrete strategy modules for their @register_strategy side effect.
from strategies import (  # noqa: F401
    bollinger_breakout,
    precision_pullback,
    rsi_mean_reversion,
    sma_crossover,
    trend_ladder,
)

__all__ = [
    "Strategy",
    "StrategyConfig",
    "get_strategy",
    "register_strategy",
    "available_strategies",
]
