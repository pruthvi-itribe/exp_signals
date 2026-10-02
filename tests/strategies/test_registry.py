"""Correctness tests for strategies/registry.py's name -> class registry.

_REGISTRY is a module-level singleton shared by the whole test process --
importing strategies.__init__ once (e.g. anywhere else in the suite) already
registers the 5 real strategies as a side effect. To avoid any interaction
with those (or with other test files), every test here registers only
locally-defined dummy classes under names prefixed `_test_registry_dummy_`,
never touching a real strategy's name. Each test's docstring says what
specific bug it would catch if it failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy
from strategies.registry import available_strategies, get_strategy, register_strategy


class _DummyStrategyA(Strategy):
    base_name = "_test_registry_dummy_a"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))


class _DummyStrategyB(Strategy):
    base_name = "_test_registry_dummy_b"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))


def test_register_and_get_strategy_roundtrip():
    """Would catch: register_strategy not actually storing the class, or
    get_strategy looking it up under the wrong key.
    """
    register_strategy("_test_registry_dummy_roundtrip")(_DummyStrategyA)
    assert get_strategy("_test_registry_dummy_roundtrip") is _DummyStrategyA


def test_reregistering_same_class_under_same_name_is_a_noop():
    """A module re-imported (e.g. via two different import paths) would call
    the decorator again for the exact same class -- must not raise.

    Would catch: a registry that treats every registration as new and
    raises on any re-registration, breaking re-importable strategy modules.
    """
    register_strategy("_test_registry_dummy_idempotent")(_DummyStrategyA)
    register_strategy("_test_registry_dummy_idempotent")(_DummyStrategyA)  # must not raise
    assert get_strategy("_test_registry_dummy_idempotent") is _DummyStrategyA


def test_registering_different_class_under_existing_name_raises():
    """Would catch: a name collision between two distinct strategies being
    silently allowed (second registration overwriting the first), which
    would corrupt signals.strategy identity for whichever one loses.
    """
    register_strategy("_test_registry_dummy_collision")(_DummyStrategyA)
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("_test_registry_dummy_collision")(_DummyStrategyB)
    # The original registration must remain untouched by the failed attempt.
    assert get_strategy("_test_registry_dummy_collision") is _DummyStrategyA


def test_get_strategy_unknown_name_raises_key_error_listing_available():
    """Would catch: an unknown-name lookup raising the wrong exception type,
    or the error message failing to help the caller find a valid name.
    """
    register_strategy("_test_registry_dummy_listed")(_DummyStrategyA)
    with pytest.raises(KeyError) as exc_info:
        get_strategy("_test_registry_dummy_definitely_not_registered")
    assert "_test_registry_dummy_listed" in str(exc_info.value)


def test_available_strategies_is_sorted_and_includes_registered_names():
    """Would catch: available_strategies() returning insertion order instead
    of sorted order, or omitting a registered name.
    """
    register_strategy("_test_registry_dummy_zzz_last")(_DummyStrategyA)
    register_strategy("_test_registry_dummy_aaa_first")(_DummyStrategyB)
    names = available_strategies()
    assert names == sorted(names)
    assert "_test_registry_dummy_zzz_last" in names
    assert "_test_registry_dummy_aaa_first" in names
    assert names.index("_test_registry_dummy_aaa_first") < names.index("_test_registry_dummy_zzz_last")


def test_real_strategies_are_registered_on_package_import():
    """A light integration check that importing the strategies package
    actually registers the real, shipped strategies under their documented
    names -- would catch a real strategy module losing its
    @register_strategy decorator, or its base_name being renamed without
    updating callers that look it up by string.
    """
    import strategies  # noqa: F401  (import triggers registration as a side effect)

    for name in ("sma_crossover", "rsi_mean_reversion", "bollinger_breakout", "trend_ladder", "precision_pullback"):
        assert name in available_strategies()
        assert issubclass(get_strategy(name), Strategy)
