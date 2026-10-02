"""Correctness tests for strategies/base.py's Strategy/StrategyConfig contract.

Uses locally-defined dummy Strategy/StrategyConfig subclasses (never
registered via @register_strategy) so these tests exercise only the base
class's own machinery, independent of any concrete strategy's logic or the
shared strategy registry. Each test's docstring says what specific bug it
would catch if it failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig


@dataclass(frozen=True)
class _DummyConfig(StrategyConfig):
    threshold: int = field(default=5, metadata={"description": "A dummy threshold."})
    label: str = field(default="x", metadata={"description": ""})

    def validate(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")


class _DummyStrategy(Strategy):
    base_name = "dummy"
    config_cls = _DummyConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))


def test_validate_columns_passes_when_all_present():
    """Would catch: validate_columns raising a false positive when every
    required column is actually present."""
    strategy = _DummyStrategy()
    df = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"], "adj_close": [1.0]})
    strategy.validate_columns(df)  # must not raise


def test_validate_columns_raises_with_exact_missing_names():
    """Would catch: a missing-column check that reports the wrong names, or
    silently passes with columns actually absent.
    """
    strategy = _DummyStrategy()
    df = pd.DataFrame({"symbol": ["A"]})
    with pytest.raises(ValueError) as exc_info:
        strategy.validate_columns(df)
    assert "date" in str(exc_info.value)
    assert "adj_close" in str(exc_info.value)
    assert "symbol" not in str(exc_info.value)  # symbol WAS present, must not be reported missing


def test_strategy_config_base_validate_is_a_noop():
    """The base StrategyConfig has no fields and validate() must be a no-op
    -- would catch a base-class validate() that raises for the trivial
    no-fields case.
    """
    StrategyConfig().validate()  # must not raise


def test_config_kwargs_build_dataclass_and_call_validate():
    """__init__ must forward kwargs into config_cls and then call
    config.validate() -- an invalid combination must surface as ValueError
    at construction time, not later inside generate_signals.

    Would catch: validate() never being called (an invalid config
    silently accepted) or being called before the config is fully built.
    """
    strategy = _DummyStrategy(threshold=10, label="y")
    assert strategy.config.threshold == 10
    assert strategy.config.label == "y"

    with pytest.raises(ValueError, match="threshold"):
        _DummyStrategy(threshold=-1)


def test_unknown_config_kwarg_raises_type_error():
    """Would catch: silently ignoring an unrecognized parameter name (e.g. a
    typo'd kwarg) instead of failing loudly.
    """
    with pytest.raises(TypeError):
        _DummyStrategy(not_a_real_field=1)


def test_name_defaults_to_base_name():
    """Would catch: `name` not falling back to `base_name` when a subclass
    doesn't override the property.
    """
    assert _DummyStrategy().name == "dummy"


def test_name_can_be_overridden_to_fold_in_config():
    """Would catch: a subclass overriding `name` but the base class somehow
    still being consulted (property resolution bug).
    """

    class _NamedStrategy(_DummyStrategy):
        @property
        def name(self) -> str:
            return f"{self.base_name}_{self.config.threshold}"

    assert _NamedStrategy(threshold=7).name == "dummy_7"


def test_generate_signals_is_abstract_and_cannot_be_instantiated_without_it():
    """Would catch: generate_signals losing its @abstractmethod status,
    which would let a broken Strategy subclass silently do nothing instead
    of failing at construction time.
    """

    class _IncompleteStrategy(Strategy):
        base_name = "incomplete"

    with pytest.raises(TypeError):
        _IncompleteStrategy()


def test_param_info_reflects_fields_defaults_and_descriptions():
    """Would catch: param_info() not picking up a field's default or its
    metadata description, or misnaming a field.
    """
    info = _DummyConfig.param_info()
    assert set(info) == {"threshold", "label"}
    assert info["threshold"].default == 5
    assert info["threshold"].description == "A dummy threshold."
    assert info["label"].description == ""


def test_param_info_empty_for_base_class_with_no_fields():
    assert StrategyConfig.param_info() == {}
