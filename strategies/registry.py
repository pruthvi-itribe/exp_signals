"""Name -> class registry for strategies, so callers can look strategies up
by string (``get_strategy("sma_crossover")``) instead of an if/elif chain
or importing each concrete strategy module directly.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}

StrategyClass = TypeVar("StrategyClass", bound=type[Strategy])


def register_strategy(name: str) -> Callable[[StrategyClass], StrategyClass]:
    """Class decorator registering a ``Strategy`` subclass under ``name``.

    Args:
        name: Lookup key, e.g. ``"sma_crossover"``. By convention this
            matches the strategy's ``base_name``, though the registry itself
            doesn't enforce that.

    Raises:
        ValueError: If ``name`` is already registered to a different class
            (re-registering the same class under the same name, e.g. from a
            module being imported twice, is a no-op).
    """

    def decorator(cls: StrategyClass) -> StrategyClass:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Strategy name '{name}' is already registered to {existing!r}.")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> type[Strategy]:
    """Look up a registered strategy class by name.

    Args:
        name: Registry key, e.g. ``"sma_crossover"``.

    Returns:
        The ``Strategy`` subclass registered under ``name`` (not an
        instance — call it with config kwargs to instantiate).

    Raises:
        KeyError: If ``name`` isn't registered. The message lists what is.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}") from None


def available_strategies() -> list[str]:
    """Return all registered strategy names, sorted."""
    return sorted(_REGISTRY)
