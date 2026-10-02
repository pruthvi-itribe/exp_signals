"""Signal-screening research layer.

Sits *before* ``strategies/`` in the workflow: a lightweight way to test
whether a candidate signal has any statistical relationship with forward
returns before investing in building a full ``strategies.Strategy`` class
and running it through ``backtest.py``. Nothing in this package writes to
``signals``/``backtest_*`` — it only reads market data and writes to its own
``forward_returns`` table.
"""
