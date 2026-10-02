# `candidates/` — Production-Candidate Strategy Writeups

One file per strategy that has cleared backtesting and looks genuinely
worth taking toward real capital — as opposed to every strategy in
`strategies/`, most of which exist as research/comparison points. A
strategy earns a file here only after a real backtest against a
buy-and-hold benchmark over the same window, not from the screening stage
alone.

Each writeup should cover, in plain language:

- What the strategy actually does and why (the one-paragraph version
  someone could act on without reading the code).
- How it decides what to hold — the actual rule, not just "it's a
  momentum strategy."
- Backtested performance against a buy-and-hold benchmark over the
  identical window and universe — CAGR alone is not enough; see
  `PERFORMANCE.md` for the raw numbers this pulls from.
- What's genuinely good about it (the reason it's here at all).
- Open questions and risks that should be resolved before real capital —
  named honestly, not glossed over. A candidate with no listed risks
  hasn't been looked at hard enough.
- Pointers to the strategy's own module (`strategies/<name>.py`) and
  `PERFORMANCE.md` for the full technical trail, rather than repeating it.

This directory is a running log, not a one-time snapshot — update a
strategy's file as new backtests or live-readiness findings come in,
rather than only writing it once.
