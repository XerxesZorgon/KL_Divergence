# ADR-011: No fractional Kelly or free risk knobs in the sizing API
**Date:** 2026-11-01
**Status:** Accepted

## Context
The parent article's thesis is that the bet fraction comes exactly from the distribution. Its own
λ·Var penalty and half-Kelly simulations conflict with that thesis (listed as errata). The library
must not reintroduce such knobs.

## Decision
No exported function or keyword scales, shrinks or penalizes the log-optimal bet by a free
parameter. Risk enters only through (a) the distribution itself (posteriors, mixtures, sampled
payoffs) or (b) a robust ambiguity set whose radius ε is estimated from data.

## Rationale
- Matches John's thesis and the article's message ("fix the distribution, not the fraction").
- Avoids users tuning a knob to taste, which the article argues against.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| `fraction=` keyword with default 1.0 | Invites use; contradicts thesis |
| Drawdown-constrained Kelly (Busseti et al. 2016) | A different objective with a free risk limit; out of scope for v1 |

## Consequences
- Experiments that need a fractional baseline (E4 comparisons) compute c·a* inline in the
  experiment script, labeled as a baseline.
- The drawdown law x^{2/c−1} is exposed as a diagnostic (`drawdown_probability`), never as a way to
  choose c.
- ε for robust bets is a required argument; helpers estimate it from data, there is no default.
