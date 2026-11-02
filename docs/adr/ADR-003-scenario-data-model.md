# ADR-003: Scenario / returns-matrix data model
**Date:** 2026-11-01
**Status:** Accepted

## Context
The library must handle single events with mutually exclusive outcomes, simultaneous independent
bets, correlated bets, sampled continuous payoffs, uncertain probabilities and model mixtures.
Separate types for each would duplicate the solver, diagnostics and robust code.

## Decision
Represent every problem as a `ScenarioModel`: scenario probabilities π ∈ Δ^S and a gross-returns
matrix X ∈ ℝ_{≥0}^{S×J} (cash, when present, is a column of ones), with allocation a in the
probability simplex Δ^J, wealth V_s(a) = (X a)_s and growth G(a) = Σ_s π_s log V_s(a).

## Rationale
- E1's proof sketch (L ≤ D(p‖q)) holds for any convex allocation set with wealth linear in a;
  this model is exactly that setting, so the bound and its test apply to every problem type.
- Single events, simultaneous bets (2^n joint outcomes), correlated bets (joint scenario table),
  sampled payoffs (Monte Carlo rows) and mixtures (shared rows, mixed π) all map to (π, X).
- One solver, one KKT certificate and one robust formulation serve all cases.
- Proportional fees and commissions are folded into X (e.g. ô = 1 + (o − 1)(1 − c)), so they keep
  wealth linear.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Separate types per bet structure (as in the parent repo's four modules) | Duplicates solvers; the parent repo's split is part of what this project replaces |
| Net-return (profit) matrix | Gross returns make V = X a directly and match Cover–Thomas notation |
| Continuous distributions handled analytically | No general closed form; sampling to scenarios is exact for the sampled measure and testable |

## Consequences
- Simultaneous independent bets create 2^n scenarios; v1 documents a practical limit of n ≤ 16
  (65 536 rows) before sampling is required.
- Fixed per-bet costs and minimum stakes cannot be expressed (wealth would be affine or the set
  non-convex). They are only simulated in the ruin tools (SDD Non-Goals).
- Allocation set is the simplex only in v1: no leverage, no shorting. Wider convex sets would keep
  the theory but need new constraints and tests.
- Payoff convention is **gross**: a winning unit stake at gross payoff w returns w (net profit
  w − 1). The parent article's two-bet example (p, w) = (0.3, 4), (0.25, 6) reproduces its
  published fractions (0.0645, 0.0992) only under the gross convention (checked numerically
  2026-11-01), so this convention is consistent with the parent article.
