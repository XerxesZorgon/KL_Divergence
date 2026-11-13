# ADR-015: Diagnostics stay uncapped in v0.2
**Date:** 2026-11-11
**Status:** Accepted (approved by John 2026-11-11; closes DESIGN.md §8 OQ-3)

## Context
v0.2 adds per-asset caps to the solvers (ADR-012). Several diagnostics compare with or rescale an
optimal allocation: `growth_loss(m_true, a_belief)` compares with the uncapped optimum a_p under the
truth; `growth_curve` and `zero_growth_point` scale the bet part by c (c > 1 can exceed caps);
`simulate_ruin` applies a given allocation. For a capped bettor, the fair benchmark is the capped optimum.

## Decision
`growth_loss`, `growth_curve`, `zero_growth_point` and `simulate_ruin` take no `upper` in v0.2. Their
docstrings state that they compare with, or scale, the uncapped optimum (or apply the allocation exactly
as given). A capped `growth_loss` is deferred to a later version with its own ADR.

## Rationale
- AFS-GEM's v0.2 need is the capped solve and its certificate; no consumer needs capped diagnostics yet.
- Keeps `diagnostics.jl` and every E2–E7 output untouched (Success Criterion 1).
- A capped `growth_loss` raises its own questions (cap the truth's optimum, the belief's, or both) that
  deserve a separate decision.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Add `upper` to `growth_loss` now | Benchmark choice unsettled; widens M9 scope |
| Clip `growth_curve` at caps | Changes the curve's meaning (no longer a ray from cash through a*) |

## Consequences
- Calling a diagnostic with a capped allocation is allowed but measures against the uncapped benchmark;
  docstrings say so (PLAN.md M9, docstring task).
- For later: E1's bound L ≤ D(p‖q) holds for any convex allocation set with wealth linear in a
  (ADR-003), so it will still hold for a capped `growth_loss` when both optima use the same caps.
