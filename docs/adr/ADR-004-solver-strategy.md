# ADR-004: Solver strategy — closed form first, Clarabel + Newton polish, KKT certification
**Date:** 2026-11-01
**Status:** Accepted

## Context
Log-optimal allocation is a concave maximization. Closed forms exist only for special cases.
General cases (simultaneous, correlated, sampled payoffs, robust objectives) need a conic solver
with exponential cones; HiGHS does not support exponential cones. E1 showed Cover's
multiplicative update can stall when a weight underflows to 0 and produce a false counterexample.
The success criterion requires KKT residual ≤ 1e-10 for every computed bet.

## Decision
Dispatch to (1) the exact closed form when the problem is a single event (full investment: a = q;
with cash: greedy Smoczynski–Tomkins), otherwise (2) Clarabel.jl via JuMP on the exponential-cone
formulation, followed by Newton polishing on the identified active set; certify every result with
the KKT residual and throw if it exceeds 1e-10. SCS.jl is a fallback selectable by keyword only.

## Rationale
- Closed forms are exact and need no tolerance.
- Clarabel is free (Apache-2.0), pure Julia, supports `MOI.ExponentialCone`, and is an interior-point
  method (more accurate than SCS's first-order ADMM).
- Interior-point tolerances (~1e-8) are not enough for the 1e-10 KKT target. On the active set
  A = {j : a_j > τ}, the problem is smooth with one equality (Σ a = 1); a few Newton steps
  converge quadratically. Variables are then re-checked for r_j ≤ 1 off the active set.
- A certificate catches silent solver failure, as Cover's update showed in E1.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Cover's multiplicative update as primary | Stalls on underflow (E1: false violation, KKT 1.29); kept as cross-check only |
| HiGHS | No exponential cones |
| SCS as primary | First-order method; lower accuracy; slower to 1e-10 |
| MOSEK | Proprietary (free academic licence only); nearest free alternative is Clarabel |
| Ipopt / Optim.jl on the smooth problem | Need log-barrier handling of V_s > 0 and simplex bounds; conic form is more robust |

## Consequences
- JuMP 1.32.1, MathOptInterface 1.54.0, Clarabel 0.11.1 and SCS 2.7.1 become dependencies;
  ForwardDiff 1.4.6 supplies Jacobians for the robust-bet Newton polish (DESIGN.md §5.6, §5.9).
- Robust bets are certified through the saddle-point property (log-optimal under the worst-case
  distribution); an alternating best-response polish was prototyped and rejected (did not converge).
- KKT residual definition (DESIGN.md §5.2) is the single source of truth for "optimal".
- Degenerate problems (several equally optimal allocations) still certify, but allocations may
  differ between solvers; parity tests compare growth and KKT, and allocations only on
  non-degenerate cases.
- If polishing fails to reach 1e-10, the call throws `CertificationError` rather than returning a
  weaker answer.
