# ADR-006: Finite-support Wasserstein-ball bet with L1 ground metric (resolves OQ-2)
**Date:** 2026-11-01
**Status:** Accepted

## Context
E7 compares Bayesian, KL-ball and Wasserstein-ball bets when the truth has an unseen tail outcome.
A Wasserstein ball can move probability mass to outcomes the model never saw, but the continuous
Mohajerin Esfahani–Kuhn (2018) form needs a ground metric on payoffs, and log-return distances break
for binary bets whose losing payoff is 0 (log 0 = −∞). Outline OQ-2 asked which metric to use.

## Decision
Use a finite candidate support: the S model scenarios plus T₀ user-supplied unseen scenarios
(rows of a matrix X̃ with T = S + T₀ rows). The transport cost is c_{st} = ‖X_{s,:} − X̃_{t,:}‖₁ by
default, or a user-supplied S×T cost matrix. Solve

maximize Σ_s π_s u_s − λε subject to u_s − λ c_{st} ≤ log V_t(a) for all s, t; a ∈ Δ^J; λ ≥ 0,

with each constraint written as (u_s − λ c_{st}, 1, (X̃ a)_t) ∈ K_exp.

## Rationale
- The inner problem min over transport plans γ ≥ 0 with Σ_t γ_{st} = π_s and Σ γ_{st} c_{st} ≤ ε is
  a linear program; its dual is max_{λ≥0} −λε + Σ_s π_s min_t (log V_t(a) + λ c_{st}). Strong
  duality holds (feasible LP), so the joint problem above is exact.
- It is one convex exponential-cone program: same Clarabel pipeline (ADR-004).
- No log(0) problem: the metric is on gross returns, and log only appears on wealth V_t(a) > 0.
- Unseen catastrophes enter explicitly as candidate rows, which is what E7 needs and what the
  article can explain.
- Verified numerically on 2026-11-01 (S = 6, J = 4, one unseen cash-only scenario, ε = 0.05):
  joint objective 0.2850014 = inner transport LP 0.2850014; the robust bet held 6.8% cash.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Continuous support, log-return metric | log 0 for losing binary bets; needs a payoff floor (an extra free knob) |
| Continuous support, L1 on returns (MEK) | −log(a·ξ) is convex in ξ, so the worst case is not tractable in general without support restrictions |
| Defer Wasserstein to v2 | Conflicts with brief scope and E7 |

## Consequences
- Results depend on which unseen scenarios the user lists; the API requires them explicitly (empty
  is allowed, then the ball only reshuffles mass among observed scenarios).
- Problem size is S·T exponential cones; v1 documents S·T ≤ ~10⁵.
- If any candidate scenario pays only cash, a large enough ε forces a positive cash holding; this is
  intended behavior and is tested.
- Choice of L1 is a modeling assumption; E7 reports one sensitivity run with L∞ to show its effect.
