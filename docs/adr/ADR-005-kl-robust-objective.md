# ADR-005: KL-ball worst-case bet via the dual, solved as one exponential-cone program
**Date:** 2026-11-01
**Status:** Accepted

## Context
Article §7 needs a bet that maximizes worst-case growth over all distributions Q within KL
radius ε of the model P: max_a min_{Q : D(Q‖P) ≤ ε} E_Q[log V(a)]. The inner minimum over Q must be
removed to get a solvable problem, and the result must be checkable against an independent solve.

## Decision
Use the standard KL dual (Ben-Tal et al. 2013) for the inner problem,

$$\min_{Q:\,D(Q\|P)\le\varepsilon}\mathbb{E}_Q[f]=\sup_{\lambda>0}\Big\{-\lambda\log\mathbb{E}_P\big[e^{-f/\lambda}\big]-\lambda\varepsilon\Big\},\qquad f_s=\log V_s(a),$$

and solve the joint problem in (a, λ) as one exponential-cone program:

maximize t − λε subject to Σ_s π_s z_s ≤ λ, (t − w_s, λ, z_s) ∈ K_exp, (w_s, 1, V_s(a)) ∈ K_exp,
a ∈ Δ^J, λ ≥ 0, where K_exp = cl{(x, y, z) : y e^{x/y} ≤ z, y > 0}.

## Rationale
- Since e^{−f/λ} = V^{−1/λ}, the dual objective is −λ log E_P[V^{−1/λ}] − λε: at fixed λ* the bet
  maximizes CRRA utility with relative risk aversion γ = 1 + 1/λ*. This gives an independent oracle.
- For small ε the worst-case growth is ≈ E_P[log V] − √(2ε Var_P[log V]) (second oracle, asymptotic).
- Joint formulation is convex and fits the same Clarabel pipeline as ADR-004.
- Verified numerically on 2026-11-01 (random S = 6, J = 4, ε = 0.05, cvxpy + Clarabel): joint
  objective 0.2575362, direct inner solve 0.2575362, 1-D dual 0.2575362; CRRA bet with
  γ = 1 + 1/λ* matched the robust bet to ~1e-5 (solver tolerance).

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Nested solve (outer over a, inner over Q) | Slow, non-smooth outer problem, harder to certify |
| λ·Var penalty | A free risk knob with no data-driven radius; listed as an erratum of the parent article |
| Fractional Kelly | Excluded by the thesis (SDD Non-Goals) |

## Consequences
- A KL ball contains only Q ≪ P: it cannot add mass to outcomes P gives zero probability. Unseen
  outcomes require the Wasserstein ball (ADR-006). The docs must state this.
- ε must be supplied or estimated from data; the library offers `kl_radius_chi2(K, n; level)`
  (DESIGN.md §5.8) rather than a default value.
- Certification uses the saddle-point property: the robust bet a* must be log-optimal under the
  worst-case distribution Q* ∝ π·V(a*)^{−1/λ*}, with plain KKT residual ≤ 1e-10 under Q*
  (DESIGN.md §5.6).

## Amendment (Task 036–040, 2026-11-03)
Implemented as: cash test via the no-edge I-projection (`kl_cash_radius`: robust bet is all cash iff ε ≥ D_min = min{D(Q‖π): E_Q[X_j] ≤ 1 ∀ bets}), else conic seed + 1-D λ search with the CRRA polish. Prototype: 120/120 cases classified like the joint conic; 83/83 interior cases certified to 1.6e-14. v1 limit: λ* → 0 without cash-optimality throws `CertificationError`.

## Amendment (ADR-012, 2026-11-11)
With caps, the saddle-point certificate uses the generalized residual ρ_u (DESIGN.md §10.2) under Q*,
and the joint program adds a_j ≤ u_j. The CRRA equivalence holds unchanged over A_u. The cash test is
claimed unchanged (no cap is active at all-cash); this claim is Unverified until T-BOX-6 passes.


**Amendment 2026-11-13 (Fix 074a).** The λ search chained warm starts through `polish_crra` without checking its residual. A stalled polish at large k = 1/λ returned a near-riskless allocation with D ≈ 0, which drove the bisection to λ ≈ 1e-7 (model 84 of T-BOX-6, ε ≥ 0.7·D_min; true λ* = 0.04364, conic seed within 1e-4). v0.1 never returned an uncertified bet because the final check threw. Fix: if any intermediate polish has ρ > tol or a final check fails, rerun the search with every polish started from the conic seed or the last converged allocation, and a geometric bracket (×1.25) around the seed. Results where the original search succeeds are unchanged bitwise.
Test reference: robust value vs tight conic ≤ 1e-8 and vs `kl_worst_case` ≤ 1e-10; λ vs the conic only to 1e-4, since the conic's λ is noisy at ~1e-6 (flat objective) while λ* is pinned by the saddle and boundary certificates.
