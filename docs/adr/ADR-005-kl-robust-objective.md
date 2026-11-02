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
