# ADR-013: Budget multiplier ν by minimax in the capped KKT residual
**Date:** 2026-11-11
**Status:** Accepted (approved by John 2026-11-11; closes DESIGN.md §8 OQ-1; supersedes the residual section of ADR-012)

## Context
ADR-012 defined the capped residual ρ_u with the closed-form multiplier
ν = (1 − Σ_{j∈U} u_j r_j)/(1 − Σ_{j∈U} u_j). That formula is 0/0 when the free set F is empty (then
Σ_U u_j = 1), and it is ill-conditioned when little stake is free. Example: no cash,
π = (0.5, 0.3, 0.2), X = diag(3, 4, 6), u = (0.2, 0.3, 0.5): A_u = {u}, r = (2.5, 1.0, 0.4), and ν is
undefined although a = u is trivially optimal.

## Decision
Use the ν that best satisfies the KKT inequalities, computed without division:

- Sets as in ADR-012, τ_a = 1e-12: U = {j : u_j < ∞, a_j ≥ u_j − τ_a}, Z = {j ∉ U : a_j ≤ τ_a}, F = the rest;
  r_j = Σ_{s: π_s>0} π_s X_sj / V_s(a).
- A = max_{j∈F∪Z} r_j (max ∅ = −∞), B = min_{j∈F∪U} r_j (min ∅ = +∞).
- ν-part of the residual = max(0, (A − B)/2), or 0 if A or B is infinite.
- Reported ν* = (A + B)/2 if both are finite; A if B = +∞; B if A = −∞.
- ρ_u(a) = max{ν-part, max_j(−a_j)⁺, max_j(a_j − u_j)⁺, |Σ_j a_j − 1|}.
- `upper = nothing` (or every cap normalized to ∞, ADR-012) keeps the v0.1 residual and code path
  unchanged. With any finite cap the minimax residual is used, also when U = ∅ at the solution.
- Feasibility without cash: `ArgumentError` if Σ_j u_j < 1 − 1e-12; otherwise accepted. (What the
  solver path does for 1 − 1e-12 ≤ Σ u_j ≤ 1, where A_u is empty or a single point, is DESIGN §8 OQ-4.)
- The closed-form ν is kept only as a test oracle, used when F ≠ ∅.

## Rationale
- **Exactness.** KKT holds iff some ν satisfies r_j = ν on F, r_j ≤ ν on Z and r_j ≥ ν on U, i.e. iff
  A ≤ B. The ADR-012 ν-terms equal max(A − ν, ν − B, 0), whose minimum over ν is max(0, (A − B)/2),
  reached at ν = (A + B)/2. So the ν-part is the smallest possible violation, not one particular choice.
- **No division, no threshold.** F = ∅ is handled by the infinities (example above: A = −∞, B = 0.4,
  ν-part 0, ν* = 0.4).
- **Agreement.** If F ≠ ∅ and a is optimal, A = B = the common r_j on F, which is the closed-form ν (the
  identity Σ_j a_j r_j = 1 fixes it). Single bet (p = 0.6, payoff 2, cap 0.1): ν* = 98/99 by both.
- **Meaning of the number** (derived by Claude 2026-11-11 from concavity; reviewer to confirm). For any
  b ∈ A_u, G(b) − G(a) ≤ Σ_j r_j(b_j − a_j) = Σ_j (r_j − ν*)(b_j − a_j) when Σa = Σb = 1. With δ = ν-part:
  |r_j − ν*| ≤ δ on F; on Z, r_j − ν* ≤ δ and b_j − a_j ≥ −τ_a; on U, r_j − ν* ≥ −δ and b_j − a_j ≤ τ_a. Hence

  $$\max_{b\in A_u}G(b)-G(a)\ \le\ 2\delta+\tau_a\sum_{j\in Z\cup U}|r_j-\nu^*| .$$

  A certified ρ_u ≤ 1e-10 therefore means growth within about 2e-10 of the capped optimum. The same
  bound holds for the v0.1 residual (ν = 1 is one admissible choice).

### Evidence
| Check | Result | By / date |
|---|---|---|
| 5 random capped cases (S = 8, J = 5), cvxpy + Clarabel at 1e-12, allocations clipped to caps | minimax ν matched formula ν within 5e-10; residual ≤ 5.8e-10 | Claude, 2026-11-11 (brief) |
| F = ∅ example above | ρ_u = 0, ν* = 0.4 | Claude, 2026-11-11 (brief; re-run, same) |
| Single bet, a = (0.9, 0.1) | ρ_u = 0, ν* = 0.98989898989899 = 98/99 = formula ν | Claude, 2026-11-11 |
| 30 random capped cases, raw (unpolished) Clarabel output, τ_a = 1e-12 | 7/30 had ρ_u 0.005–0.066: Clarabel leaves inactive bets at 2e-12–5e-12, which τ_a = 1e-12 classes as free. With τ = 1e-8: 29/30 ≤ 1e-8 | Claude, 2026-11-11 |

The last row is why τ_a = 1e-12 is only for polished allocations (the polish sets inactive coordinates
to exactly 0 or u_j) and why the oracle uses its own τ_o (ADR-014).

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Closed-form ν (ADR-012) with a threshold on 1 − Σ_U u_j | Needs an arbitrary threshold; ill-conditioned near it; undefined at F = ∅ |
| ν = weighted mean Σ_F a_j r_j / Σ_F a_j | Same 0/0 at F = ∅; not the minimal violation |
| Minimax ν also when `upper = nothing` | Would change v0.1 residuals bit-for-bit (ν = 1 is hard-coded there); breaks Success Criterion 1 |

## Consequences
- `kkt_components` returns (ρ_u, ν*, U, Z, F); the polish add/release rules (DESIGN §10.4) compare with ν*.
- With finite caps and U = ∅, ν* is within about ρ_u of 1, not bitwise 1 (T-BOX-4).
- Every computed bet's certificate keeps the same meaning: ρ ≤ 1e-10 bounds the growth gap by ≈ 2e-10.
- Applies unchanged under any weights summing to 1 (Q* in the KL saddle certificate, CRRA weights in
  `polish_crra`).

## Amendment (ADR-016, 2026-11-11)
OQ-4 is closed by ADR-016: without cash and |Σ_j u_j − 1| ≤ 1e-12, the solver is skipped and
a = u/Σ_j u_j is returned (method `:single_point`), in `optimal_allocation` and `kl_robust_allocation`.
