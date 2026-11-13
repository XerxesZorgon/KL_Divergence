# ADR-016: Shortcut when caps sum to 1 without cash (near-point allocation set)
**Date:** 2026-11-11
**Status:** Accepted (approved by John 2026-11-11; closes DESIGN.md §8 OQ-4)

## Context
ADR-013 throws `ArgumentError` without cash when Σ_j u_j < 1 − 1e-12 and accepts larger sums. When
Σ_j u_j is within 1e-12 of 1, A_u = {a ≥ 0, Σa = 1, a ≤ u} is empty in exact arithmetic (Σu < 1) or a
set of width ≤ 2e-12 around u (Σu ≥ 1). A conic solver may then report the problem infeasible or return
an inaccurate point, although the answer is fixed by the caps alone.

## Decision
When the model has no cash and |Σ_j u_j − 1| ≤ 1e-12 (both sides), skip the solver and return
a = u / Σ_j u_j, certified by ρ_u (ADR-013). This applies in `optimal_allocation` and in
`kl_robust_allocation` (every ε ≥ 0, including the ε = 0 path). The result's method is `:single_point`.

## Rationale
- All caps are finite here (no cash, and any u_j ≥ 1 is normalized to ∞, which would make Σu infinite),
  so a = u/Σu is defined and every a_j > 0.
- Certification is immediate: a_j ≥ u_j − 1e-12·u_j ≥ u_j − τ_a, so every asset is in U, A = −∞ and the
  ν-part is 0; the cap overshoot (a_j − u_j)⁺ ≤ 1e-12 when Σu < 1; |Σa − 1| is rounding. So ρ_u ≲ 1e-12 ≤ 1e-10.
- Optimality: A_u has diameter ≤ 2e-12 in the 1-norm, so by the ADR-013 bound the growth gap is at most
  about 2e-12·max_j |r_j|. V_s(a) > 0 for every positive-probability scenario by DESIGN §3.1 rule 4
  (each such row has a positive entry, and every a_j > 0).
- KL-robust: the allocation set is the same near-point, so the robust bet is the same a for every ε; only
  the reported worst case depends on ε.
- The band matches the ADR-013 rejection threshold (1 − 1e-12), so there is no gap between "rejected"
  and "solved".

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Send the case to Clarabel | May report infeasible (Σu < 1) or lose accuracy on a set of width ~1e-12 |
| Reject Σu < 1 exactly | Rejects inputs that miss 1 only by rounding (e.g. caps from order-book sizes) |
| Shortcut only for Σu ≤ 1 | A set of width ≤ 2e-12 just above 1 has the same numerical problem |

## Consequences
- `BetResult.method` gains the value `:single_point` (DESIGN §3.2/§10.1); Python mirrors it as a string.
- `kl_robust_allocation` shortcut result: a = u/Σu; `(worst_case_growth, λ, Q) = kl_worst_case(m, a, ε)`;
  `kkt` = ρ_u of a under Q; `certificate = :kkt`. The KL-boundary check |D(Q‖π) − ε| ≤ tol is not applied
  (the bet does not depend on ε; when ε ≥ D_0 the worst case sits at λ = 0 with D < ε).
- Known risk: just outside the band (Σu = 1 + 2e-12 … 1 + 1e-9) the set is still very thin and goes to the
  conic path. T-BOX-16 (c) tests it; if it fails, a new ADR widens the band rather than loosening a
  tolerance.
