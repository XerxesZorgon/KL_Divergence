# ADR-012: Box-constrained allocation set (per-asset upper bounds)
**Date:** 2026-11-11
**Status:** Accepted (implementation gated on DESIGN.md §8 OQ-1…OQ-3 being resolved as ADRs)

## Context
v0.1 restricts allocations to the probability simplex Δ^J (ADR-003). AFS-GEM needs per-asset upper
bounds a_j ≤ u_j for (1) order-book depth, where each price level is its own asset and can absorb only
a limited stake, and (2) exits, where an existing position is an asset worth its liquidation bid that
can be held up to, but not beyond, its current size. Clipping a v0.1 allocation after the solve is not
optimal: the stake removed from a capped asset must be re-optimized across the others.

## Decision
Add an optional keyword `upper = nothing` (a length-J vector, entries in (0, ∞], cash entry ∞) to
`optimal_allocation`, `kl_robust_allocation`, `kl_cash_radius` and `kkt_residual`, store it in
`BetResult` / `RobustResult`, and optimize over

$$A_u=\{a\in\mathbb R^J:\ a\ge 0,\ \textstyle\sum_j a_j=1,\ a_j\le u_j\},$$

certified by the generalized KKT residual below.

## Rationale
- Option (a) leaves the ADR-003 data model (`ScenarioModel`), `mixture`, `predictive_from_draws`
  and every constructor untouched; caps describe the *market* (depth, position size), not the
  *beliefs* (π, X), so they belong with the solve, not the model.
- `upper = nothing` runs the v0.1 code path unchanged, so existing callers are byte-identical.
- A_u is convex and compact and wealth stays linear in a, so the theory of ADR-003/005 carries over:
  E1's bound L ≤ D(p‖q) holds for any such set, and the KL-robust saddle point exists (Sion's
  minimax theorem: A_u and the KL ball are convex and compact, E_Q[log V(a)] is concave in a and
  linear in Q).
- At fixed λ the KL dual objective −λ log E_π[V^{−1/λ}] does not depend on the allocation set, so
  the CRRA equivalence (γ = 1 + 1/λ*) still holds when maximized over A_u.

### Feasibility
A_u ≠ ∅ ⇔ cash is present (all-cash is feasible because cash is uncapped) or Σ_j u_j ≥ 1. Otherwise
throw `ArgumentError`. DESIGN §3.1 rule 4 (some a with V_s > 0) needs no change: every u_j > 0, so
any column that pays in scenario s can receive positive weight.

### Generalized KKT conditions
Let r_j(a) = Σ_{s: π_s>0} π_s X_sj / V_s(a) (marginal growth of asset j), and partition the assets into
the capped set U = {j : a_j = u_j}, the zero set Z = {j : a_j = 0} and the free set F (the rest).
With multiplier ν for Σa = 1, α_j ≥ 0 for a_j ≥ 0 and μ_j ≥ 0 for a_j ≤ u_j, stationarity is
r_j = ν − α_j + μ_j, which gives

$$r_j=\nu\ (j\in F),\qquad r_j\le\nu\ (j\in Z),\qquad r_j\ge\nu\ (j\in U).$$

ν is the common marginal growth of every asset that is free to move; μ_j = r_j − ν ≥ 0 is the growth
gained per unit of extra cap on asset j (its shadow price). Because Σ_j a_j r_j = Σ_s π_s = 1 for any a,

$$1=\sum_{j\in U}u_jr_j+\nu\Big(1-\sum_{j\in U}u_j\Big)\ \Longrightarrow\ \nu=\frac{1-\sum_{j\in U}u_jr_j}{1-\sum_{j\in U}u_j}.$$

With U = ∅, ν = 1 and the conditions are exactly v0.1's (r_j ≤ 1, r_j = 1 on the support). Intuition: a
cap stops investment in an asset whose marginal growth is still above the others'; the remaining money
goes to the free assets until their common marginal growth falls to ν < 1.

Residual (replaces DESIGN §5.2 when caps are supplied):

$$\rho_u(a)=\max\Big\{\max_{j\in F}|r_j-\nu|,\ \max_{j\in Z}(r_j-\nu)^+,\ \max_{j\in U}(\nu-r_j)^+,\ \max_j(-a_j)^+,\ \max_j(a_j-u_j)^+,\ \big|\textstyle\sum_j a_j-1\big|\Big\}.$$

Set membership uses the v0.1 activity tolerance τ_a = 1e-12: U = {j : u_j < ∞, a_j ≥ u_j − τ_a},
Z = {j ∉ U : a_j ≤ τ_a}, F = the rest. The polish (DESIGN §10.4) sets capped coordinates to u_j exactly.
The case F = ∅ (then Σ_U u_j = 1 and the formula for ν is 0/0) is open: DESIGN §8 OQ-1.

### Single-bet check (exact)
p = 0.6, gross payoff 2, cash, cap 0.1 on the bet (unconstrained Kelly f* = (pw − 1)/(w − 1) = 0.2).
At a = (0.9, 0.1): V = (1.1, 0.9); r_cash = 0.6/1.1 + 0.4/0.9 = 98/99 (free, so ν = 98/99 < 1);
r_bet = 0.6·2/1.1 = 12/11 ≥ ν (capped, μ = 12/11 − 98/99 = 10/99). Formula: ν = (1 − 0.1·12/11)/0.9 = 98/99. ✓

### Numerical evidence
| Check | Setup | Result | By / date |
|---|---|---|---|
| KKT sign pattern and ν range | 5 random cases, S = 8, J = 5, caps 0.05–0.3, cvxpy + Clarabel at 1e-12 | free r_j − ν = 0 to 8 printed decimals; zero-set r_j − ν ≤ 0; capped r_j − ν ≥ 0; ν ∈ [0.68, 0.96] | Claude, 2026-11-11 (brief) |
| Single bet | above, cvxpy 1.9.3 + Clarabel 0.11.1 | a = (0.9, 0.1); r = (0.98989899, 1.09090909) = (98/99, 12/11) | Claude, 2026-11-11 |
| Cash radius unchanged by caps | 19 random models with cash (S = 8, J = 5, bet caps U(0.02, 0.15)), joint capped KL program at ε = 0.9·D_min and 1.1·D_min | all-cash exactly when ε ≥ D_min: 19/19 (0 mismatches) | Claude, 2026-11-11 |
| F = ∅ example | no cash, π = (0.5, 0.3, 0.2), X = diag(3, 4, 6), u = (0.2, 0.3, 0.5) | a = u (only feasible point); r = (2.5, 1.0, 0.4); ν undefined (0/0) → OQ-1 | Claude, 2026-11-11 |

Clarabel at 1e-12 returned "solution may be inaccurate" warnings on some of these solves; the oracle's
accuracy at that tolerance is part of OQ-2.

### Cash radius (Unverified claim, to be tested by T-BOX-6)
Claim: `kl_cash_radius(m)` is unchanged by bet caps; the capped KL-robust bet is all cash iff ε ≥ D_min.
Proof sketch: at all-cash every bet has a_j = 0 < u_j, so no cap is active and the KKT conditions there
are the v0.1 ones (U = ∅, ν = 1). By the saddle point, all-cash is robust-optimal over A_u iff some Q in
the ball makes it log-optimal over A_u, i.e. E_Q[X_j] ≤ 1 for every bet, i.e. ε ≥ D_min. Status:
**Unverified** until reviewed and covered by T-BOX-6.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| (b) `upper` field in `ScenarioModel` | Changes the ADR-003 data model, its constructor and validation, `mixture` (which caps win when models are mixed?) and `predictive_from_draws`; mixes market limits with beliefs |
| Clip the v0.1 allocation to the caps, renormalize | Not optimal: removed stake is spread pro rata instead of re-optimized; can push other assets over their caps |
| Penalty / barrier term for cap violation | Approximate; cannot certify to 1e-10; adds a tuning knob |
| Represent depth by splitting an asset's payoff column without caps | Unconstrained splits are interchangeable; nothing limits each level's stake |
| General polyhedral constraints (G a ≤ h) now | Wider than AFS-GEM needs; residual and polish would need a general active-set method; can follow later with its own ADR |

## Consequences
- Public signatures gain `upper = nothing` (DESIGN §10.1). `BetResult` and `RobustResult` gain a field
  `upper::Union{Nothing, Vector{Float64}}`, mirrored in the Python dataclasses (ADR-007).
- `certify(result)` compares the stored residual, which was already computed with `result.upper`; it
  takes no `upper` keyword. The certification hook in the test suite must recompute with `result.upper`.
- Any finite cap routes `:single_event` problems to the conic path; the closed forms (greedy,
  full investment) ignore caps.
- Caps u_j ≥ 1 cannot bind (a_j ≤ 1 on Δ^J) and are normalized to ∞; if every normalized cap is ∞ the
  call is the v0.1 call (byte-identical). Finite caps that do not bind give the v0.1 allocation to the
  certification tolerance (≤ 1e-10), not bit-for-bit, because the conic program has extra constraints.
- Wasserstein-robust bets do not support caps in v0.2: passing any `upper !== nothing` throws
  `ArgumentError` (an explicit keyword is needed, because an unknown keyword would throw
  `MethodError`).
- Diagnostics (`growth_loss`, `growth_curve`, `zero_growth_point`, `simulate_ruin`) are unchanged in
  v0.2; whether any of them takes caps is DESIGN §8 OQ-3.
- Out of scope (unchanged from the brief): lower bounds other than 0; leverage or shorting; caps on
  cash; fixed costs or minimum stakes (non-convex, ADR-003).
- Future: the shadow prices μ_j = r_j − ν (value of one more unit of depth) are computed by the
  residual and could be returned; not part of v0.2.

## Amendment (ADR-013, ADR-014, ADR-015, 2026-11-11)
- The residual definition above (closed-form ν inside ρ_u, and "if U = ∅ use the v0.1 expression") is
  **superseded by ADR-013**: ν* is chosen by minimax, ν-part = max(0, (A − B)/2) with
  A = max_{F∪Z} r_j, B = min_{F∪U} r_j. The v0.1 expression is used only when `upper = nothing` (or every
  cap normalizes to ∞). The closed-form ν remains a test oracle when F ≠ ∅. This closes OQ-1.
- Feasibility without cash is now `ArgumentError` iff Σ_j u_j < 1 − 1e-12 (ADR-013).
- The oracle and its tolerances are fixed by ADR-014 (OQ-2); diagnostics stay uncapped by ADR-015 (OQ-3).
- Clarabel at 1e-12 leaves inactive bets at 2e-12–5e-12, so residuals computed on unpolished Clarabel
  output (as in the numerical evidence above) depend on the classification tolerance; see ADR-013
  evidence and ADR-014 τ_o.
