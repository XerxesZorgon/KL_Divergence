# E1 — Growth lost to a wrong belief vs. D(p‖q)

Run from `code/`: `julia --project=. experiments/e1_kl_growth_loss.jl` (writes `experiments/output/e1_results.csv`, prints the summary and the stress test), then `julia --project=. experiments/e1_plot.jl` (writes `experiments/output/e1_kl_growth_loss.svg` and copies it to `../img/`). Both call the Kelly_KL package (exact greedy solver; every bet certified, full KKT residual ≤ 1e-10). Reference outputs: git tag `e1-original`, Julia 1.12.1 on Windows 11; the summary is identical to the original Julia 1.13.3 (Linux) run at printed precision. Seeds 20261101 (main), 7 (stress), 1 (plot subsample).

## Result (resolves OQ-1)

| Case | Trials | max \|L − D\| | L ≤ D | median L/D |
|---|---|---|---|---|
| Full investment, no fees | 3000 | 3.3e-16 | 100% | 1.000 |
| Full investment, margin + commission | 3000 | 2.8e-16 | 100% | 1.000 |
| Cash, fair book (Σ1/ô = 1) | 3000 | 2.0e-15 | 100% | 1.000 |
| Cash, superfair book (Σ1/ô < 1) | 3000 | 1.1e-15 | 100% | 1.000 |
| Cash, 10% margin | 3000 | 0.19 | 100% | 0.80 |
| Cash, 10% margin + 5% commission | 3000 | 0.18 | 100% | 0.69 |
| Stress (cash, n 2–12, wide σ, τ, ρ, c) | 50 000 | — | 100% (0 violations) | max L/D = 1.000000 (6 decimals; the exact maximum is 1 + 1.3e-7, at a trial with D = 1.45e-9 — rounding error in L amplified by 1/D. The tested bound is L ≤ D + 1e-10; Task 014.) |

Note:

* $\tau$ is market error
* $\rho$ is overround
* $c$ is commission

All allocations certified optimal: worst KKT residual 4e-12.

**Reading:** D(p‖q) is exact whenever cash is never optimal (full investment, or a fair /
superfair book) and fees only rescale payoffs. When holding cash is optimal (subfair book), the
loss is strictly smaller than D — D(p‖q) is an **upper bound**, not the value.

A wrong belief produced negative growth G_p(a_q) < 0 in 1326 / 3000 (margin) and 1399 / 3000
(margin + commission) trials — the bridge to the ruin section.

## Why the bound holds in general (proof sketch — check before publishing)

Let the allocation set 𝒜 be convex, wealth V_i(a) > 0 linear in a for each outcome (or joint
scenario) i, and a_q = argmax_{a∈𝒜} Σ_i q_i log V_i(a). First-order optimality of the concave
problem gives, for every a ∈ 𝒜,

  Σ_i q_i V_i(a) / V_i(a_q) ≤ 1.

Then, with a_p the truth-optimal allocation,

  L = Σ_i p_i log[V_i(a_p)/V_i(a_q)]
    = Σ_i p_i log[(q_i/p_i) · V_i(a_p)/V_i(a_q)] + D(p‖q)
    ≤ log Σ_i q_i V_i(a_p)/V_i(a_q) + D(p‖q)        (Jensen)
    ≤ D(p‖q).

Equality needs V_i(a_p)/V_i(a_q) ∝ p_i/q_i, which happens with full investment.

**What this covers:** cash, linear fees and commissions, simultaneous bets, and correlated bets
(take i to range over joint scenarios), plus continuous outcomes with sums replaced by integrals.
**What it does not cover:** fixed per-bet costs and minimum stakes, which make 𝒜 non-convex.

Prior art (Unverified, needs checking): Barron & Cover (1988), "A bound on the financial value of
information," IEEE Trans. Inf. Theory 34(5), appears to contain this result.

## Solver note

Cover's multiplicative update (kept in the file as a cross-check) stalled on 1 / 20 000 stress
trials because a weight underflowed to 0, which produced a false "violation" (L/D = 1.0004,
KKT residual 1.29). The main results use the exact greedy solution (Kelly 1956; Smoczynski &
Tomkins 2010), checked against the KKT conditions.
