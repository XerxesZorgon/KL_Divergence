# Test Plan
**Project:** Kelly_KL.jl  **Date:** 2026-11-01

## Test Types
| Type | Tool | Coverage Target | Who Runs It |
|---|---|---|---|
| Unit + oracle | Julia `Test` stdlib via `julia --project=. -e "using Pkg; Pkg.test()"` | every exported function has ≥ 1 test; line coverage of `src/` ≥ 85% (reported, not a gate alone) | Antigravity |
| Certification invariant | `test/certify_hook.jl`: every `BetResult` / `RobustResult` created in the suite is passed to `assert_certified` | 100% of computed bets | Antigravity |
| Package hygiene | Aqua.jl 0.8.18 (`Aqua.test_all(Kelly_KL)`) | whole package | Antigravity |
| Integration (experiments) | each `experiments/eN_*.jl` `main(quick=true)` called from `test/experiments_smoke.jl` | E1–E7 run without error in quick mode | Antigravity |
| End-to-end | `julia --project=. experiments/reproduce_all.jl` | all E1–E7 outputs vs `experiments/reference/` | Antigravity (John confirms on reference machine) |
| Python parity | `pytest python/tests` | every case in `test/cases/*.json` | Antigravity |

Random tests use fixed seeds (StableRNG for new tests; E1 tests keep Xoshiro, ADR-009).
`KELLY_KL_QUICK=1` shortens the stress tests for inner-loop runs; the full run is the gate.

## Test Catalog

### A. Oracle tests (Success Criterion 2)
| ID | What | Inputs | Pass criterion |
|---|---|---|---|
| T-ORC-1 | Classic single-bet Kelly | 1 000 random (p, w), gross payoff w ∈ (1, 50]; cash allowed | a_bet = (p w − 1)/(w − 1) if p w > 1 else 0; abs error ≤ 1e-12 (greedy) and ≤ 1e-10 (forced conic path) |
| T-ORC-2 | Cover–Thomas identity | 1 000 random (p, q, o), n ∈ 2:12, full investment | a_q = q exactly; G_p(q) = Σ p_i log o_i − H(p) − D(p‖q) within 1e-12; L = D(p‖q) within 1e-12 |
| T-ORC-3 | L ≤ D(p‖q) beyond single events | 2 000 random simultaneous (n ∈ 2:6) and correlated (S ≤ 64) models with cash, belief = tilted truth | 0 cases with L > D(p‖q) + 1e-10; all bets certified |
| T-ORC-4 | KL worst-case dual vs direct solve | 200 random (π, X, a, ε), ε ∈ [1e-4, 1] | (i) \|kl_worst_case − kl_inner_direct\| ≤ 1e-7 (direct solve is conic); (ii) dual's Q* satisfies \|D(Q*‖π) − ε\| ≤ 1e-10 and \|E_{Q*}f − value\| ≤ 1e-12 |
| T-ORC-5 | KKT residual on every bet | 500 random conic problems (S ≤ 256, J ≤ 12) + hook over whole suite | ρ ≤ 1e-10 (DESIGN §5.2) for all |
| T-ORC-6 | KL robust ⇔ CRRA | 100 random models, ε ∈ [1e-3, 0.5] | CRRA bet with γ = 1 + 1/λ* matches robust a* within 1e-8 (both certified) (also: CRRA bet from a uniform start equals the robust bet ≤ 1e-8; robust worst case ≥ that of random bets) |
| T-ORC-7 | Small-ε expansion | 20 models, ε ∈ {1e-4, 1e-5, 1e-6} | \|value − (E f − √(2ε Var f))\| / ε stays bounded (ratio between successive ε ≤ 2) |
| T-ORC-8 | Greedy = conic on single events with cash | 1 000 random non-degenerate single events | allocations agree within 1e-10 |
| T-ORC-9 | Wasserstein value consistency | 100 random models with 0–3 unseen rows | (i) `RobustResult.worst_case_growth` = `wasserstein_worst_case(a*)` within 1e-10; (ii) vs transport LP solved by Clarabel within 1e-7 (certificate tier reported; tier 2 allowed at gap ≤ 1e-7) |
| T-ORC-10 | Robust reduces to nominal at ε = 0 | 50 models | KL and Wasserstein bets equal `optimal_allocation` within 1e-9 |
| T-ORC-11 | Unseen catastrophe forces cash | model + one unseen cash-only row, ε large | Wasserstein bet has cash > 0; KL bet unaffected by the unseen row (KL ball cannot reach it) |
| T-ORC-12 | Posterior collapse (DESIGN §5.7) | Beta(sm, s(1−m)) for s ∈ {2, 10, 1000}, same mean m | identical bets (abs ≤ 1e-12) for independent bets |
| T-ORC-13 | Mixture linearity | 3 models, random weights | G_mixture(a) = Σ w_m G_m(a) within 1e-12 |

### B. Diagnostic tests
| ID | What | Pass criterion |
|---|---|---|
| T-DIA-1 | Classical ruin formula | equals solution of the gambler's-ruin linear system for p ∈ {0.4, 0.5, 0.55}, N ≤ 200, within 1e-12; p = 1/2 branch = 1 − a/N |
| T-DIA-2 | Drawdown law values | `drawdown_probability(x, 1) == x`; c = 2 → 1; c = 0.5 → x³ (exact arithmetic within 1e-15) |
| T-DIA-3 | Zero-growth point | small-edge binary bet (e = 0.01, w = 2): \|c₀ − 2\| < 0.05; c₀ is a root (\|G(c₀)\| ≤ 1e-12) |
| T-DIA-4 | Sample sizes | p = 0.5, w = 2.2, e = 0.1: n_detect(z = 1.96) = 1.96²·0.25·4.84/0.01 = 464.8 (within 1e-9); n_relative = 100·0.25·4.84/0.01 = 12 100 |
| T-DIA-5 | Log-score difference | analytic case with known p, π1, π2: expected value of `logscore_difference` equals D(p‖π2) − D(p‖π1) (exact expectation over outcomes, within 1e-12) |
| T-DIA-6 | Ruin simulator sanity | min_stake = 0, fixed_cost = 0 → Pr(ruin) = 0 for any certified Kelly bet with cash; even-money unit bets reproduce `ruin_probability_classical` within 3 SE (10⁵ trials) |
| T-DIA-7 | KL radius helper | `kl_radius_chi2(K, n)` = quantile(Chisq(K−1), level)/(2n) within 1e-14 |

### C. E1 tests
| ID | What | Pass criterion |
|---|---|---|
| T-E1-1 | E1 stress test (package code) | 50 000 trials, seed 7, n ∈ 2:12, σ ∈ [0,2], τ ∈ [0,1.5], ρ ∈ [0,0.3], c ∈ [0,0.2]: violations of L ≤ D(p‖q) + 1e-10 = 0; worst KKT ≤ 1e-10 (L/D is not tested: ill-conditioned when D is tiny — Task 014) |
| T-E1-2 | E1 refactor identity | refactored `experiments/e1_kl_growth_loss.jl` and `e1_plot.jl` produce `e1_results.csv` and `e1_kl_growth_loss.svg` byte-identical to the golden files made by the **original** scripts (git tag `e1-original`) on the same machine and Julia version |
| T-E1-3 | Cover update cross-check | `cover_update` agrees with greedy within 1e-8 on 1 000 non-degenerate cases; documented known stall case (from E1) reproduces kkt > 1e-3 (test that the cross-check is not trusted) |

### D. Errata re-verification tests
| ID | What | Pass criterion |
|---|---|---|
| T-ERR-1 | Parent README example "(0.0667, 0.12) vs numerical optimum (0.0745, 0.3725)" | **Blocked until source is supplied** (PLAN task T-ERR-1). Then: re-solve the stated objective with the package; test asserts the certified optimum and records whether (0.0745, 0.3725) and/or (0.0667, 0.12) is correct to 4 d.p. If no source exists, the test is removed and the erratum dropped. |
| T-ERR-2 | Parent two-bet example (regression) | (p, w) = (0.3, 4.0), (0.25, 6.0), gross payoffs, cash: certified bet rounds to the published (0.0645, 0.0991) at 4 d.p. (prototype 2026-11-01: 0.06450023, 0.09915443); full-precision value stored |
| T-ERR-3 | Parent "weak prior reduces fractions 15%" claim (candidate erratum) | Re-run the parent `kelly_general_bayes.jl` inputs (s_A = 1000 vs s_B = 2) with the package's predictive; report whether the reduction persists when posterior means are equal. Pass = test documents the observed reduction (value stored), not a fixed expectation |

### E. Python parity (Success Criterion 3)
| ID | What | Pass criterion |
|---|---|---|
| T-PY-1 | Shared cases | every case in `test/cases/*.json`: Python result vs Julia expected, abs ≤ 1e-10 on a, growth, worst-case growth, λ |
| T-PY-2 | Error mapping | invalid π (sum 0.9) → `ValueError`; forced certification failure → `kelly_kl.CertificationError` |
| T-PY-3 | Same Julia | `kelly_kl.julia_version()` equals the version recorded in PLAN.md |

### F. End-to-end (Success Criterion 1)
| ID | What | Pass criterion |
|---|---|---|
| T-REP-1 | Reproduce all | `julia --project=. experiments/reproduce_all.jl` exits 0; table shows PASS for every file of E1–E7 |
| T-REP-2 | Comparator catches changes | altering one byte of a reference SVG and one digit in a reference CSV → both FAIL, exit 1 |
| T-AQ-1 | Aqua | `Aqua.test_all(Kelly_KL)` passes |

### G. Experiment pass criteria (from OUTLINE.md, checked inside each experiment and in T-REP-1)
| Exp | Pass criterion |
|---|---|
| E1 | as T-E1-1, T-E1-2 |
| E2 | share of 1 000 replications whose ±2 SE interval covers the true ΔKL lies within 3 binomial SE of the normal-theory 95.45% (i.e. 93.47–97.43%) — Task 043: '≥ 95%' would fail about 1 run in 10 by chance (binomial SE 0.66%) |
| E3 | mean growth of BMA bet ≥ best single model bet across replications (paired difference > −2 SE) |
| E4 | (i) zero-growth point \|c0 − 2\| ≤ 0.01; (ii) simulated Pr(wealth ever falls to x) within 3 SE of x^{2/c−1} for c ∈ {0.5, 1, 1.5} and x ∈ {0.5, 0.3, 0.1}; rows x ∈ {0.9, 0.7} are reported, not tested — a 10–30% drawdown is only a few losing bets, so wealth overshoots x and the simulated probability is biased low (discreteness, Task 045) |
| E5 | minimum-stake ruin Monte Carlo vs classical formula within 3 SE (even-money unit bets) |
| E6 | 95% CI for the edge covers the true edge in 94–96% of 10 000 replications at n = n_detect |
| E7 | growth and ruin table (Bayesian, KL-ball, Wasserstein L1, Wasserstein L∞ sensitivity) across truth scenarios with an unseen tail; KL radius coverage of `kl_radius_chi2` within ±2 pp of level |

## Definition of Pass
All of the following, on John's machine with the pinned Manifest:
1. `Pkg.test()` exits 0 with the full (non-quick) stress test; every A–D test green (T-ERR-1 green
   or formally removed per PLAN).
2. Certification hook reports 0 bets with ρ > 1e-10.
3. `pytest python/tests` exits 0 (T-PY-1..3).
4. `reproduce_all.jl` exits 0 (T-REP-1) and T-REP-2 passes.
5. Aqua passes; coverage of `src/` ≥ 85% is reported (a drop below triggers review, not automatic failure).

## Failure Protocol
Per AGENTS.md Part 8 (Diagnostic Protocol): Antigravity **stops** making changes, reports the
failing test ID and the **verbatim** terminal output (including seed, Julia version, and for
numeric failures the observed value, expected value, tolerance and KKT residual), shows
`git diff HEAD~1`, and waits for a named root cause. One targeted fix per approved diff; if it fails,
`git revert` and return to step 1. A `CertificationError` is never "fixed" by loosening the
tolerance — tolerance changes require an ADR. A reference-output mismatch is never fixed by
regenerating references inside the same task.
