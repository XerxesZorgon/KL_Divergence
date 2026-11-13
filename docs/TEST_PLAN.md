# Test Plan
**Project:** Kelly_KL.jl  **Date:** 2026-11-01

## Test Types
| Type | Tool | Coverage Target | Who Runs It |
|---|---|---|---|
| Unit + oracle | Julia `Test` stdlib via `julia --project=. -e "using Pkg; Pkg.test()"` | every exported function has ≥ 1 test; line coverage of `src/` ≥ 85% (reported, not a gate alone) | Antigravity |
| Certification invariant | `test/certify_hook.jl`: every `BetResult` / `RobustResult` created in the suite is passed to `assert_certified` | 100% of computed bets | Antigravity |
| Package hygiene | Aqua.jl 0.8.18 (`Aqua.test_all(Kelly_KL)`) | whole package | Antigravity |
| Integration (experiments) | each `experiments/eN_*.jl` `main(quick=true)` called from `test/test_experiments_smoke.jl` | E2–E7 run without error in quick mode and write every output (E1 is covered by test_e1.jl) | Antigravity |
| End-to-end | `julia --project=. experiments/reproduce_all.jl` | all E1–E7 outputs vs `experiments/reference/` | Antigravity (John confirms on reference machine) |
| Python parity | `pytest python/tests` | every case in `test/cases/cases.toml` | Antigravity |

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
| T-ERR-1 | Parent README example "(0.0667, 0.12) vs numerical optimum (0.0745, 0.3725)" | **Removed (Task 056).** No source for the README pair (0.0667, 0.12) was found by M2's end, so by the PLAN rule the test is not written and the erratum is dropped from the article. |
| T-ERR-2 | Parent two-bet example (regression) | (p, w) = (0.3, 4.0), (0.25, 6.0), gross payoffs, cash: certified bet rounds to the published (0.0645, 0.0991) at 4 d.p. (prototype 2026-11-01: 0.06450023, 0.09915443); full-precision value stored |
| T-ERR-3 | Parent "weak prior reduces fractions 15%" claim (candidate erratum) | Re-run the parent `kelly_general_bayes.jl` inputs (s_A = 1000 vs s_B = 2) with the package's predictive; report whether the reduction persists when posterior means are equal. Pass = test documents the observed reduction (value stored), not a fixed expectation |

### E. Python parity (Success Criterion 3)
| ID | What | Pass criterion |
|---|---|---|
| T-PY-1 | Shared cases | every case in `test/cases/cases.toml`: Python result vs Julia expected, abs ≤ 1e-10 on a, growth, worst-case growth, λ |
| T-PY-2 | Error mapping | invalid π (sum 0.9) → `ValueError`; forced certification failure → `kelly_kl.CertificationError` |
| T-PY-3 | Same Julia | `kelly_kl.julia_version()` equals the version recorded in PLAN.md |

### F. End-to-end (Success Criterion 1)
| ID | What | Pass criterion |
|---|---|---|
| T-REP-1 | Reproduce all | `julia --project=. experiments/reproduce_all.jl` exits 0; table shows PASS for every file of E1–E7 — a separate command, not part of Pkg.test() (Task 055) |
| T-REP-2 | Comparator catches changes | altering one byte of a reference SVG and one digit in a reference CSV → both FAIL, exit 1 |
| T-AQ-1 | Aqua | `Aqua.test_all(Kelly_KL; stale_deps = (ignore = [:StableRNGs],))` passes (StableRNGs is used only by experiments/tests; Task 056) |

### G. Experiment pass criteria (from OUTLINE.md, checked inside each experiment and in T-REP-1)
| Exp | Pass criterion |
|---|---|
| E1 | as T-E1-1, T-E1-2 |
| E2 | share of 1 000 replications whose ±2 SE interval covers the true ΔKL lies within 3 binomial SE of the normal-theory 95.45% (i.e. 93.47–97.43%) — Task 043: '≥ 95%' would fail about 1 run in 10 by chance (binomial SE 0.66%) |
| E3 | mean growth of BMA bet ≥ best single model bet across replications (paired difference > −2 SE) |
| E4 | (i) zero-growth point \|c0 − 2\| ≤ 0.01; (ii) simulated Pr(wealth ever falls to x) within 3 SE of x^{2/c−1} for c ∈ {0.5, 1, 1.5} and x ∈ {0.5, 0.3, 0.1}; rows x ∈ {0.9, 0.7} are reported, not tested — a 10–30% drawdown is only a few losing bets, so wealth overshoots x and the simulated probability is biased low (discreteness, Task 045) |
| E5 | (a) minimum-stake ruin Monte Carlo vs the classical formula within 3 SE for p ∈ {0.48, 0.50, 0.52}, a ∈ {2, 5, 10, 15, 18}, N = 20 (even-money unit bets); (b) Kelly bettor with a minimum stake: Pr(wealth falls below minimum stake / b*, after which every Kelly stake is below the minimum) within 3 SE of the drawdown law x = (1/b*)/w0 for w0 ∈ {40, 100, 200} units (Task 046). SE is computed at the formula's probability; (b) uses 4 000 trials per start (Fix 046a). |
| E6 | 95% CI for the edge covers the true edge in 94–96% of 10 000 replications at n = n_detect |
| E7 | growth / Pr(halved) / median-final table for Kelly, KL-robust (ε = kl_radius_chi2, n ∈ {200, 1 000, 5 000}), Wasserstein L1 and L∞ (crash as unseen scenario, radius = 1% crash mass) across q ∈ {0, 0.005, 0.01, 0.02, 0.05}. Pass: (i) each Wasserstein bet's growth under the truth ≥ its certified worst-case growth − 1e-9 whenever the truth is inside its ball (q ≤ 0.01); (ii) the same for each KL bet at q = 0; (iii) KL-radius coverage D(π‖p̂) ≤ kl_radius_chi2(4, n) within ±2 pp of 95% for n ∈ {200, 1 000}; n = 50 reported only — the χ² radius is asymptotic and its coverage there (≈ 93.3%) sits on the band edge (Task 048) |

### H. Box constraints (v0.2, ADR-012, DESIGN §10) — added 2026-11-11
Residual ρ_u is DESIGN §10.2 (minimax ν*, ADR-013). "Oracle" means the committed fixtures
`test/oracle/box_oracle.toml` from the offline cvxpy 1.9.3 + Clarabel 0.11.1 generator (ADR-014): read by
Julia tests, no Python at test time. Fixture cases carry a `degenerate` flag (strict complementarity with
margin 1e-6 fails). Tolerances follow ADR-014 and change only through an ADR. Random tests use StableRNG with
fixed seeds. *(Amended 2026-11-11 for ADR-013…015; the earlier "(OQ-2)" markers are now fixed values.)*

| ID | What | Inputs | Pass criterion |
|---|---|---|---|
| T-BOX-1 | Single bet, cap binds | p = 0.6, gross payoff 2, cash, `upper = [Inf, 0.1]` (unconstrained f* = 0.2) | `method == :clarabel` (conic dispatch, not greedy); bet stake `== 0.1` exactly and cash within 1e-15 of 0.9; ν = 98/99 within 1e-12; r_bet = 12/11 within 1e-12; ρ_u ≤ 1e-10 |
| T-BOX-2 | Capped problems vs fixtures | every `kind = "allocation"` fixture (generated as: 200 models, S ≤ 64, J ≤ 10, with and without cash, caps U(0.02, 0.3) on a random subset of bets, feasible by construction, ≥ 1 binding cap per case) | library ρ_u ≤ 1e-10 for all; \|G − G_oracle\| ≤ 1e-10 + 3·ρ_u(oracle) for all (ADR-014 amendment); on non-degenerate fixtures also \|a − a_oracle\|∞ ≤ 1e-8 and library U/Z/F (τ_a = 1e-12) == stored U/Z/F (τ_o = 1e-8); degenerate fixtures: growth only |
| T-BOX-3 | Minimax ν* | T-BOX-2 fixtures + library results | (i) every free j: \|r_j − ν*\| ≤ 1e-10; (ii) non-degenerate fixtures: \|ν* − ν*_oracle\| ≤ 1e-8; (iii) closed-form ν (test oracle, F ≠ ∅ and Σ_F a_j ≥ 0.01) agrees with ν* within 1e-9; (iv) ν* < 1 whenever a cap binds with r_j > 1; (v) hand cases: A = −∞ ⇒ ν* = B, B = +∞ ⇒ ν* = A, ν-part 0 when A or B infinite |
| T-BOX-4 | No caps / non-binding caps | 100 models from T-ORC-5 and 100 single events | (i) `upper = nothing`, `upper = fill(Inf, J)` and caps all ≥ 1 give results `===`-identical to v0.1 (a, growth, kkt, method); (ii) finite caps 0.05 above the v0.1 stake (and < 1) give \|a − a_v0.1\|∞ ≤ 1e-10, U = ∅, \|ν* − 1\| ≤ 1e-9 (minimax residual, ADR-013: not bitwise 1) |
| T-BOX-5 | Residual reduces to v0.1 | 500 random (X, a, π), a on and off the optimum | `kkt_residual(X, a, π; upper = nothing)` and with all-`Inf` caps are bitwise equal to the v0.1 expression (copied into the test as a frozen reference function) |
| T-BOX-6 | Cash radius unchanged by caps (tests the Unverified claim of ADR-012) | 100 random models with cash, bet caps U(0.02, 0.3), D_min finite and > 1e-3 | `kl_cash_radius(m; upper) === kl_cash_radius(m)`; `kl_robust_allocation(m, 1.1·D_min; upper)` is all cash (certified); at 0.9·D_min cash < 1 − 1e-6 |
| T-BOX-7 | KL-robust bet with binding caps | every `kind = "kl"` fixture (generated as: 100 random models with cash, ε ∈ [1e-3, 0.5], ε < D_min, ≥ 1 cap binding at the result) | saddle ρ_u(a*) under Q* ≤ 1e-10; \|D(Q*‖π) − ε\| ≤ 1e-10; worst-case growth vs the `kind = "kl"` fixtures within 1e-7 (as T-ORC-4; ADR-014); capped CRRA bet with γ = 1 + 1/λ* equals a* within 1e-8 (as T-ORC-6) |
| T-BOX-8 | Invalid caps and feasibility threshold (ADR-013) | (a) no cash, Σu_j = 0.9 and Σu_j = 1 − 2e-12; (b) u_j = 0, u_j = −0.1, u_j = NaN; (c) wrong length; (d) finite cap on cash; (e) no cash, Σu_j = 1 − 5e-13 | (a)–(d) each throw `ArgumentError`, (a) message names Σ u_j; (e) validation does **not** throw; the solve uses the ADR-016 shortcut (T-BOX-16) |
| T-BOX-9 | Wasserstein rejects caps | any model, `upper` = any vector (including all `Inf`) | `wasserstein_robust_allocation(m, 0.1; upper)` throws `ArgumentError` (not `MethodError`) |
| T-BOX-10 | Polish bound handling | (a) warm start with a coordinate at its cap whose capped optimum is interior; (b) a capped model where an unheld asset has ν < r_j < 1 at the start | (a) the coordinate is released (a_j < u_j − 1e-9 at the result); (b) the asset is added (a_j > 0); both ρ_u ≤ 1e-10 |
| T-BOX-11 | Projection onto A_u | 1 000 random x ∈ ℝ^J, random feasible u | result in A_u (feasibility terms of ρ_u ≤ 1e-15); idempotent; equals clamp.(x .− θ, 0, u) with θ found by an independent bisection in the test (the projection's exact characterization) within 1e-12 — no Python oracle needed |
| T-BOX-12 | Free set empty (ADR-013) | no cash, π = (0.5, 0.3, 0.2), X = diag(3, 4, 6), u = (0.2, 0.3, 0.5), a = u | U = {1, 2, 3}, F = Z = ∅; A = −∞, B = 0.4; ρ_u == 0; ν* == 0.4 (within 1e-15); no division performed |
| T-BOX-13 | Python parity with caps | capped `allocation` and `kl` cases added to `test/cases/cases.toml` by `scripts/fill_cases.jl` | pytest: abs ≤ 1e-10 on a, growth, worst-case growth, λ; `upper=None` default; invalid caps → `ValueError` |
| T-BOX-14 | No regression (Success Criterion 1) | full suite, no caps | full `Pkg.test()` green; T-E1-2 byte-identical; `reproduce_all.jl` with `KELLY_KL_REFERENCE_HOST=1` ends `reproduce_all: PASS — 8 experiments, 23 files`; pytest parity ≤ 1e-10 on all pre-existing cases |
| T-BOX-15 | Fixture integrity (ADR-014) | `test/oracle/box_oracle.toml` | header rejection rate ≤ 10%; every stored case recomputed in Julia has oracle ρ_u ≤ 1e-8 at τ_o = 1e-8 (allocation) or saddle ρ_u ≤ 1e-8 and \|D(Q*‖π) − ε\| ≤ 1e-8 (kl); stored `degenerate` flags reproduce from the stored a, r and ν* at margin 1e-6; header records seed and cvxpy/clarabel/Python versions; header `requirements_sha256` equals the SHA-256 of the committed `test/oracle/requirements.txt` (SHA stdlib) |
| T-BOX-16 | Near-point shortcut (ADR-016) | no cash, J = 4, caps scaled so Σu ∈ {1 − 1e-12, 1 − 5e-13, 1, 1 + 5e-13, 1 + 1e-12}; plus Σu ∈ {1 + 2e-12, 1 + 1e-9} and Σu = 1 − 2e-12 | (a) in the band: `optimal_allocation` returns method `:single_point`, a == u ./ sum(u) bitwise, ρ_u ≤ 1e-11; (b) in the band: `kl_robust_allocation` for ε ∈ {0, 0.01, 10} returns the same a bitwise, `certificate == :kkt`, kkt ≤ 1e-11, `worst_case_growth == kl_worst_case(m, a, ε)[1]` bitwise; (c) outside the band (1 + 2e-12, 1 + 1e-9): conic path (method `:clarabel`), ρ_u ≤ 1e-10 — a failure here triggers a new ADR, not a tolerance change; (d) Σu = 1 − 2e-12 → `ArgumentError` |

Note on T-BOX-14: the brief counts "21 E1–E7 reference outputs". `experiments/reference/` holds 23 files on
2026-11-11 (22 from E1–E7 plus `e8_demere.csv`), and Task 063's criterion is "8 experiments, 23 files". The
criterion above uses the current count.

## Definition of Pass
All of the following, on John's machine with the pinned Manifest:
1. `Pkg.test()` exits 0 with the full (non-quick) stress test; every A–D test green (T-ERR-1 removed per PLAN).
2. Certification hook reports 0 bets with ρ > 1e-10.
3. `pytest python/tests` exits 0 (T-PY-1..3).
4. `reproduce_all.jl` exits 0 (T-REP-1) and T-REP-2 passes.
5. Aqua passes; coverage of `src/` ≥ 85% is reported (a drop below triggers review, not automatic failure).
6. (v0.2) T-BOX-1…16 green; the certification hook recomputes every capped result with `result.upper`
   and reports 0 results with ρ_u > 1e-10.

## Failure Protocol
Per AGENTS.md Part 8 (Diagnostic Protocol): Antigravity **stops** making changes, reports the
failing test ID and the **verbatim** terminal output (including seed, Julia version, and for
numeric failures the observed value, expected value, tolerance and KKT residual), shows
`git diff HEAD~1`, and waits for a named root cause. One targeted fix per approved diff; if it fails,
`git revert` and return to step 1. A `CertificationError` is never "fixed" by loosening the
tolerance — tolerance changes require an ADR. A reference-output mismatch is never fixed by
regenerating references inside the same task.
