# Changelog
All notable changes to Kelly_KL.jl are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Antigravity appends one entry per
completed task (AGENTS.md Part 7).

## [Unreleased]
### Added
- [Docs] Planning documents: `docs/SDD.md`, `docs/adr/ADR-001` … `ADR-011`, `docs/DESIGN.md`,
  `docs/TEST_PLAN.md`, `docs/PLAN.md`, `README.md`, `CHANGELOG.md` (2026-11-01).
- Tasks 001–004: git repository, golden E1 outputs (tag `e1-original`), pinned environment (Julia 1.12.1)
- Task 005: test harness and Aqua
- Task 006: `ScenarioModel`, `single_event`, `payoffs_from_odds`, `growth`
- Task 007: `BetResult`, `CertificationError`, `SolverError`, `kkt_residual`, `certify`
- Task 008: exact full-investment and greedy allocations (T-ORC-1 greedy, T-ORC-2)
- Task 009: certified `optimal_allocation` for single events
- Task 010: `kl`
- Task 011: `cover_update` cross-check (T-E1-3)
- Task 014: E1 regression tests T-E1-1, T-E1-2
- Task 016: `simultaneous_bets`, `correlated_bets`
- Task 017: exponential-cone solver `solve_conic` (Clarabel; SCS by keyword)
- Task 018: Newton polish on the active set
- Task 019: `optimal_allocation` general path (conic + polish + certify) and `method = :auto | :conic`
- Task 020: oracle tests T-ORC-1 (conic), T-ORC-8, T-ERR-2
- Task 021: oracle tests T-ORC-3 (L ≤ D beyond single events), T-ORC-5
- Task 023: `beta_predictive`, `dirichlet_predictive`, `mixture`, `bma_weights` (T-ORC-12, T-ORC-13)
- Task 024: `predictive_from_draws` (dependent posteriors)
- Task 025: `sampled_payoffs`, `growth_se` (payoff uncertainty)
- Task 026: T-ERR-3 errata test — the parent article's weak-prior reduction is not reproduced
- Task 028: `growth_loss`, `holdout_logscore`, `logscore_difference` (T-DIA-5)
- Task 029: `growth_curve`, `zero_growth_point` (T-DIA-3)
- Task 030: `drawdown_probability`, `effective_multiple`, `ruin_probability_classical`, `n_detect`, `n_relative` (T-DIA-1, 2, 4)
- Task 031: `simulate_ruin` with `fixed_stake`, `target` and `min_fraction` (T-DIA-6)
- Task 033: `kl_worst_case` (1-D dual), `kl_inner_direct` (oracle) (T-ORC-4, T-ORC-7)
- Task 034: `polish_crra` (CRRA-weighted polish; log-optimal is k = 0)
- Task 035: `kl_cash_radius` — KL distance to the nearest no-edge world
- Task 036: `RobustResult`, `kl_robust_allocation` (T-ORC-6, T-ORC-10)
- Task 037: `wasserstein_worst_case` (exact breakpoint scan) (T-ORC-9 ii)
- Task 038: `wasserstein_robust_allocation` with two-tier certificate (T-ORC-9 i, T-ORC-10, T-ORC-11)
- Task 039: `kl_radius_chi2` (T-DIA-7)
- Task 042: experiments/plotting.jl — shared SVG plot, CSV and run-metadata helpers
- Task 043: E2 — holdout log-score difference estimates ΔKL (coverage 94.8%)
- Task 044: E3 — model averaging beats the best single model at n = 10, 50, 200
- Task 045: E4 — overbetting cliff (zero growth at c0 = 1.997) and drawdown law x^(2/c−1)
- Task 046: E5 — minimum-stake ruin vs the classical formula; Kelly bettor "priced out" by a minimum stake
- Task 047: E6 — edge-CI coverage at n_detect; Kelly-stake spread vs sample size
- Task 048: E7 — Kelly vs KL-robust vs Wasserstein-robust bets with an unseen crash; KL-radius coverage
- Task 049: test/test_experiments_smoke.jl — E2–E7 run in quick mode
- Task 050: python/ — kelly_kl wrapper over the whole public API via juliacall (dataclass results, error mapping)
- Task 051: shared parity cases test/cases/cases.toml (26 cases), scripts/fill_cases.jl, test/test_cases.jl
- Task 052: python/tests — pytest parity T-PY-1…3 and API coverage
- Task 053: README Python section; PLAN Python pins
- Task 054: experiments/compare.jl — reference comparator (T-REP-2)
- Task 055: experiments/reproduce_all.jl, update_reference.jl; committed experiments/reference/ (21 files); .gitattributes
- Task 056: Aqua stale-dependency check re-enabled (StableRNGs exempt)

### Changed
- Tasks 012–013, 015: E1 experiment, plot and results moved to `experiments/` and refactored to call Kelly_KL; outputs byte-identical to `e1-original`
- Task 014: T-E1-1 no longer tests the L/D ratio (ill-conditioned for tiny D)
- Task 040: robust methods documented as implemented (λ search + CRRA polish; two-tier Wasserstein certificate, ADR-006 exception)
- Tasks 043, 045, 046, 048: TEST_PLAN §G criteria for E2, E4, E5, E7 made precise (binomial bands, tested ranges, SE at the hypothesised value)
- Task 051: shared cases are TOML, not JSON (no extra dependency); Python ≥ 3.11
- Task 055: experiment smoke test calls main through invokelatest (Julia 1.12 world age)

### Removed
- Task 040: ForwardDiff dependency
- Task 056: T-ERR-1 (no source for the (0.0667, 0.12) README pair; erratum dropped)

### Fixed
- Task 021: polish no longer zeroes a weight needed for feasibility, and drops a weight a Newton step would push below zero (ratio test) instead of stalling
- Task 041: `kl_cash_radius` — projected Newton with line search; infinite radius now proved, not inferred
- Fix 050a: wrapper venv on Python 3.14 (bundled OpenSSL 3.5) instead of editing juliacall's juliapkg.json

### Existing before this changelog
- E1 experiment (standard library only, Julia 1.13.3 Linux): `e1_kl_growth_loss.jl`, `e1_plot.jl`,
  `e1_results.csv`, `E1_RESULTS.md`, `../img/e1_kl_growth_loss.svg`. To be moved to `experiments/`
  and refactored onto the package in milestone M1 with byte-identical outputs.
