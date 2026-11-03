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

### Changed
- Tasks 012–013, 015: E1 experiment, plot and results moved to `experiments/` and refactored to call Kelly_KL; outputs byte-identical to `e1-original`
- Task 014: T-E1-1 no longer tests the L/D ratio (ill-conditioned for tiny D)
- Task 040: robust methods documented as implemented (λ search + CRRA polish; two-tier Wasserstein certificate, ADR-006 exception)

### Removed
- Task 040: ForwardDiff dependency

### Fixed
- Task 021: polish no longer zeroes a weight needed for feasibility, and drops a weight a Newton step would push below zero (ratio test) instead of stalling
- Task 041: `kl_cash_radius` — projected Newton with line search; infinite radius now proved, not inferred

### Existing before this changelog
- E1 experiment (standard library only, Julia 1.13.3 Linux): `e1_kl_growth_loss.jl`, `e1_plot.jl`,
  `e1_results.csv`, `E1_RESULTS.md`, `../img/e1_kl_growth_loss.svg`. To be moved to `experiments/`
  and refactored onto the package in milestone M1 with byte-identical outputs.
