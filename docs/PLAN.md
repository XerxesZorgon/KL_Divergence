# Project Plan
**Project:** Kelly_KL.jl  **Date:** 2026-11-01

Phase order (from the brief): core data model and exact solver (refactor E1 first, outputs
unchanged) → simultaneous/correlated bets → uncertain p and payoffs + model averaging → diagnostics
→ robust objectives → E2–E7 experiments → Python wrapper → reproduce-all command and reference outputs.

## Milestones
| # | Milestone | Gate Criterion | Depends On |
|---|---|---|---|
| M0 | Environment and golden E1 outputs | T0: exact `julia --version` (expected 1.12.x; stop and upgrade via juliaup if < 1.12) and Python version recorded below; `git init` in `code/`; original E1 scripts run on this machine produce `e1_results.csv` + `e1_kl_growth_loss.svg`, committed as golden files and tagged `e1-original`; `Project.toml` + `Manifest.toml` resolved with the Dependency Lock versions (any resolver change recorded) | — |
| M1 | Core data model, exact solver, certification; E1 refactored | `ScenarioModel`, `single_event`, `payoffs_from_odds`, greedy + full-investment solvers, `kkt_residual`, `certify`, `cover_update` implemented; E1 moved to `experiments/` and calling the package; T-ORC-1 (greedy path), T-ORC-2, T-ORC-8 (greedy side), T-E1-1, **T-E1-2 byte-identical**, T-E1-3 green | M0 |
| M2 | Simultaneous and correlated bets (conic + polish) | `simultaneous_bets`, `correlated_bets`, Clarabel/SCS conic path, Newton polish; T-ORC-1 (conic path), T-ORC-3, T-ORC-5, T-ORC-8, T-ERR-2 green | M1 |
| M3 | Uncertain p and payoffs, model averaging | `beta_predictive`, `dirichlet_predictive`, `predictive_from_draws`, `sampled_payoffs`, `growth_se`, `mixture`, `bma_weights`; T-ORC-12, T-ORC-13, T-ERR-3 green | M2 |
| M4 | Diagnostics | all §4.5 functions; T-DIA-1…6 green | M2 |
| M5 | Robust objectives | KL-ball and Wasserstein-ball bets with polish and saddle-point certification, `kl_radius_chi2`; T-ORC-4, 6, 7, 9, 10, 11, T-DIA-7 green | M2, M4 |
| M6 | Experiments E2–E7 | `experiments/plotting.jl` extracted (E1 SVG still byte-identical); E2–E7 scripts meet their pass criteria (TEST_PLAN §G); figures in `../img/` | M3, M4, M5 |
| M7 | Python wrapper | `python/kelly_kl` + `juliapkg.json`; shared JSON cases filled from Julia; T-PY-1…3 green | M5 |
| M8 | Reproduce-all and reference outputs | `reproduce_all.jl` + comparator; references generated on John's machine and committed; T-REP-1, T-REP-2, T-AQ-1 green; README/CHANGELOG updated | M6, M7 |

### Errata track (article content, not library)
| Task | Content | Status / evidence (2026-11-01) |
|---|---|---|
| T-ERR-1 | Source of the README example "(0.0667, 0.12)" and the claimed optimum (0.0745, 0.3725) | **Source not found**: not in XerxesZorgon/Extended-Kelly (README, `code/`, `test/` validation reports) nor in `Articles/2025-11-24-extended-kelly.mdx`. Observation (Unverified as the source): 0.0667 equals the single-bet gross-payoff Kelly fraction for (p, w) = (0.3, 4); the second single-bet fraction is 0.10, not 0.12. **Action:** John supplies the objective + inputs; then test T-ERR-1 is written. If no source by M2's end, drop this erratum. |
| T-ERR-2 | Parent two-bet example (0.0645, 0.0991) | Reproduced: (0.06450023479783634, 0.09915443806486002). Values match; the parent article prints 0.0991, a 4-decimal truncation — correctly rounded 0.0992 (minor presentation erratum, Task 020). |
| — | λ·Var penalty conflicts with thesis | Present in the parent article (Beta-uncertainty section, `max_b {E_P[g] − λ Var_P[g]}`). Erratum stands. |
| — | Half-Kelly simulations conflict with thesis | Present (code block building `res_half = 0.5 .* res_kelly.b`, "Half Kelly vs Full Kelly Battle" exercise). Relabel as baseline or remove. |
| T-ERR-3 | Candidate erratum: "weak prior reduces fractions by over 15%" | Not reproduced. With p ~ Beta(s·m, s(1−m)) and win ~ Bernoulli(p), each bet wins with probability m for every s, so the optimum is identical for s = 2 and s = 1000 (package: identical bets, ≤ 1e-12; total 0.163655). The reported 15.44% is consistent with Monte Carlo noise: the repo's objective re-samples with GLOBAL_RNG at every evaluation, and the article's Scenario A itself misses the exact bet by 0.0012. Erratum stands (Task 026); the parent code was not re-run. |

## Dependency Lock
Latest registered versions on 2026-11-01 (Julia General registry / PyPI). T0 records any change the
resolver makes on Julia 1.12; no other changes without a dedicated task.

| Package | Version | Kind |
|---|---|---|
| Julia | 1.12.1 (recorded T0, 2026-11-01) | runtime |
| JuMP | 1.32.1 | dep |
| MathOptInterface | 1.54.0 | dep |
| Clarabel | 0.11.1 | dep |
| SCS | 2.7.1 | dep |
| ForwardDiff | 1.4.6 | removed (Task 040) |
| Distributions | 0.25.132 | dep |
| StableRNGs | 1.1.0 | dep (experiments + tests) |
| Aqua | 0.8.18 | test |
| Random, LinearAlgebra, Printf, Statistics, TOML, Test | stdlib | dep / test |
| Python | 3.13.12 (recorded T0, 2026-11-01) | runtime |
| juliacall | 0.9.37 | Python dep |
| numpy | pin at M7 install (latest at install) | Python dep |
| pytest | pin at M7 install (latest at install) | Python test |

`[compat]` in Project.toml uses caret bounds at these versions (e.g. `JuMP = "1.32.1"`) plus
`julia = "1.12"`; the Manifest is the exact lock.

### Resolver changes (Task 004)
none

## Integration Checkpoints
- **End of every milestone:** full `Pkg.test()` (non-quick), Aqua, and — from M1 on — T-E1-2 byte
  identity re-run. John reviews the diff summary before the next milestone starts.
- **Every 5 tasks within a milestone:** quick suite (`KELLY_KL_QUICK=1`) and a `git log --oneline -5`
  review against `tasks.md`.
- **After M5:** review robust-bet polish failure rate on the T-ORC-4/9 random suites; any
  `CertificationError` triggers a design review (possible ADR) before M6.
- **After M6:** article draft check — every number quoted in OUTLINE sections 2–7 traces to an
  `experiments/output/` file.

## Definition of Done (Project)
Shipped ⇔ all of:
1. `julia --project=. experiments/reproduce_all.jl` exits 0 on John's machine (T-REP-1), with
   references committed and T-REP-2 passing.
2. `Pkg.test()` (full) exits 0, including oracle tests T-ORC-1…13, T-E1-1…3, T-DIA-1…7, T-ERR-2, T-ERR-3
   and T-ERR-1 (or its documented removal), with 0 uncertified bets.
3. `pytest python/tests` exits 0 (parity ≤ 1e-10).
4. README documents install, usage (Julia + Python), tests and reproduce-all; CHANGELOG has an entry
   per completed task; DESIGN.md §8 still empty.

## Handoff to tasks.md
- **Research summary:** E1 established that growth lost to a wrong belief equals D(p‖q) with full
  investment and is bounded by it otherwise; the library generalizes E1's exact, certified solver to a
  scenario/returns model with conic solving, robust (KL, finite-support Wasserstein) bets and
  diagnostics. Robust formulations and the parent two-bet example were checked numerically on
  2026-11-01.
- **Open questions:** source of the (0.0667, 0.12) errata pair (T-ERR-1, content only); Barron & Cover
  (1988) attribution and Thorp (2006) drawdown law still to verify for the article (E4 tests the law).
- **First atomic task (T0):** in `code/`, run `julia --version` and `python --version`, write both
  into this file's Dependency Lock table. Criterion: both cells filled; if Julia < 1.12, stop.
- **Phase to start from:** software-project Step 0 gate → M0 (no Spec Kit phases, per AGENTS.md Part 6).
- **Files to read first:** `docs/SDD.md`, `docs/DESIGN.md` (§3–§5), `docs/adr/ADR-003`, `ADR-004`,
  `ADR-009`, `e1_kl_growth_loss.jl`, `e1_plot.jl`, `E1_RESULTS.md`, `../OUTLINE.md`.
