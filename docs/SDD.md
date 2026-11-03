---
{
  "id": "file_spfisqj8",
  "filetype": "document",
  "filename": "SDD",
  "created_at": "2026-11-03T21:36:57.097Z",
  "updated_at": "2026-11-03T21:36:57.097Z",
  "meta": {
    "location": "/"
  }
}
---
# Software Description Document

**Project:** Kelly_KL.jl (module `Kelly_KL`) **Version:** 0.1 **Date:** 2026-11-01 **Author:** John Peach / eurAIka

## 1. Purpose

Extended Kelly sizing needs the full distribution of win probabilities and payoffs, but no free library computes these bets exactly, reports how much growth an error in the distribution costs, or protects against parts of the model that are missing. Kelly_KL.jl is a reusable Julia library, callable from Python, that fills that gap. It is needed now because *Extended Kelly II: When the Distribution Is Wrong* (experiments E1–E7) and John's betting and investment projects both depend on it.

## 2. Users and Use Cases

| User | Use Case | Priority |
|---|---|---|
| John (research) | Reproduce every number and figure in *Extended Kelly II* with one command | High |
| John (betting / investing projects) | Size single, simultaneous and correlated bets exactly from a scenario distribution, from Julia or Python | High |
| John (model checking) | Measure growth lost to a wrong distribution, compare models by holdout log score, check ruin / drawdown risk and data needs | High |
| Wild Peaches readers | Run the article's experiments and try their own distributions | Med |
| Python users (future projects) | Call the same solvers through a thin `kelly_kl` Python package (juliacall) | Med |

## 3. Key Features

 1. **Scenario model.** One data model for every case: scenario probabilities π (length S) and a returns matrix X (S scenarios × J assets, cash is an asset); wealth V(a) = X a.
 2. **Exact single-event solver.** Closed-form greedy Kelly allocation with cash for mutually exclusive outcomes (Kelly 1956; Smoczynski & Tomkins 2010), plus the full-investment identity a = q.
 3. **General log-optimal solver.** Simultaneous, correlated and sampled-payoff bets solved as an exponential-cone program (Clarabel via JuMP), polished by Newton steps, and certified by a KKT residual ≤ 1e-10.
 4. **Uncertain probabilities.** Beta / Dirichlet posteriors converted to the exact predictive scenario distribution (closed form where possible, Monte Carlo otherwise).
 5. **Uncertain payoffs.** Sampled payoff distributions turned into scenarios (sample-average approximation) with a reported Monte Carlo standard error of growth.
 6. **Model averaging.** Mixture predictive π = Σ\_m w_m π\_m over a common scenario set, with helpers for BMA weights from log marginal likelihoods.
 7. **Diagnostics.** Growth loss vs D(p‖q); holdout log-score comparison; growth curve and its zero-growth point (exact and the 2b\* approximation); drawdown law x^{2/c−1} (diagnostic only); classical minimum-stake ruin formula and a Monte Carlo ruin / drawdown simulator; sample sizes n_detect and n_10%.
 8. **Robust bets.** KL-ball worst-case bet (dual form, solved jointly as an exponential-cone program) and finite-support Wasserstein-ball worst-case bet (mass may move to user-supplied unseen scenarios).
 9. **Python wrapper.** `kelly_kl` Python package over juliacall, with parity tests against Julia.
10. **Reproduce-all command.** `julia --project=. experiments/reproduce_all.jl` regenerates E1–E7 outputs and compares them with stored reference outputs.

## 4. Non-Goals

- Live data feeds, bookmaker or broker connections, and placing bets.
- Domain-specific models (horse-racing features, news signals, etc.).
- A dashboard or graphical interface.
- Fractional Kelly as a sizing option. It appears only as a comparison baseline inside experiments, never as an exported sizing function or keyword argument.
- Optimizing with fixed per-bet costs or minimum stakes (non-convex). These are only simulated, in the ruin tools.
- Allocation sets other than the probability simplex (no leverage, no short selling) in v1.
- Registering the package in Julia's General registry.

## 5. Success Criteria

1. **Reproduce-all:** `julia --project=. experiments/reproduce_all.jl` runs E1–E7 on the pinned Manifest and exits 0 only if every output matches its stored reference (rules in ADR-009).
2. **Oracle tests pass** (`julia --project=. -e "using Pkg; Pkg.test()"`): classic single-bet Kelly formula; Cover–Thomas identity; L ≤ D(p‖q); KL worst-case dual vs a direct solve of the inner problem; KKT residual ≤ 1e-10 for every computed bet (exception: tied Wasserstein worst cases are certified by a duality gap ≤ 1e-7 — ADR-006 amendment, decided 2026-11-03).
3. **Python parity:** `pytest python/tests` passes; every shared case agrees with Julia to ≤ 1e-10 (absolute, on allocations and growth).
4. **E1 refactor:** the package-based E1 produces byte-identical `e1_results.csv` and `e1_kl_growth_loss.svg` to the original scripts run on the same machine and Julia version.

## 6. Constraints

- **Language:** Julia, compat `julia = "1.12"`; development machine Windows 11 (Dell XPS 8930), Julia 1.12.x (exact patch recorded in task T0). E1 was first verified on Julia 1.13.3 Linux.
- **Dependencies:** free, open-source, registered Julia packages only, pinned by `Manifest.toml`(PLAN.md Dependency Lock). Python side: juliacall from PyPI.
- **Numerics:** every reported allocation is certified (KKT residual ≤ 1e-10). Cover's multiplicative update may only be used as a cross-check.
- **Thesis constraint:** the bet comes exactly from the distribution; no free risk knobs (no fractional multiplier, no λ·Var penalty) in the sizing API.
- **Licensing:** MIT, same as the parent repository XerxesZorgon/Extended-Kelly (`license.txt`).
- **Timeline:** no fixed date; milestones gated by tests (PLAN.md).

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Clarabel's default tolerance (\~1e-8) cannot reach KKT ≤ 1e-10 | High | High | Newton polish on the identified active set after the conic solve (ADR-004); certify, never trust solver status alone |
| Julia RNG streams differ between 1.12 and 1.13, so E1 numbers change | High | Med | Reference outputs generated on the pinned Julia version; E1 identity is checked against the original script on the same version; E2–E7 use StableRNGs (ADR-009) |
| Floating-point differences across OS / libm break byte comparison | Med | Med | Byte comparison only on the reference machine; elsewhere numeric tolerance 1e-12 relative (ADR-009) |
| Drawdown law x^{2/c−1} or attribution to Thorp (2006) is wrong | Low | Med | Marked "verify"; E4 tests it by simulation before the article uses it |
| Barron & Cover (1988) prior-art attribution for L ≤ D(p‖q) is wrong | Med | Low | Article cites it only after the paper is checked; library docstring cites E1 proof sketch |
| Errata item "(0.0667, 0.12) vs (0.0745, 0.3725)" has no traceable source | High | Low | Task asks John for the source; if none, the item is dropped (PLAN.md §Errata) |
| juliacall install on Windows fails or picks a different Julia | Med | Med | Python package pins the Julia project path via `juliapkg.json`; parity tests run in CI-like local script |
| Joint Newton polish for robust bets fails on degenerate problems (ties in the Wasserstein min, λ\* → 0) | Med | Med | Explicit point-mass / λ\* = 0 branches; otherwise throw `CertificationError` with the best residual (DESIGN.md §6) |
| Wasserstein program grows as S·T exponential-cone constraints | Med | Low | v1 targets article-sized problems (S·T ≤ \~10⁵); documented limit |
