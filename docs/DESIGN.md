# Software Design Document
**Project:** Kelly_KL.jl  **Version:** 0.1  **Date:** 2026-11-01

Notation used throughout: S scenarios indexed s; J assets indexed j (cash, when present, is
asset 1 with return 1 in every scenario); π ∈ Δ^S scenario probabilities; X ∈ ℝ_{≥0}^{S×J} gross
returns; a ∈ Δ^J allocation (fractions of current wealth, summing to 1); V_s(a) = (Xa)_s wealth
multiple in scenario s; G_π(a) = Σ_s π_s log V_s(a) expected log growth (nats per bet).
Payoffs are **gross** (ADR-003): a winning unit stake at payoff w returns w.

## 1. Architecture Overview

```
                 ┌──────────────────────────── src/ (module Kelly_KL) ───────────────────────────┐
 inputs          │                                                                                 │
 (p, w, odds,    │  model.jl ──► ScenarioModel(π, X) ◄── uncertainty.jl (posteriors, sampled        │
  posteriors,    │      │                                payoffs, mixtures)                         │
  samples)  ───► │      ▼                                                                          │
                 │  optimize.jl  optimal_allocation(model)                                         │
                 │      ├── exact.jl    closed forms (a = q; greedy with cash)                     │
                 │      └── conic.jl    JuMP + Clarabel (SCS fallback) ──► polish.jl (Newton)      │
                 │                │                                                                │
                 │                ▼                                                                │
                 │  certify.jl  kkt_residual ≤ 1e-10 or throw CertificationError ──► BetResult     │
                 │                                                                                 │
                 │  robust.jl   KL ball, Wasserstein ball (conic.jl + polish.jl + certify.jl)       │
                 │  diagnostics.jl  growth loss, KL, log score, growth curve, drawdown, ruin, n    │
                 │  crosscheck.jl   Cover multiplicative update (tests only)                       │
                 └─────────────────────────────────────────────────────────────────────────────────┘
        ▲                                   ▲                                      ▲
        │ juliacall                         │ using Kelly_KL                       │ using Kelly_KL
 python/kelly_kl (thin wrapper)     test/ (Pkg.test)                  experiments/ e1…e7, plotting.jl,
 python/tests (pytest parity)       test/cases/*.json (shared) ──►    reproduce_all.jl ──► output/ vs reference/
                                                                      figures ──► ../img/*.svg
```

## 2. Module / Component Breakdown
| Component | Responsibility | Interface |
|---|---|---|
| `src/Kelly_KL.jl` | Module definition, includes, exports | `using Kelly_KL` |
| `src/model.jl` | `ScenarioModel` type, validation, constructors for single events, simultaneous bets, correlated tables | §4.1 |
| `src/exact.jl` | Closed-form allocations: full investment (a = q), greedy with cash | §4.2 |
| `src/conic.jl` | JuMP models for log-optimal, KL-robust and Wasserstein-robust problems; solver selection | internal: `solve_conic(problem; solver)` |
| `src/polish.jl` | Newton polishing on the active set (log-optimal) and on the smooth saddle-point system (robust) | internal: `polish_log_optimal`, `polish_kl`, `polish_wasserstein` |
| `src/certify.jl` | KKT residuals, `CertificationError`, `certify` | §4.3 |
| `src/optimize.jl` | Dispatch exact vs conic; build `BetResult` | §4.2 |
| `src/uncertainty.jl` | Beta / Dirichlet predictives, predictive from posterior draws, sampled payoffs, mixtures, BMA weights | §4.4 |
| `src/diagnostics.jl` | Growth, KL, growth loss, holdout log score, growth curve, zero-growth point, drawdown law, classical ruin, ruin simulator, sample sizes | §4.5 |
| `src/robust.jl` | KL-ball and Wasserstein-ball bets, inner worst-case evaluators, KL radius helper | §4.6 |
| `src/crosscheck.jl` | Cover (1984) multiplicative update, never used for returned results | §4.7 |
| `experiments/plotting.jl` | Stdlib SVG helper extracted from `e1_plot.jl` (ADR-008) | `svg_scatter`, `svg_lines`, `svg_table` |
| `experiments/eN_*.jl` | One script per experiment E1–E7; each has `main(; seed, out)` | writes `output/eN_*.csv|svg|_meta.toml` |
| `experiments/reproduce_all.jl` | Runs E1–E7, compares to `reference/`, prints PASS/FAIL, exit code | `julia --project=. experiments/reproduce_all.jl` |
| `python/kelly_kl/` | One-to-one Python wrappers over the public API (ADR-007) | §4.8 |

## 3. Data Model

### 3.1 `ScenarioModel`
```julia
struct ScenarioModel
    π::Vector{Float64}          # length S; π_s ≥ 0; |Σπ − 1| ≤ 1e-12
    X::Matrix{Float64}          # S × J; X_sj ≥ 0; finite
    assets::Vector{String}      # length J; unique names; "cash" if present
    has_cash::Bool              # true ⇒ X[:,1] == ones(S) and assets[1] == "cash"
    structure::Symbol           # :single_event, :simultaneous, :correlated, :sampled, :mixture
end
```
Validation (constructor throws `ArgumentError` with the failing rule):
1. size(X, 1) == length(π) ≥ 1, size(X, 2) == length(assets) ≥ 1;
2. π_s ≥ 0, |Σ_s π_s − 1| ≤ 1e-12 (stored as given, not renormalized — keeps E1 bit-identical; Task 006);
3. all X_sj finite and ≥ 0;
4. **feasibility:** some a ∈ Δ^J has V_s(a) > 0 for every s with π_s > 0 (checked by
   a = uniform: V_s(uniform) > 0 ⇔ every positive-probability row has a positive entry). If it fails,
   throw `DomainError` — every allocation risks total loss with positive probability, so G = −∞.

`structure = :single_event` means each non-cash asset pays in exactly one scenario (mutually
exclusive outcomes); `exact.jl` relies on this.

### 3.2 Results
```julia
struct BetResult
    a::Vector{Float64}          # allocation, Σa = 1, a ≥ 0
    growth::Float64             # G_π(a)
    kkt::Float64                # residual (§5.2), ≤ 1e-10
    method::Symbol              # :full_investment, :greedy, :clarabel, :scs
    active::Vector{Int}         # indices with a_j > τ_a
    assets::Vector{String}
end

struct RobustResult
    a::Vector{Float64}
    worst_case_growth::Float64  # min_{Q ∈ ball} E_Q[log V(a)]
    nominal_growth::Float64     # G_π(a)
    λ::Float64                  # dual multiplier of the ball constraint
    Q::Vector{Float64}          # worst-case distribution (length S for KL, T for Wasserstein)
    support::Matrix{Float64}    # X (KL) or X̃ (Wasserstein) rows on which Q lives
    kkt::Float64                # residual of a under Q (§5.6), ≤ 1e-10
    ball::Symbol                # :kl, :wasserstein
    ε::Float64
end

struct CertificationError <: Exception
    kkt::Float64; tol::Float64; method::Symbol; msg::String
end
```

### 3.3 Shared test case file (`test/cases/*.json`)
```json
{ "id": "two_bets_parent_article",
  "kind": "simultaneous",               // single_event | simultaneous | correlated | kl | wasserstein | diagnostic
  "inputs": { "p": [0.3, 0.25], "w": [4.0, 6.0], "cash": true },
  "expected": { "a": [ … ], "growth": … },  // filled from Julia reference run (T-M7)
  "tol": 1e-10 }
```

## 4. API / Interface Specification
All functions validate inputs and throw `ArgumentError` (bad shape/values), `DomainError`
(infeasible: G = −∞ for every allocation), `SolverError` (solver did not reach an optimal or
almost-optimal status) or `CertificationError` (§6).

### 4.1 Model construction
| Function | Inputs | Output | Notes |
|---|---|---|---|
| `ScenarioModel(π, X; assets, has_cash, structure)` | as §3.1 | `ScenarioModel` | validated |
| `single_event(q, ohat; cash=true)` | q ∈ Δ^n, gross payoffs ô ∈ ℝ_{>0}^n | S = n, J = n + cash | matches E1 `returns_matrix` |
| `payoffs_from_odds(o; commission=0.0)` | decimal odds o, commission c ∈ [0,1) | ô = 1 + (o − 1)(1 − c) | E1 payoff model |
| `simultaneous_bets(p, w; cash=true)` | win probs p ∈ [0,1]^n, gross payoffs w > 0, independent | S = 2^n rows (bit k of s−1 = bet k wins), π_s = Π_k p_k^{b_k}(1−p_k)^{1−b_k} | throws for n > 16 |
| `correlated_bets(P_joint, W)` | S × 1 probabilities and S × n gross payoff table | `ScenarioModel` with cash | user supplies joint table |

### 4.2 Optimization
```julia
optimal_allocation(m::ScenarioModel; solver = :clarabel, tol = 1e-10) -> BetResult
optimal_allocation(q, ohat; cash::Bool)                         -> BetResult   # single event shortcut
```
Dispatch: `structure == :single_event` and `!has_cash` → a = q (Cover–Thomas); `:single_event` with
cash → greedy (§5.1); otherwise conic (§5.3) + polish (§5.4). `solver ∈ (:clarabel, :scs)`. Always
certified (§5.2). `growth(m, a)` and `growth(X, a, π)` return G.

### 4.3 Certification
```julia
kkt_residual(X, a, π; τ_a = 1e-12) -> Float64
certify(result; tol = 1e-10)       -> result  # throws CertificationError
```

### 4.4 Uncertain probabilities, payoffs, model averaging
| Function | Inputs | Output |
|---|---|---|
| `beta_predictive(α, β)` | vectors of Beta posterior parameters for independent binary events | p̄ = α ./ (α .+ β) (exact predictive, §5.7) |
| `dirichlet_predictive(α)` | Dirichlet parameters | α ./ sum(α) |
| `predictive_from_draws(Pdraws, w; cash=true)` | M × n matrix of posterior draws of win probabilities (dependent posteriors), gross payoffs w | `ScenarioModel` with π_s = (1/M) Σ_m Π_k p_{mk}^{b_k}(1−p_{mk})^{1−b_k} |
| `sampled_payoffs(p, samplers, M; rng, cash=true)` | win probs p, one payoff sampler per bet (any `Distributions.Sampleable` or function `rng -> w`), M draws | `ScenarioModel` with S = M·2^n rows, π_s = (1/M)·Π p…; structure `:sampled` |
| `growth_se(m, a)` | sampled model | Monte Carlo standard error of G across the M payoff draws |
| `mixture(models, weights)` | models with identical X and assets; weights ∈ Δ | `ScenarioModel` with π = Σ w_m π_m, structure `:mixture` |
| `bma_weights(logml)` | log marginal likelihoods | softmax weights (computed with log-sum-exp) |

### 4.5 Diagnostics
| Function | Formula / behavior |
|---|---|
| `kl(p, q)` | D(p‖q) = Σ_i p_i log(p_i/q_i), 0·log 0 = 0, +Inf if p_i > 0 = q_i |
| `growth_loss(m_true, a_belief)` | L = G_p(a_p) − G_p(a_q), with a_p certified optimal under p |
| `holdout_logscore(π, y)` | mean and SE of log π_{y_i} over held-out outcome indices y |
| `logscore_difference(π1, π2, y)` | paired mean and SE of log π1_{y_i} − log π2_{y_i}; estimates D(p‖π2) − D(p‖π1) |
| `growth_curve(m, a⋆, cs)` | G(c) = G_π(c·a⋆_bets, cash = 1 − c·Σ a⋆_bets) for each c; −Inf when some V_s ≤ 0; requires cash |
| `zero_growth_point(m, a⋆)` | (c₀ exact root of G(c) = 0 with c₀ > 1 by bisection to 1e-14, c_quad = 2 from the quadratic approximation) |
| `drawdown_probability(x, c)` | x^{2/c − 1} for 0 < x < 1, 0 < c < 2; 1.0 for c ≥ 2 (continuous-limit law, Thorp 2006 — verify in E4) |
| `effective_multiple(e_belief, e_true)` | c ≈ e_belief / e_true (small-edge continuous approximation) |
| `ruin_probability_classical(p, a, N)` | ((q/p)^a − (q/p)^N)/(1 − (q/p)^N), q = 1 − p; 1 − a/N when p = 1/2; integer 0 ≤ a ≤ N; ±1-unit even-money steps only |
| `simulate_ruin(m, a; rng, trials, rounds, w0, min_stake, lot, fixed_cost, fixed_stake = nothing, target = Inf)` | Monte Carlo: stakes = a·W rounded down to multiples of `lot` (or `fixed_stake` on the first bet); bets below `min_stake` skipped; `fixed_cost` charged per placed bet; ruin when W ≤ 0 or W < min_stake; stops at ruin, `target`, or `rounds`. Returns (Pr ruin, SE, drawdown quantiles, final-wealth quantiles, `min_fraction`) (Task 031: needed to check the classical gambler's-ruin formula, T-DIA-6, and the drawdown law, E4) |
| `n_detect(p, w; z = 1.96)` | z² p(1−p) w² / e², e = p w − 1 |
| `n_relative(p, w; rel = 0.10)` | p(1−p) w² / (rel·e)²; rel = 0.10 gives 100 p(1−p) w²/e² |

### 4.6 Robust bets
```julia
kl_robust_allocation(m, ε; solver = :clarabel, tol = 1e-10)               -> RobustResult
kl_worst_case(m, a, ε)              -> (value, λ, Q)    # 1-D dual (§5.6), no conic solver
kl_inner_direct(m, a, ε)            -> (value, Q)       # direct conic solve over Q (test oracle)
kl_radius_chi2(K, n; level = 0.95)  -> ε                # §5.8
wasserstein_robust_allocation(m, ε; unseen = zeros(0, J), cost = :l1, tol = 1e-10) -> RobustResult
wasserstein_worst_case(m, a, ε; unseen, cost)        -> (value, λ, Q)   # exact 1-D dual (§5.9)
```
`ε` is required (ADR-011). `cost ∈ (:l1, :linf)` or an S × T matrix. `unseen` rows must have J columns
and non-negative entries.

### 4.7 Cross-check
`cover_update(X, π; tol = 1e-11, maxiter = 2_000_000) -> (a, kkt, iters)` — E1's `log_optimal`,
unchanged. Not exported; used only in tests.

### 4.8 Python wrapper (`kelly_kl`)
One function per exported Julia function, same names and argument order, NumPy arrays in and out,
results as frozen dataclasses mirroring `BetResult` / `RobustResult`. Julia exceptions are re-raised
as `ValueError` (ArgumentError, DomainError) or `kelly_kl.CertificationError` / `kelly_kl.SolverError`.

## 5. Key Algorithms

### 5.1 Exact single event with cash (Kelly 1956; Smoczynski & Tomkins 2010)
Sort outcomes by expected return e_i = q_i ô_i (descending). Grow the bet set B while e_i > R(B), with
reserve rate

$$R(B)=\frac{1-\sum_{i\in B}q_i}{1-\sum_{i\in B}1/\hat o_i},$$

(R = 0 if Σ_B q_i = 1 or Σ_B 1/ô_i ≥ 1). Stake f_i = q_i − R/ô_i on i ∈ B, cash 1 − Σ f_i. Intuition: keep
adding the best remaining outcome while its expected return beats what the cash-plus-current-bets
portfolio earns on it. Full investment without cash: a = q for any ô (Cover & Thomas Thm 6.1.2).

### 5.2 KKT residual (single source of truth for "optimal")
For max_{a ∈ Δ^J} G_π(a), define r_j(a) = Σ_{s: π_s>0} π_s X_sj / V_s(a) (marginal growth of asset j).
Since Σ_j a_j r_j = Σ_s π_s = 1, the KKT conditions are r_j ≤ 1 for all j and r_j = 1 where a_j > 0. Residual

$$\rho(a)=\max\Big\{\max_j (r_j-1),\ \max_{j:\,a_j>\tau_a}|r_j-1|,\ \max(0,-\min_j a_j),\ \big|\textstyle\sum_j a_j-1\big|\Big\}.$$

E1 used only the first term; the full residual adds the support and feasibility terms. Pass ⇔ ρ ≤ 1e-10.

### 5.3 Conic log-optimal program
maximize Σ_s π_s t_s subject to (t_s, 1, (Xa)_s) ∈ K_exp, a ∈ Δ^J, where
K_exp = cl{(x, y, z): y e^{x/y} ≤ z, y > 0}; the constraint means t_s ≤ log V_s(a). Solved by Clarabel
0.11.1 through JuMP with default settings; SCS 2.7.1 on request.

### 5.4 Newton polish (log-optimal)
Start from the conic solution; active set A = {j: a_j > 1e-6}. Repeat (≤ 50 iterations): with
g = X_Aᵀ(π ./ V), H = −X_Aᵀ diag(π ./ V²) X_A, solve

$$\begin{pmatrix}H & \mathbf 1\\ \mathbf 1^\top & 0\end{pmatrix}\begin{pmatrix}\Delta\\ \nu\end{pmatrix}=\begin{pmatrix}-g\\ 0\end{pmatrix},$$

backtrack so a_A + tΔ > 0 and V > 0; drop j from A if a_j hits 0. On convergence (‖tΔ‖_∞ < 1e-15), add any
j ∉ A with r_j > 1 + 1e-12 and repeat. Then certify (§5.2).

### 5.5 Growth lost to a wrong belief (E1 result)
L(p, q) = G_p(a_p) − G_p(a_q) ≥ 0. With full investment L = D(p‖q) exactly; for any convex allocation set
with wealth linear in a, L ≤ D(p‖q) (proof sketch in `experiments/E1_RESULTS.md`; prior art Barron &
Cover 1988 — Unverified).

### 5.6 KL-ball bet (ADR-005)
Inner worst case, for fixed a, with f_s = log V_s(a):

$$\phi(\lambda)=-\lambda\log\sum_s\pi_s e^{-f_s/\lambda}-\lambda\varepsilon,\qquad \min_{D(Q\|\pi)\le\varepsilon}\mathbb E_Q[f]=\sup_{\lambda>0}\phi(\lambda),$$

φ is concave in λ; `kl_worst_case` finds the root of φ′(λ) = −log E_π[e^{−g/λ}] − ε − E_w[g]/λ (g = f − min f
for stability, w ∝ π e^{−g/λ}) by safeguarded Newton/bisection to |φ′| ≤ 1e-13. Worst case
Q*_s ∝ π_s e^{−f_s/λ*}. Let s_w = argmin_s f_s; since D(δ_{s_w}‖π) = log(1/π_{s_w}), if ε ≥ log(1/π_{s_w}) the
ball contains the point mass on s_w, λ* → 0 and value = f_{s_w} (handled explicitly, no root search).
Joint program: ADR-005. **Polish:** joint Newton on the smooth system in (a_A, ν, λ):
r^{Q*(a,λ)}_j(a) − ν = 0 (j ∈ A), Σ_A a_j = 1, φ′(λ; a) = 0, with the Jacobian from ForwardDiff.jl
(prototype 2026-11-01 in Python with a central-difference Jacobian: 2 Newton steps took the conic
solution to residual 2e-16; a naive alternating best-response scheme did **not** converge — it
drifted to a different allocation with residual 0.30 — so it is not used). **Certify:** saddle-point
property — a* must satisfy ρ(a*) ≤ 1e-10 with π replaced by Q*, and |D(Q*‖π) − ε| ≤ 1e-10 when λ* > 0.
Oracles: CRRA bet with γ = 1 + 1/λ* equals a*; small ε: value ≈ E_π f − √(2ε Var_π f).

### 5.7 Uncertain probabilities collapse to the predictive
G is linear in the scenario probabilities. For independent posteriors P_k, E[Π_k P_k^{b_k}(1−P_k)^{1−b_k}] =
Π_k E[P_k]^{b_k}(1−E[P_k])^{1−b_k}, so the Bayes-optimal bet uses the posterior means exactly — prior
width alone does not shrink the bet. Width matters only through dependence between bets'
probabilities (`predictive_from_draws`) or payoff uncertainty (`sampled_payoffs`, where Jensen's
inequality on log V applies). This is stated in docstrings because the parent article reports a 15%
reduction from a weaker prior with the same mean (§PLAN Errata, candidate item).

### 5.8 KL radius from data
`kl_radius_chi2(K, n; level)` = χ²_{K−1, level} / (2n): under the null that the model is right, 2n·D(p̂‖π)
is asymptotically χ²_{K−1} (likelihood-ratio / G-test), so a ball of this radius around the empirical
distribution p̂ of n observations over K outcomes covers the truth with probability ≈ level (Ben-Tal
et al. 2013 use φ-divergence radii of this form — verify direction convention before the article
cites it). E7 checks coverage by simulation.

### 5.9 Wasserstein-ball bet (ADR-006)
Candidate support X̃ = [X; unseen] (T rows); costs c_st. Inner worst case for fixed a:
value = max_{λ≥0} −λε + Σ_s π_s min_t (f_t + λ c_st), f_t = log V_t(a) — piecewise linear and concave in λ;
solved exactly by scanning its breakpoints. Q*_t = Σ_s γ_st, with each s sending π_s to its argmin t
(one s may split at the breakpoint so the budget binds). Joint program: ADR-006. **Polish:** Newton on the
active-piece KKT system (active pairs (s, t), γ on active pairs, a_A, ν, λ). **Certify:** ρ(a*) ≤ 1e-10 under
Q* and transport budget |Σ γ c − ε| ≤ 1e-10 when λ* > 0.

### 5.10 Sample sizes (single binary bet, gross payoff w)
Net return per unit: R = w − 1 w.p. p, −1 w.p. 1 − p; mean e = pw − 1; Var R = p(1−p)w². Detecting e ≠ 0
at z standard errors needs e/√(Var/n) ≥ z ⇒ n_detect = z² p(1−p) w²/e². Measuring e to ±10% (one SE)
needs √(Var/n) = 0.1e ⇒ n_10% = 100 p(1−p) w²/e².

## 6. Error Handling Strategy
- **Validate early:** constructors and public functions check shapes, signs, probability sums and
  feasibility; throw `ArgumentError` / `DomainError` with the failing rule and offending index.
- **Never return an uncertified bet:** every solver path ends in `certify`; failure throws
  `CertificationError(kkt, tol, method, msg)` — no warning-and-continue.
- **Solver status:** JuMP termination other than `OPTIMAL` / `ALMOST_OPTIMAL` throws `SolverError`
  with the status; polishing still decides acceptance.
- **Polish failure** (Newton stalls, active set cycles > 10 times, or robust system singular): throw
  `CertificationError` with the best residual reached. Users may retry with `solver = :scs`.
- **Numerical guards:** log-sum-exp for mixtures, BMA and KL duals; V_s ≤ 0 → G = −Inf (diagnostics
  only, never inside solvers).
- **Experiments:** each `main` writes `_meta.toml` and returns a summary; `reproduce_all.jl` catches an
  experiment's exception, marks it FAIL with the message, continues the rest, and exits 1.
- **Python:** Julia exceptions mapped to Python exceptions (§4.8); no silent fallbacks.
- No logging framework; failures surface as exceptions, experiment progress via `println`.

## 7. Dependency List
Versions are the latest registered in Julia's General registry (or PyPI) on 2026-11-01; T0 resolves
the Manifest on Julia 1.12.x and records any version the resolver had to change.

| Library | Version (pinned) | Purpose |
|---|---|---|
| Julia | 1.12.x (patch recorded in T0) | runtime (ADR-001) |
| JuMP | 1.32.1 | modeling layer for conic programs |
| MathOptInterface | 1.54.0 | cone types (`ExponentialCone`), solver interface |
| Clarabel | 0.11.1 | primary interior-point conic solver (ADR-004) |
| SCS | 2.7.1 | fallback conic solver |
| ForwardDiff | 1.4.6 | Jacobians for robust Newton polish (§5.6, §5.9) |
| Distributions | 0.25.132 | Beta / Dirichlet / payoff samplers, χ² quantiles |
| StableRNGs | 1.1.0 | version-stable RNG for E2–E7 (ADR-009) |
| Aqua | 0.8.18 | package hygiene tests (test-only) |
| Random, LinearAlgebra, Printf, Statistics, Test, TOML | stdlib | E1 code, tests, metadata |
| juliacall (PyPI) | 0.9.37 | Python bridge (ADR-007) |
| PythonCall.jl | 0.9.37 | installed by juliacall/juliapkg (not a Kelly_KL dependency) |
| numpy (PyPI) | pinned in T0 (latest at install) | arrays for the wrapper |
| pytest (PyPI) | pinned in T0 (latest at install) | parity tests |

## 8. Open Design Questions
None.

## 9. Resolved Design Questions (record)
- Wasserstein ground metric and support (outline OQ-2) → ADR-006 (finite support, L1).
- Figure backend → ADR-008. Julia version → ADR-001 (exact patch is a recorded fact, not a design choice).
- Robust-bet certification and polishing method → §5.6 / §5.9 (joint Newton; alternating scheme rejected after test).
- Source of the errata pair (0.0667, 0.12) is a **content** question for the article, handled in PLAN.md
  (task T-ERR-1), not a library design question.
