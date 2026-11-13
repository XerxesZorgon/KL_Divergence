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
 python/tests (pytest parity)       test/cases/cases.toml (shared) ──►    reproduce_all.jl ──► output/ vs reference/
                                                                      figures ──► ../img/*.svg
```

## 2. Module / Component Breakdown
| Component | Responsibility | Interface |
|---|---|---|
| `src/Kelly_KL.jl` | Module definition, includes, exports | `using Kelly_KL` |
| `src/model.jl` | `ScenarioModel` type, validation, constructors for single events, simultaneous bets, correlated tables | §4.1 |
| `src/exact.jl` | Closed-form allocations: full investment (a = q), greedy with cash | §4.2 |
| `src/conic.jl` | JuMP models for log-optimal, KL-robust and Wasserstein-robust problems; solver selection | internal: `solve_conic(problem; solver)` |
| `src/polish.jl` | Newton polishing on the active set for log-optimal (k = 0) and CRRA-weighted (k = 1/λ) objectives; KL-robust λ search uses it | `polish_crra`, `polish_log_optimal` |
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
    certificate::Symbol  # :kkt (saddle KKT ≤ tol) or :duality_gap (Wasserstein ties: gap ≤ 1e-7)
end

struct CertificationError <: Exception
    kkt::Float64; tol::Float64; method::Symbol; msg::String
end
```

### 3.3 Shared test case file (test/cases/cases.toml)
One `[[case]]` table per case with `id`, `kind` (`allocation` | `kl` | `wasserstein` | `diagnostic`),
`tol` (1e-10), `[case.inputs]` (a model table `type = single_event | simultaneous | correlated | scenario` with its arguments,
plus `eps`, `unseen`, `cost`, `function`, `args` as the kind needs) and `[case.expected]` (outputs as numbers or arrays).
Generated by `scripts/fill_cases.jl` on the reference machine — never edited by hand; read by `test/test_cases.jl` and
`python/tests/test_parity.py` through the mirrored runners `test/cases/case_runner.jl` / `python/tests/case_runner.py`.
TOML, not JSON: Julia's TOML stdlib and Python's tomllib (≥ 3.11) read it with no extra dependency (Task 051).

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
**v0.2 (ADR-012):** both methods take `upper = nothing` (per-asset caps); any finite cap forces the conic
path. Signatures, defaults and errors: §10.1, §10.3.

### 4.3 Certification
```julia
kkt_residual(X, a, π; τ_a = 1e-12) -> Float64
certify(result; tol = 1e-10)       -> result  # throws CertificationError
```
**v0.2 (ADR-012):** `kkt_residual` takes `upper = nothing`; with caps it returns ρ_u (§10.2). `certify` is
unchanged (it compares the stored residual, computed with `result.upper`).

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
kl_cash_radius(m) -> (radius, Q)    # KL distance to the nearest no-edge world; the KL-robust bet is all cash iff ε ≥ radius; Inf when some bet mix beats cash in every scenario
kl_radius_chi2(K, n; level = 0.95)  -> ε                # §5.8
wasserstein_robust_allocation(m, ε; unseen = zeros(0, J), cost = :l1, tol = 1e-10) -> RobustResult
wasserstein_worst_case(m, a, ε; unseen, cost)        -> (value, λ, Q)   # exact 1-D dual (§5.9)
```
`ε` is required (ADR-011). `cost ∈ (:l1, :linf)` or an S × T matrix. `unseen` rows must have J columns
and non-negative entries.
**v0.2 (ADR-012):** `kl_robust_allocation` and `kl_cash_radius` take `upper = nothing`;
`wasserstein_robust_allocation` takes `upper = nothing` only to reject caps (`ArgumentError`). See §10.

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
**Amended in v0.2 (ADR-012):** when caps are supplied, the generalized residual ρ_u of §10.2 replaces ρ
(the multiplier ν replaces the constant 1). With `upper = nothing`, ρ is computed by the expression above,
unchanged, so v0.1 residuals are bit-identical.

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
**Algorithm (Task 036, replaces the planned joint Newton):** (i) ε = 0 → nominal bet; (ii) with cash, if ε ≥ `kl_cash_radius(m)` the robust bet is all cash, certified by the I-projection Q (max_j E_Q[X_j] − 1 ≤ tol); (iii) otherwise the joint conic program of ADR-005 seeds λ, and a 1-D search on log λ solves D(Q(λ)‖π) = ε, where for each λ the bet is the CRRA bet with γ = 1 + 1/λ (`polish_crra`, k = 1/λ; dH/dλ = D − ε by the envelope theorem). **Certify:** KKT residual of a* under Q* ∝ π·V^{−1/λ*} ≤ 1e-10 and |D(Q*‖π) − ε| ≤ 1e-10. **v1 limit:** if λ* → 0 (no bracket) — e.g. a bet mix that beats cash in every scenario with a large ε — `CertificationError` is thrown.
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
(one s may split at the breakpoint so the budget binds). Joint program: ADR-006. **Algorithm (Task 038):** joint conic program (ADR-006) → structure iteration (polish the bet under the exact worst case Q with a 1e-300 weight floor on every support row, recompute Q, repeat ≤ 10 times). **Certify (two-tier, ADR-006 amendment):** tier 1 — Q unchanged and KKT of a* under Q* ≤ 1e-10; tier 2 (tied worst cases) — duality gap U − L ≤ 1e-7, with U = optimal growth under the conic dual transport plan's marginal (a distribution in the ball) and L = the exact worst case of the returned bet.

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
### v0.2 (moved from §8, 2026-11-11)
- OQ-1 → **ADR-013** (ν by minimax; no division; F = ∅ handled; feasibility threshold 1 − 1e-12).
- OQ-2 → **ADR-014** (offline cvxpy + Clarabel fixtures, τ_o = 1e-8, strict-complementarity margin 1e-6,
  tolerances; amends ADR-010).
- OQ-3 → **ADR-015** (diagnostics stay uncapped in v0.2).
- OQ-4 → **ADR-016** (no cash and |Σu − 1| ≤ 1e-12: skip the solver, a = u/Σu, method `:single_point`; also in
  `kl_robust_allocation`). Raised 2026-11-11 while closing OQ-1: for Σu just below 1, A_u is empty in exact
  arithmetic; just above 1 it is a near-point, and the conic solver may report the problem infeasible.

Original text, kept for the record:

- **OQ-1 — Free set empty (F = ∅).** Then Σ_{j∈U} u_j = 1 and ν = (1 − Σ_U u_j r_j)/(1 − Σ_U u_j) is 0/0
  (example: no cash, u summing to 1, A_u = {u}). Any ν in [max_Z r_j, min_U r_j] satisfies the KKT
  conditions. Recommendation: in that case set the ν-terms of ρ_u to (max_Z r_j − min_U r_j)⁺ (0 when Z = ∅).
  Also decide (i) a threshold below which 1 − Σ_U u_j is treated as 0 (the formula is ill-conditioned
  near it), and (ii) whether Σ u_j that misses 1 only by rounding (no cash) is infeasible or accepted.
- **OQ-2 — Independent oracle and comparison tolerance.** The brief cites "the Python oracle pattern from
  ADR-010", but ADR-010 defines no such pattern and the repository has no cvxpy code (python/tests holds
  only `case_runner.py`, `conftest.py`, `test_parity.py`). Decide: where the cvxpy + Clarabel oracle runs
  (pytest test with pinned `cvxpy`/`clarabel` test dependencies, or an offline script whose results are
  stored as fixtures), and the comparison tolerances. Clarabel's 1e-12 is a *solver* tolerance; allocations
  from an interior-point method are not accurate to 1e-12 in general (it warned "solution may be
  inaccurate" in the 2026-11-11 checks; T-ORC-4 uses 1e-7 for the same reason). Proposed: growth ≤ 1e-10,
  allocation and ν ≤ 1e-8 on non-degenerate cases. Amends ADR-010.
- **OQ-3 — Diagnostics with caps.** `growth_loss(m_true, a_belief)` compares with the uncapped optimum
  a_p; for a capped bettor the fair comparison is the capped optimum. `growth_curve` / `zero_growth_point`
  scale bets by c > 1, which can exceed caps. Recommendation: no caps in v0.2 diagnostics (they take the
  allocation as given; docstrings say so); `growth_loss` gains `upper` in a later version. Confirm or reject.

### v0.1
- Wasserstein ground metric and support (outline OQ-2) → ADR-006 (finite support, L1).
- Figure backend → ADR-008. Julia version → ADR-001 (exact patch is a recorded fact, not a design choice).
- Robust-bet certification and polishing method → §5.6 / §5.9 (joint Newton; alternating scheme rejected after test).
- Source of the errata pair (0.0667, 0.12) is a **content** question for the article, handled in PLAN.md
  (task T-ERR-1), not a library design question.

## 10. v0.2 amendments: per-asset upper bounds (ADR-012, 2026-11-11)
Allocation set with caps: A_u = {a : a ≥ 0, Σ_j a_j = 1, a_j ≤ u_j}, u_j ∈ (0, ∞], cash uncapped. Use cases
(AFS-GEM): order-book depth (one asset per price level, each with limited stake) and exits (an existing
position valued at its liquidation bid, held up to its current size). Sections §1–§9 describe v0.1 and
remain valid when `upper = nothing`.

### 10.1 Public interface changes
```julia
optimal_allocation(m::ScenarioModel; method = :auto, solver = :clarabel, tol = 1e-10, upper = nothing) -> BetResult
optimal_allocation(q, ohat; cash = true, method = :auto, solver = :clarabel, tol = 1e-10, upper = nothing) -> BetResult
kkt_residual(X, a, π; τ_a = 1e-12, upper = nothing)        -> Float64     # ρ (§5.2) or ρ_u (§10.2)
kl_robust_allocation(m, ε; solver = :clarabel, tol = 1e-10, upper = nothing) -> RobustResult
kl_cash_radius(m; maxit = 2000, tol = 1e-13, upper = nothing) -> (radius, Q)  # upper validated; result independent of it (claim, T-BOX-6)
wasserstein_robust_allocation(m, ε; unseen, cost, solver, tol, gap_tol, upper = nothing) -> RobustResult  # upper !== nothing ⇒ ArgumentError
certify(result; tol = 1e-10) -> result                      # unchanged: compares result.kkt, computed with result.upper
# internal (not exported), used by tests and the polish:
kkt_components(X, a, π; τ_a = 1e-12, upper = nothing) -> (ρ, ν*, U, Z, F)   # ν* by minimax (ADR-013)
```
- **Default:** `upper = nothing` means no caps and runs the v0.1 code path unchanged.
- **Form:** `AbstractVector{<:Real}` of length J (cash included, in model column order). Uncapped entries
  are `Inf`. For the single-event shortcut, J = n + 1 when `cash = true`.
- **Normalization:** entries u_j ≥ 1 cannot bind on the simplex and are replaced by `Inf`. If every
  entry is then `Inf`, `upper` is stored as `nothing` and the call is the v0.1 call (byte-identical).
- **Results:** `BetResult` and `RobustResult` (§3.2) gain a last field `upper::Union{Nothing, Vector{Float64}}`
  (normalized caps). `BetResult(m, a, method; τ_a, upper)` passes `upper` to `kkt_residual`.
- **Method values:** `BetResult.method` may also be `:single_point` (near-point shortcut, ADR-016).
- **Python (ADR-007):** every wrapped function above gains `upper: Optional[np.ndarray] = None` (`np.inf` for
  uncapped); result dataclasses gain `upper`. No extra logic in Python.
- **Shared cases (§3.3):** `[case.inputs]` may hold `upper` (TOML array; `inf` is a valid TOML float, read by
  Julia's TOML stdlib and Python's tomllib).

### 10.2 Generalized KKT residual (single source of truth with caps; ADR-013 supersedes the ADR-012 form)
r_j(a) = Σ_{s: π_s>0} π_s X_sj / V_s(a). Sets, with τ_a = 1e-12: U = {j : u_j < ∞, a_j ≥ u_j − τ_a} (capped),
Z = {j ∉ U : a_j ≤ τ_a} (zero), F = the rest (free). KKT holds iff some ν has r_j = ν on F, r_j ≤ ν on Z,
r_j ≥ ν on U. With A = max_{j∈F∪Z} r_j (max ∅ = −∞) and B = min_{j∈F∪U} r_j (min ∅ = +∞):

$$\text{ν-part}=\begin{cases}\max\big(0,\tfrac{A-B}{2}\big)&A,B\text{ finite}\\0&\text{otherwise}\end{cases},\qquad \nu^*=\begin{cases}\tfrac{A+B}{2}&A,B\text{ finite}\\A&B=+\infty\\B&A=-\infty\end{cases}$$

$$\rho_u(a)=\max\Big\{\text{ν-part},\ \max_j(-a_j)^+,\ \max_j(a_j-u_j)^+,\ \big|\textstyle\sum_j a_j-1\big|\Big\}.$$

The ν-part is the minimum over ν of the ADR-012 terms max_F|r_j − ν|, max_Z(r_j − ν)⁺, max_U(ν − r_j)⁺.
Meaning: every free asset has the same marginal growth ν*; an unheld asset would not add growth at the
margin (r_j ≤ ν*); a capped asset would add growth if its cap were raised (r_j ≥ ν*, shadow price r_j − ν*).
For any b ∈ A_u, G(b) − G(a) ≤ 2·ν-part + τ_a Σ_{Z∪U}|r_j − ν*| (ADR-013), so ρ_u ≤ 1e-10 means growth
within ≈ 2e-10 of the capped optimum.
Implementation rule: if `upper === nothing` (or every cap normalizes to ∞), ρ is the v0.1 expression of
§5.2, unchanged. With any finite cap, ρ_u above is used, also when U = ∅. The closed form
ν = (1 − Σ_U u_j r_j)/(1 − Σ_U u_j) is a test oracle only (F ≠ ∅). The same residual applies under any weights
summing to 1 (Q* for the KL saddle certificate, the CRRA weights in `polish_crra`). Pass ⇔ ρ_u ≤ 1e-10.
τ_a = 1e-12 is meant for polished allocations (inactive coordinates exactly 0 or u_j); raw interior-point
output needs a looser classification (ADR-014 uses τ_o = 1e-8 for the oracle).

### 10.3 Error cases (new)
| Condition | Exception |
|---|---|
| `length(upper) != size(m.X, 2)` | `ArgumentError` |
| any u_j ≤ 0 or NaN | `ArgumentError` |
| `m.has_cash` and `upper[1]` finite (caps on cash are out of scope) | `ArgumentError` |
| infeasible caps: `!m.has_cash` and Σ_j u_j < 1 − 1e-12 (ADR-013) | `ArgumentError` (message names Σ u_j); \|Σ_j u_j − 1\| ≤ 1e-12: no error, near-point shortcut (ADR-016) |
| `wasserstein_robust_allocation(...; upper)` with `upper !== nothing` | `ArgumentError` ("caps not supported for Wasserstein balls in v0.2") |
| polish cannot reach ρ_u ≤ tol | `CertificationError` (unchanged rule, §6) |

### 10.4 Code locations that change
| File / function | Change |
|---|---|
| `src/conic.jl` `solve_conic` | add `a[j] <= u[j]` for finite caps |
| `src/conic.jl` `_simplex_value` | rescales with `a ./ sum(a)`, which can push a capped asset over its cap. Do **not** change it: it is also used for Q in `kl_inner_direct` and for a in `wasserstein_robust_conic`. Add a projection onto A_u (below) and use it only when caps are present |
| new `src/box.jl` | cap validation and normalization (§10.1, §10.3) and the projection below; included from `src/Kelly_KL.jl`; near-point test \|Σu − 1\| ≤ 1e-12 (ADR-016) |
| projection onto A_u (new internal helper, in `src/box.jl`) | Euclidean projection: a_j = clamp(x_j − θ, 0, u_j) with θ chosen so Σ a_j = 1 (piecewise-linear in θ; exact by sorting breakpoints). Alternative in the brief: fix capped coordinates at u_j and renormalize only the free ones — can itself push a free coordinate over its cap, so it must iterate; the projection is preferred |
| `src/polish.jl` `polish_crra` | (1) start: replace `a ./= sum(a)` and the `0.999 a + 0.001/J` fallback by the projection; (2) Newton step on F only, capped and zero coordinates fixed (same bordered system, Σ_F Δ = 0); (3) ratio test on both bounds: d_i < 0 reaching 0 (drop to Z, existing) and d_i > 0 reaching u_i (move to U, set a_i = u_i exactly); (4) in-loop `a ./= sum(a)` replaced by the projection; (5) release rule: j ∈ U leaves U when r_j < ν − add_tol; (6) add rule: j ∈ Z joins when r_j > ν + add_tol (v0.1 compares with 1; with caps ν < 1 is possible, so an asset with ν < r_j < 1 must be added); (7) stopping test uses `kkt_residual(...; upper)` |
| `src/polish.jl` `polish_log_optimal` | pass `upper` through |
| `src/certify.jl` | `kkt_residual(...; upper)`, `kkt_components`; `BetResult` field and constructor keyword |
| `src/optimize.jl` `optimal_allocation` | `upper` keyword; any finite cap ⇒ conic path even for `:single_event` (greedy and full investment ignore caps); pass `upper` to conic, polish and `BetResult`; no cash and \|Σu − 1\| ≤ 1e-12 ⇒ return u/Σu with method `:single_point`, no solver (ADR-016) |
| `src/robust.jl` `kl_robust_conic` | add `a[j] <= u[j]`; projection instead of `_simplex_value` for a |
| `src/robust.jl` `kl_robust_allocation` | ε = 0 path passes `upper` to `optimal_allocation`; cash path unchanged (no cap active at all-cash, residual computed with `upper`); λ search passes `upper` to `polish_crra`; saddle certificate ρ_u(a*) under Q*; `RobustResult.upper`; same near-point shortcut for every ε: a = u/Σu, worst case from `kl_worst_case`, saddle ρ_u under its Q, no KL-boundary check (ADR-016) |
| `src/robust.jl` `kl_cash_radius` | validate `upper`; math unchanged (claim: at all-cash no bet cap binds, so the all-cash test is the v0.1 test — **Unverified** until T-BOX-6) |
| `src/robust.jl` `wasserstein_robust_allocation` | add `upper = nothing` keyword; throw `ArgumentError` if not `nothing` |
| `test/certify_hook.jl` | recompute residuals with `result.upper` |
| `scripts/fill_cases.jl`, `test/cases/case_runner.jl`, `python/tests/case_runner.py` | read/write `upper` in `[case.inputs]` |
| `python/kelly_kl/` | `upper` keyword on every wrapped function above; `upper` field on result dataclasses |
| unchanged | `model.jl`, `uncertainty.jl`, `diagnostics.jl` (ADR-015: code unchanged; docstrings of `growth_loss`, `growth_curve`, `zero_growth_point`, `simulate_ruin` state they use the uncapped optimum or the allocation as given), `exact.jl`, `crosscheck.jl`, `experiments/` |
| new `test/oracle/` (ADR-014) | `generate_box_oracle.py` (offline, by hand), `requirements.txt` (exact pins), `box_oracle.toml` (committed fixtures, read by Julia tests) |
