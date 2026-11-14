# Kelly_KL.jl
Exact, certified Extended Kelly bets from a full scenario distribution, with tools to measure what
a wrong distribution costs and to bet robustly against what the model misses.

> **Status:** planning documents complete (v0.1, 2026-11-01); code not yet written. Sections marked
> *(planned)* describe the API in `docs/DESIGN.md` and will be updated after each milestone.

## What it does
- Sizes single, simultaneous and correlated bets (or investments) exactly from scenario
  probabilities π and a gross-returns matrix X; every result is certified optimal
  (KKT residual ≤ 1e-10).
- Turns uncertain probabilities (Beta / Dirichlet posteriors), sampled payoffs and model mixtures
  into the exact predictive distribution — the bet comes from the distribution, with **no fractional
  Kelly** and no risk knobs.
- Diagnoses model error: growth lost to a wrong belief vs D(p‖q), holdout log-score comparison,
  the overbetting curve and its zero-growth point, the drawdown law, minimum-stake ruin, and the data
  needed to measure an edge.
- Computes worst-case bets over a KL ball or a finite-support Wasserstein ball (which can include
  outcomes the model never saw).
- Reproduces every number and figure of *Extended Kelly II: When the Distribution Is Wrong*
  (experiments E1–E7) with one command.
- Callable from Python through juliacall.

Background: [Extended Kelly Criterion](https://www.wildpeaches.xyz/blog/2025-11-24-extended-kelly)
(Wild Peaches, 2025) and the *Extended Kelly II* outline in `../OUTLINE.md`.

## Installation
Requires Julia ≥ 1.12 (developed on 1.12.1, Windows 11). Not registered; install from the folder:

```julia
julia> ]
pkg> activate path/to/extended-kelly-KL/code
pkg> instantiate        # installs the exact versions in Manifest.toml
```

Python wrapper `kelly_kl` (thin layer over the Julia package through juliacall 0.9.37; no numerical code in Python).
Requires Python ≥ 3.11. **On Windows the Python build must bundle OpenSSL ≥ 3.5** — juliapkg ties Julia's OpenSSL to
Python's and Julia 1.12 needs 3.5: python.org 3.14 builds work, 3.13 builds (OpenSSL 3.0) do not (ADR-007). From `code/`:

```bash
py -3.14 -m venv python/.venv                          # Windows; elsewhere: python3.14 -m venv python/.venv
python/.venv/Scripts/python -m pip install -e python[test]   # Linux/macOS: python/.venv/bin/python
```

The first `import kelly_kl` installs a Julia environment inside the venv (`python/.venv/julia_env`, with `Kelly_KL` as a
dev path dependency and the Manifest's solver versions) and precompiles it — a few minutes once, seconds afterwards.
juliacall uses the `julia` on PATH; set `PYTHON_JULIAPKG_EXE` to point it at a specific Julia. Never edit files inside the
venv to get past an install error.

## Usage
Full API: `docs/DESIGN.md` §4.

```julia
using Kelly_KL

# Two simultaneous independent bets, gross payoffs, cash allowed (parent-article example)
m   = simultaneous_bets([0.3, 0.25], [4.0, 6.0]; cash = true)
res = optimal_allocation(m)          # res.a ≈ [cash, 0.0645, 0.0992], res.kkt ≤ 1e-10

# Growth lost by betting belief q when the truth is p
L = growth_loss(m_true, optimal_allocation(m_belief).a)   # ≤ kl(p, q)

# Worst case over a KL ball of radius ε, and a Wasserstein ball with an unseen crash scenario
rk = kl_robust_allocation(m, 0.05)
rw = wasserstein_robust_allocation(m, 0.05; unseen = [1.0 0.0 0.0])
```

```python
import numpy as np, kelly_kl as kk
m = kk.simultaneous_bets(np.array([0.3, 0.25]), np.array([4.0, 6.0]), cash=True)
print(kk.optimal_allocation(m).a)          # [0.83634533 0.06450023 0.09915444]
rk = kk.kl_robust_allocation(m, 0.01)      # RobustResult dataclass: rk.a, rk.worst_case_growth, rk.λ, …
```

Python follows Julia's conventions where they differ from Python's: outcome indices (`y` in `holdout_logscore`) and
`BetResult.active` are 1-based; functions that need randomness take an integer `seed` and use `StableRNG(seed)`; Julia
`ArgumentError` / `DomainError` arrive as `ValueError`, and certification or solver failures as
`kelly_kl.CertificationError` / `kelly_kl.SolverError`.

Reproduce the article (Success Criterion 1, ADR-009):

```bash
julia --project=. experiments/reproduce_all.jl     # ≈ 15 min; exit code 0 = everything reproduced
```

It deletes `experiments/output/`, runs experiments E1–E7 at full size (refreshing the figures in `../img/`), prints each
experiment's own pass/fail verdict, and compares all 21 outputs with the committed `experiments/reference/`: byte-identical
on the reference machine (`KELLY_KL_REFERENCE_HOST=1`), within relative 1e-12 elsewhere (run metadata such as wall time is
not compared). References were generated on Julia 1.12.1, Windows 11, with the pinned `Manifest.toml`; regenerating them
(`experiments/update_reference.jl`, reference machine only) is a reviewed change of its own.

## Per-asset caps (v0.2)
`optimal_allocation`, `kl_robust_allocation`, `kl_cash_radius` and the building blocks (`kkt_residual`, `kkt_components`,
`solve_conic`, `polish_crra`) take `upper`: one cap per column of the payoff matrix, `a_j ≤ upper[j]`. `Inf` (or any entry ≥ 1)
means uncapped; there is no cap on cash. The result is certified by the capped residual ρ_u (ADR-012, ADR-013), and
`result.upper` stores the caps that were used. `kkt_components` returns `(ρ, ν*, U, Z, F)`: ν* is the common marginal growth of
the free assets, and for a capped asset r_j − ν* ≥ 0 is its **shadow price**, the growth gained per unit increase of its cap.

**(a) One bet, capped at 10% (T-BOX-1).** Kelly alone would stake 0.2.

```julia
using Kelly_KL
m = ScenarioModel([0.6, 0.4], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
r = optimal_allocation(m; upper = [Inf, 0.1])
ρ, ν, U, Z, F = kkt_components(m.X, r.a, m.π; upper = r.upper)
r_bet = (m.X' * (m.π ./ (m.X * r.a)))[2]
println("a = ", r.a); println("method = ", r.method); println("kkt = ", r.kkt)
println("nu* = ", ν, ", r_bet = ", r_bet, ", shadow price = ", r_bet - ν)
```
```
a = [0.9, 0.1]
method = clarabel
kkt = 0.0
nu* = 0.9898989898989898, r_bet = 1.0909090909090908, shadow price = 0.101010101010101
```
The stake is the cap, ν* = 98/99 and r_bet = 12/11, with kkt ≤ 1e-10. The shadow price r_bet − ν* ≈ 0.101 means raising the cap
by 0.01 adds about 0.001 to expected log growth.

**(b) Order-book depth.** One binary contract that resolves YES with your belief p = 0.6; no fees, for clarity. The asks form
three assets, each returning 1/price per dollar on YES and 0 on NO: level 1 at 0.50 (at most 5% of bankroll), level 2 at 0.53
(at most 10%), level 3 at 0.56 (uncapped).

```julia
using Kelly_KL
prices = [0.50, 0.53, 0.56]
X = [1.0 1/prices[1] 1/prices[2] 1/prices[3]; 1.0 0.0 0.0 0.0]          # rows: YES, NO
m = ScenarioModel([0.6, 0.4], X; assets = ["cash", "ask 0.50", "ask 0.53", "ask 0.56"], has_cash = true, structure = :correlated)
r = optimal_allocation(m; upper = [Inf, 0.05, 0.10, Inf])
ρ, ν, U, Z, F = kkt_components(m.X, r.a, m.π; upper = r.upper)
rv = m.X' * (m.π ./ (m.X * r.a))
println("allocation = ", round.(r.a; digits = 6), "  kkt = ", r.kkt, "  capped = ", U, "  nu* = ", ν)
for j in 2:4; println(m.assets[j], ": stake ", round(r.a[j]; digits = 6), ", shadow price r_j - nu* = ", round(rv[j] - ν; digits = 6)); end
```
```
allocation = [0.853617, 0.05, 0.096383, 0.0]  kkt = 1.6653345369377348e-16  capped = [2]  nu* = 0.9970089730807576
ask 0.50: stake 0.05, shadow price r_j - nu* = 0.059821
ask 0.53: stake 0.096383, shadow price r_j - nu* = 0.0
ask 0.56: stake 0.0, shadow price r_j - nu* = -0.053411
```
Level 1, the cheapest, is exhausted first: its cap binds and its shadow price (0.0598) is what an extra unit of depth at 0.50 would
be worth. The bet goes into level 2 (9.6% of bankroll, below its cap) and stops there; level 3 stays at zero because its edge
is below the marginal growth ν*.

**(c) Exit.** You hold 100 YES contracts bought earlier, the bid is 0.55 and cash is $945, so wealth at liquidation value is
W0 = 945 + 100·0.55 = 1000. Assets: cash; "hold" (pays 1/0.55 per dollar on YES, capped at 100·0.55/1000 = 0.055, since you
cannot keep more than you own); and "buy" at the 0.57 ask, uncapped.

```julia
using Kelly_KL
function exit_model(p)
    X = [1.0 1/0.55 1/0.57; 1.0 0.0 0.0]                                  # rows: YES, NO
    ScenarioModel([p, 1 - p], X; assets = ["cash", "hold", "buy"], has_cash = true, structure = :correlated)
end
for p in (0.50, 0.62)
    r = optimal_allocation(exit_model(p); upper = [Inf, 100 * 0.55 / 1000, Inf])
    println("p = ", p, ": allocation = ", round.(r.a; digits = 6), " kkt = ", r.kkt, "  hold \$ = ", round(1000 * r.a[2]; digits = 2), ", buy \$ = ", round(1000 * r.a[3]; digits = 2))
end
```
```
p = 0.5: allocation = [1.0, 0.0, 0.0] kkt = 0.0  hold $ = 0.0, buy $ = 0.0
p = 0.62: allocation = [0.885488, 0.055, 0.059512] kkt = 5.551115123125783e-17  hold $ = 55.0, buy $ = 59.51
```
At p = 0.50 the belief is below the bid, so the hold stake is 0: sell everything. At p = 0.62 the whole position is kept (the cap on
"hold" binds) and about $59.5 more is bought at the ask. Exits fall out of re-optimizing the portfolio under the cap on what you
hold, with no separate exit rule.

**(d) Python.**

```python
import numpy as np, kelly_kl as kk
m = kk.ScenarioModel(np.array([0.6, 0.4]), np.array([[1.0, 2.0], [1.0, 0.0]]), ("cash", "bet"), True, "single_event")
r = kk.optimal_allocation(m, upper=[np.inf, 0.1]); print(r.a, r.method)   # [0.9 0.1] clarabel
```

**(e) Limits.** `wasserstein_robust_allocation` rejects caps in v0.2 (`ArgumentError`, `ValueError` in Python). The diagnostics
(`growth_loss`, `growth_curve`, `zero_growth_point`, `simulate_ruin`) take no `upper`: they use the uncapped optimum or the
allocation as given (ADR-015). With no cash and |Σu − 1| ≤ 1e-12 the feasible set is a single point and
`optimal_allocation` returns it with `method = :single_point` (ADR-016). Fees: fold a per-contract fee into the effective price
(cost per contract = price + fee), which keeps the problem linear. Kalshi rounds fees per order, so compute the exact order fee
after rounding to whole contracts.

## Running tests
```bash
julia --project=. -e "using Pkg; Pkg.test()"     # full suite (includes 50 000-trial E1 stress test)
KELLY_KL_QUICK=1 julia --project=. -e "using Pkg; Pkg.test()"   # quick inner-loop run
python/.venv/Scripts/python -m pytest python/tests   # Python parity (≤ 1e-10 vs Julia; shared cases in test/cases/cases.toml)
```
On Windows PowerShell set the variable with `$env:KELLY_KL_QUICK=1`. What counts as passing:
`docs/TEST_PLAN.md`.

## Contributing
Personal research project (John Peach / Wild Peaches). Work follows the document chain in `docs/`
(SDD → ADRs → DESIGN → TEST_PLAN → PLAN → tasks.md) and the one-task-at-a-time rules in
`Templates/AGENTS.md`. Design changes need an ADR first. Issues and suggestions are welcome via the
Wild Peaches site.

## License
MIT © John Peach – Wild Peaches (wildpeaches.xyz). See `LICENSE` (added in M0).
