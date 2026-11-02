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
Requires Julia ≥ 1.12 (developed on 1.12.1, Windows 11; Python 3.13.12). Not registered; install from the folder:

```julia
julia> ]
pkg> activate path/to/extended-kelly-KL/code
pkg> instantiate        # installs the exact versions in Manifest.toml
```

Python wrapper *(planned, milestone M7)*, Python ≥ 3.10:

```bash
pip install -e path/to/extended-kelly-KL/code/python
```

The first `import kelly_kl` sets up Julia through juliacall and can take a few minutes.

## Usage
*(planned API — see `docs/DESIGN.md` §4)*

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
print(kk.optimal_allocation(m).a)
```

Reproduce the article *(planned, milestone M8)*:

```bash
julia --project=. experiments/reproduce_all.jl
```

## Running tests
```bash
julia --project=. -e "using Pkg; Pkg.test()"     # full suite (includes 50 000-trial E1 stress test)
KELLY_KL_QUICK=1 julia --project=. -e "using Pkg; Pkg.test()"   # quick inner-loop run
pytest python/tests                                # Python parity (≤ 1e-10 vs Julia)
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
