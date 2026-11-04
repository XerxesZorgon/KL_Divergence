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
