# ADR-002: Package name `Kelly_KL.jl` and repository layout
**Date:** 2026-11-01
**Status:** Accepted

## Context
The library needs a name, a module name and a place on disk next to the article. Julia's General
registry naming guidelines discourage underscores in package names (they prefer UpperCamelCase,
e.g. `KellyKL`). Registration is out of scope.

## Decision
Name the package `Kelly_KL.jl` with module `Kelly_KL`, rooted at
`Projects/extended-kelly-KL/code/`, with this layout:

```
code/
├── Project.toml, Manifest.toml
├── src/            Kelly_KL.jl + one file per component (DESIGN.md §2)
├── test/           runtests.jl + one file per component; cases/ (shared JSON cases)
├── experiments/    e1_*.jl … e7_*.jl, plotting.jl, reproduce_all.jl,
│                   output/ (generated, git-ignored), reference/ (committed)
├── python/         pyproject.toml, juliapkg.json, kelly_kl/, tests/
├── docs/           SDD.md, adr/, DESIGN.md, TEST_PLAN.md, PLAN.md
├── README.md, CHANGELOG.md, LICENSE
```

Article figures are written to `Projects/extended-kelly-KL/img/`.

## Rationale
- John chose the name; it matches the project folder and the article's KL theme.
- The underscore is legal Julia syntax; it only conflicts with registry style, and registration
  is out of scope.
- One package root keeps the article, code and figures in one Wild Peaches project folder.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| `KellyKL.jl` (registry style) | John's decision; registration not planned |
| Separate repository outside the project folder | Splits article and code; harder to reproduce |

## Consequences
- If the package is ever registered, it must be renamed (e.g. `KellyKL`), which breaks
  `using Kelly_KL` in user code and the Python wrapper's import. Recorded here so that cost is known.
- The current top-level E1 files (`code/e1_*.jl`, `code/e1_results.csv`, `code/E1_RESULTS.md`)
  move to `code/experiments/` in milestone M1.
