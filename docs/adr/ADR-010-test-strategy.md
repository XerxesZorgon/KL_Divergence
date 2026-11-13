# ADR-010: Test strategy — oracle tests first, certification everywhere, shared parity cases
**Date:** 2026-11-01
**Status:** Accepted

## Context
The library's value is exactness. Line coverage alone does not show that a bet is optimal. The
success criterion names specific known results that the tests must check.

## Decision
Use Julia's `Test` stdlib with one test file per component, organized around **oracle tests**
(known exact results), a **certification invariant** (every bet returned in any test has KKT
residual ≤ 1e-10), the **E1 stress test** (L ≤ D(p‖q), 0 violations), **regression tests** against
stored values, and **parity tests** (`pytest`) on shared JSON cases. Aqua.jl 0.8.18 checks package
hygiene (ambiguities, unbound type parameters, stale dependencies).

## Rationale
- Oracles (classic Kelly, Cover–Thomas, L ≤ D, KL dual vs direct solve, CRRA equivalence) catch
  wrong answers that run without error.
- A global certification hook makes "optimal" a tested property everywhere, not in one test.
- Shared JSON cases let Julia and Python check the same inputs.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Coverage target as primary gate | Measures execution, not correctness |
| Property-based testing library (Supposition.jl) | Seeded random sweeps (as in E1) are enough and simpler |
| Separate test suites per language with own cases | Parity could drift silently |

## Consequences
- `Pkg.test()` includes the 50 000-trial E1 stress test; it takes longer (target < 5 min). A
  `KELLY_KL_QUICK=1` env var reduces it to 2 000 trials for inner-loop runs; the full run is the gate.
- Line coverage is reported (target ≥ 85% of `src/`) but is not a pass condition by itself.
- TEST_PLAN.md lists every test ID and its pass criterion.

## Amendment (ADR-014, 2026-11-11)
A fifth test kind, **fixture oracles**: reference solutions from an independent solver (cvxpy 1.9.3 +
clarabel 0.11.1) are generated offline by `test/oracle/generate_box_oracle.py`, accepted only if the oracle
certifies itself (ρ_u ≤ 1e-8 at τ_o = 1e-8), committed as `test/oracle/box_oracle.toml`, and read by Julia
tests. `Pkg.test()` stays Julia-only and deterministic; Python is not needed to run it. Fixtures change only
by a hand-run regeneration recorded in its own commit. (The shared parity cases are TOML, not JSON:
DESIGN.md §3.3.)
