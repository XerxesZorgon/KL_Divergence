# ADR-007: Python wrapper via juliacall
**Date:** 2026-11-01
**Status:** Accepted

## Context
John's other projects use Python, and the brief requires a Python wrapper in v1 that matches Julia
to 1e-10 on shared cases.

## Decision
Ship a thin Python package `kelly_kl` in `code/python/` that calls the Julia package through
juliacall 0.9.37 (PythonCall.jl 0.9.37), with the local Julia package declared as a dev dependency
in `juliapkg.json`.

## Rationale
- juliacall is free (MIT), maintained, and installs its own Julia environment via juliapkg.
- Calling the real Julia code (not a re-implementation) makes parity a property of the bridge, not
  of two code bases.
- NumPy arrays pass to Julia without copying where layouts allow.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| PyJulia (`julia` package) | Effectively unmaintained; fragile on Windows |
| Re-implement solvers in Python (cvxpy) | Two code bases; parity would be approximate |
| Command-line / JSON subprocess bridge | Slow start-up per call; awkward API |

## Consequences
- First import starts Julia (tens of seconds; longer on first precompile).
- Python ≥ 3.11 required (tomllib, Task 051). On Windows the Python build must bundle OpenSSL ≥ 3.5, because juliapkg ties
  Julia's OpenSSL_jll to Python's and Julia 1.12 needs 3.5: python.org 3.13 builds (OpenSSL 3.0) fail, 3.14 builds work
  (Fix 050a). Never edit juliacall's juliapkg.json to bypass this.
- Wrapper functions are one-to-one with the Julia public API (DESIGN.md §4.8); no extra logic in
  Python, so any numeric difference is a bridge bug.
- Shared test cases live in `test/cases/cases.toml`, read by both `Pkg.test()` and `pytest`.
