# ADR-001: Julia 1.12 as language and runtime
**Date:** 2026-11-01
**Status:** Accepted

## Context
The article's code language is Julia, E1 is already written in Julia, and the library must be
callable from Python. John's Windows 11 desktop runs Julia 1.12.x (exact patch to be recorded),
while E1 was first verified on Julia 1.13.3 (Linux).

## Decision
Write the library in Julia with `[compat] julia = "1.12"` and generate the pinned `Manifest.toml`
and all reference outputs on John's Julia 1.12.x.

## Rationale
- Julia is the article's stated code language and E1 already exists in it.
- Every pinned dependency (JuMP 1.32, Clarabel 0.11, SCS 2.7, Distributions 0.25.132,
  MathOptInterface 1.54) declares `julia ≥ 1.10` in the General registry, so 1.12 is supported.
- Reference outputs must come from the machine that runs the reproduce-all command; that is
  John's desktop.
- `julia = "1.12"` means 1.12 ≤ version < 2.0, so 1.13 users can still install the package.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| Pin 1.13 and upgrade John's machine | Adds a setup step; no feature in 1.13 is needed |
| Python core with Julia wrapper | Article code is Julia; Python ecosystem lacks a free exact Extended Kelly solver |
| Lower bound 1.10 | Untested on 1.10; no user needs it |

## Consequences
- Task T0 records the exact `julia --version` in PLAN.md and README. If it is below 1.12, stop and
  upgrade with juliaup before any other task.
- Julia's default RNG streams are not guaranteed stable across minor versions, so E1's stored
  reference outputs are tied to the pinned version (see ADR-009).
- `Manifest.toml` is Julia-version specific; it is regenerated only by a deliberate task.
