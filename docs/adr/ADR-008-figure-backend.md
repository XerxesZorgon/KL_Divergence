# ADR-008: Figures written as standard-library SVG
**Date:** 2026-11-01
**Status:** Accepted

## Context
E1's figure is produced by hand-written SVG using only the standard library. Article figures for
E2–E7 need a backend, and the reproduce-all command compares outputs against stored references.

## Decision
Write all article figures as SVG with the Julia standard library, through a small shared helper
`experiments/plotting.jl` (axes, grid, markers, lines, legend) extracted from `e1_plot.jl`. The
plotting helper lives in `experiments/`, not in the package `src/`.

## Rationale
- Byte-identical output across runs on the pinned machine, so figures can be compared exactly.
- Zero dependencies and no slow first-plot load.
- E1's figure stays byte-identical after the refactor (requirement).
- Keeps the library free of plotting code (Four Principles: minimum code).

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| CairoMakie 0.15.17 for E2–E7 | Heavy dependency; SVG bytes change between versions, so figures could only be compared by data |
| CairoMakie everywhere | Breaks E1's byte-identical figure requirement |
| Plots.jl / GR | Same byte-stability problem; extra binary dependency |

## Consequences
- More code per new figure type (each new mark type is added to `plotting.jl` once).
- The helper must be extracted from `e1_plot.jl` without changing E1's SVG bytes; the E1 refactor
  test checks this.
- Figures follow E1's style (Helvetica/Arial, palette #2a78d6, #eb6834, #1baf7a, #eda100).
