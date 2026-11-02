# ADR-009: Reproducibility — pinned Manifest, fixed seeds, stored reference outputs
**Date:** 2026-11-01
**Status:** Accepted

## Context
Success criterion 1: one command reproduces every number and figure in *Extended Kelly II* and
matches stored reference outputs. Julia's default RNG (Xoshiro) streams and `randn`/`randexp`
algorithms are not guaranteed stable across Julia minor versions; Distributions.jl samplers can
change between package versions; floating-point results can differ across operating systems.

## Decision
1. `Manifest.toml` is committed and pins every dependency; `julia = "1.12"` compat (ADR-001).
2. **E1** keeps `Xoshiro` with seeds 20261101 (main), 7 (stress) and 1 (plot subsample), so its
   refactor can be checked against the original script.
3. **E2–E7** use `StableRNGs.StableRNG(seed)`, whose `rand`, `randn`, `randexp`, `randperm` and
   `shuffle` streams are guaranteed stable across releases. Sampling through Distributions.jl is
   stable only under the pinned Manifest.
4. Every experiment writes to `experiments/output/`; reference copies are committed in
   `experiments/reference/`, generated once on John's machine with the pinned Manifest.
5. `experiments/reproduce_all.jl` runs E1–E7 and compares:
   - **on the reference machine** (`KELLY_KL_REFERENCE_HOST` env var set): byte-identical files;
   - **elsewhere**: CSV numeric fields within relative 1e-12 (absolute 1e-15 near zero), SVG
     compared after parsing numeric attributes with the same tolerance.
   It prints a per-file PASS/FAIL table and exits non-zero on any failure.
6. Each experiment writes a `*_meta.toml` with Julia version, git commit, seed and wall time
   (excluded from comparison).

## Rationale
- Byte equality is the strongest check and is achievable on one machine.
- StableRNGs (free, MIT) removes cross-version drift for E2–E7 at no cost.
- E1 cannot switch RNG without changing its published outputs.

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| StableRNGs for E1 too | Changes E1 outputs; violates the refactor requirement |
| Tolerance comparison everywhere | Weaker; hides accidental changes on the reference machine |
| Docker image | Heavy for a Windows desktop workflow; Manifest + seeds suffice |

## Consequences
- E1 reference outputs regenerated on Julia 1.12 may differ numerically from the 1.13.3 values in
  `E1_RESULTS.md` (different RNG stream on another Julia version). T-M1 records both; the article
  quotes the 1.12 references.
- Updating any dependency means regenerating references in a dedicated, reviewed task.
