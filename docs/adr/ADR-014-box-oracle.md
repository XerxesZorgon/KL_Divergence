# ADR-014: Offline cvxpy + Clarabel oracle for capped problems, stored as fixtures
**Date:** 2026-11-11
**Status:** Accepted (approved by John 2026-11-11, with the τ_o, non-degeneracy and rejection-log additions; KL rules (i)–(iii) and the requirements hash added 2026-11-11 on John's instruction; closes DESIGN.md §8 OQ-2; amends ADR-010)

## Context
T-BOX-2/3/7 compare the library with an independent solve of the capped problem. ADR-010 defines no
Python oracle, the repository has no cvxpy code, and `Pkg.test()` must stay Julia-only and
deterministic (ADR-009). An interior-point solve at tolerance 1e-12 is not accurate to 1e-12 in the
allocation, and its inactive coordinates are not exactly 0 (ADR-013, evidence).

## Decision
An offline generator `test/oracle/generate_box_oracle.py` (pinned cvxpy 1.9.3, clarabel 0.11.1) writes
fixture files under `test/oracle/` that are committed to git; Julia tests only read them.

1. **Runs by hand only.** Never called from `Pkg.test()` or pytest. A regeneration is its own commit whose
   message says so and gives the generator's seed, Python, cvxpy and clarabel versions.
2. **Format.** TOML (`test/oracle/box_oracle.toml`), as for the shared cases (DESIGN §3.3): read by Julia's
   TOML stdlib with no new dependency; `inf` is a valid TOML float. Per case: `id`, `kind`
   (`allocation` | `kl`), π, X, `upper`, `has_cash`, `eps` (kl), oracle a, growth (or worst-case growth
   and λ for kl), ν*, the sets U/Z/F, the oracle's own ρ_u, and `degenerate` (bool). File header:
   seed, versions, date, cases attempted, cases rejected and rejection rate (per kind), and
   `requirements_sha256` = SHA-256 of the committed `test/oracle/requirements.txt` used for the run.
   `requirements.txt` is frozen and committed **before** the generator is run; the generator computes the
   hash from the file and refuses to run if the file has uncommitted changes.
3. **Oracle-side classification.** The generator computes ρ_u (ADR-013) on the oracle's allocation exactly
   as solved (only clipped to [0, u_j]), with classification tolerance **τ_o = 1e-8** instead of τ_a.
   Raising τ only turns the equality r_j = ν into the matching one-sided inequality for coordinates
   within 1e-8 of a bound; a truly free coordinate satisfies either inequality, so this cannot reject a
   correct answer, and the unchecked side affects growth by about 1e-8·|r_j − ν|. The library keeps
   τ_a = 1e-12, because its polish sets inactive coordinates to exactly 0 or u_j.
4. **Acceptance.** An allocation case is written only if the oracle's ρ_u ≤ 1e-8 (τ_o). Rejected cases are
   counted and logged, not written.

   **KL fixture rules.** The oracle solves the capped joint program of ADR-005 (plus a_j ≤ u_j) and returns
   (a, λ); Q* ∝ π·V(a)^{−1/λ}.
   - **(i) Self-certification:** the saddle residual ρ_u of a under Q* ≤ 1e-8, with classification
     tolerance τ_o = 1e-8 (as for allocation cases).
   - **(ii) Boundary and interior:** |D(Q*‖π) − ε| ≤ 1e-8 and λ > 0 (the case is not all-cash and not
     λ → 0; generation draws ε < D_min and requires ≥ 1 binding cap).
   - **(iii) Comparison:** library worst-case growth vs stored worst-case growth ≤ 1e-7 (as T-ORC-4). The
     stored a and λ are reported, not tested (allocations may differ on degenerate cases, ADR-004; λ is
     sensitive to solver tolerance).

   A kl case failing (i) or (ii) is rejected and counted; the 10% rule (item 5) applies per kind.
5. **Rejection rate.** If more than 10% of attempted cases are rejected, the generator exits non-zero and
   writes nothing; the generator is investigated before fixtures are regenerated.
6. **Non-degenerate** (strict complementarity, margin 1e-6, on the oracle's sets and ν*): every zero-stake
   asset has r_j ≤ ν* − 1e-6 and every capped asset has r_j ≥ ν* + 1e-6. Otherwise `degenerate = true`.
7. **Comparison tolerances** (library vs fixture):

| Case | Compared | Tolerance |
|---|---|---|
| allocation, non-degenerate | growth; allocation (∞-norm); ν*; sets U/Z/F | ≤ 1e-10; ≤ 1e-8; ≤ 1e-8; library sets (τ_a) equal the stored sets (τ_o) |
| allocation, degenerate | growth only (ADR-004 consequence: allocations may differ) | ≤ 1e-10 |
| kl | worst-case growth (rule iii) | ≤ 1e-7 (as T-ORC-4); a and λ reported, not tested |

   The set comparison catches a polish that converges to the wrong active set, which a 1e-8 allocation
   comparison could miss.

## Rationale
- Committed fixtures keep `Pkg.test()` Julia-only, deterministic and fast; no Python at test time.
- Accepting a case only if the oracle certifies itself (ρ_u ≤ 1e-8) stops a bad oracle solve from
  becoming a reference.
- τ_o = 1e-8 avoids rejecting cases with unheld bets, which are exactly the cases depth caps create
  (with τ_a on raw Clarabel output, 7/30 were rejected; see log).
- **Independence (partial).** cvxpy calls Clarabel's Rust core; the library uses Clarabel.jl (pure Julia).
  Same algorithm, separate implementations, separate modeling layers (cvxpy vs JuMP), and the library
  adds its own polish. The fully independent oracle is the exact single-bet case (ν = 98/99,
  r_bet = 12/11, T-BOX-1) and the F = ∅ case (T-BOX-12).

## Alternatives Considered
| Option | Rejected Because |
|---|---|
| cvxpy in pytest at test time | Adds Python and a solver to every test run; non-deterministic across platforms; breaks ADR-009 spirit |
| Oracle with τ_a = 1e-12 | Rejected 7/30 cases, biased toward cases without unheld bets |
| Snap oracle coordinates (≤ 1e-9 → 0 or u_j) then check | Changes the reference: moving 1e-9 of stake shifts growth by up to several × 1e-10, beyond the 1e-10 growth tolerance; the redistribution rule is arbitrary |
| MOSEK as oracle | Proprietary; Clarabel is the free alternative already in use |

## Consequences
- New files: `test/oracle/generate_box_oracle.py`, `test/oracle/requirements.txt` (exact pins of the
  generator's environment from `pip freeze`, including transitive packages; committed before generation),
  `test/oracle/box_oracle.toml` (header carries `requirements_sha256`). T-BOX-15 recomputes the hash in
  Julia (SHA stdlib, test-only) and fails if it differs from the header.
  The generator's virtual environment lives outside the repository or is git-ignored.
- PLAN.md Dependency Lock lists cvxpy 1.9.3 and clarabel 0.11.1 as "oracle generation only, not a runtime
  or test dependency". Both have Windows wheels for Python 3.14 (`cvxpy-1.9.3-cp314-cp314-win_amd64`,
  `clarabel-0.11.1-cp39-abi3-win_amd64`; checked with `pip download`, 2026-11-11).
- Fixture changes are reviewable diffs; a changed reference is visible in git.

## Rejection log
| Date | Generator | Attempted | Rejected | Rate | Note |
|---|---|---|---|---|---|
| 2026-11-11 | prototype (Claude, scratch script, not committed): S = 8, J = 5 with cash, bet caps U(0.05, 0.3), seed 3 | 30 | 1 | 3.3% | rejected case: ρ_u = 9.4e-8 at τ_o (Clarabel accuracy); all 29 accepted were non-degenerate at margin 1e-6; 8 had a binding cap |
| 2026-11-13 | `generate_box_oracle.py` (seed 71; Python 3.14.10, cvxpy 1.9.3, clarabel 0.11.1) | allocation 200, kl 100 | allocation 7, kl 4 | allocation 3.5%, kl 4.0% | accepted: 193 allocation + 96 kl = 289 cases. Redrawn (not counted as attempted): allocation 18 (no cap could be built: no bet with uncapped stake > 0.03, or no-cash n = 1), 0 non-binding; kl 16 (no cap could be built) + 83 (all-cash, λ ≤ 1e-9 or no binding cap). Degenerate at margin 1e-6: 0 allocation, 0 kl. Largest stored ρ_u 9.3e-9. Some solves returned `optimal_inaccurate` and were kept only if they passed the ρ_u rule. `requirements_sha256` 667ae44c1c34… |

Task 071 (PLAN.md M9) appends the row for the committed fixtures.
