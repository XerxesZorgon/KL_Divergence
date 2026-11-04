"""T-PY-1..3: Python wrapper parity with Julia (Success Criterion 3)."""
import pathlib
import re
import tomllib

import numpy as np
import pytest

import kelly_kl as kk
from case_runner import case_error, run_case

CODE = pathlib.Path(__file__).resolve().parents[2]
CASES = tomllib.loads((CODE / "test" / "cases" / "cases.toml").read_text(encoding="utf-8"))["case"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_shared_case(case):                                         # T-PY-1
    assert case["expected"], "expected values missing: run scripts/fill_cases.jl"
    err = case_error(run_case(case), case["expected"])
    assert err <= case["tol"], f"{case['id']}: max abs difference {err:.3e} > {case['tol']}"


def test_invalid_probabilities_raise_value_error():                 # T-PY-2
    with pytest.raises(ValueError):
        kk.ScenarioModel(np.array([0.5, 0.4]), np.array([[1.0, 2.0], [1.0, 0.0]]), ("cash", "bet"), True,
                         "single_event")._jl()
    with pytest.raises(ValueError):
        kk.simultaneous_bets([1.3], [2.0])


def test_certification_failure_raises_certification_error():        # T-PY-2
    m = kk.simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    with pytest.raises(kk.CertificationError) as info:
        kk.optimal_allocation(m, tol=1e-30)                          # no residual can meet 1e-30
    assert info.value.kkt > 1e-30


def test_same_julia_as_recorded():                                   # T-PY-3
    plan = (CODE / "docs" / "PLAN.md").read_text(encoding="utf-8")
    recorded = re.search(r"\|\s*Julia\s*\|\s*([0-9]+\.[0-9]+\.[0-9]+)", plan).group(1)
    assert kk.julia_version() == recorded


def test_wrapper_covers_public_api():
    from juliacall import Main as jl
    exported = {str(s) for s in jl.names(jl.Kelly_KL)} - {"Kelly_KL"}
    missing = sorted(n for n in exported if not hasattr(kk, n))
    assert not missing, f"Julia exports without a Python wrapper: {missing}"
