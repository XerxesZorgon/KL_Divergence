"""kelly_kl — Python wrapper for the Julia package Kelly_KL (ADR-007, DESIGN.md §4.8).

Every function calls the Julia function of the same name through juliacall; there is no numerical code
in Python, so any difference from Julia is a bridge bug. Arrays go in and come out as NumPy float arrays.

Conventions that follow Julia, not Python:
  * outcome indices (`y` in `holdout_logscore`, `logscore_difference`) and `BetResult.active` are 1-based;
  * functions that need a random-number generator (`simulate_ruin`, `sampled_payoffs`) take an integer
    `seed` and use `StableRNG(seed)`, so a Python call reproduces the Julia call `rng = StableRNG(seed)`;
  * `sampled_payoffs` samplers are `(name, *params)` tuples naming a Distributions.jl distribution,
    e.g. `("Uniform", 1.8, 2.2)`.

The first import starts Julia (and on first use installs the Julia environment); expect a delay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from juliacall import Main as jl, JuliaError

jl.seval("using Kelly_KL, StableRNGs, Distributions")
_K = jl.Kelly_KL

__all__ = [
    "ScenarioModel", "BetResult", "RobustResult", "CertificationError", "SolverError", "julia_version",
    "single_event", "simultaneous_bets", "correlated_bets", "payoffs_from_odds", "growth",
    "kkt_residual", "certify", "optimal_allocation",
    "beta_predictive", "dirichlet_predictive", "predictive_from_draws", "sampled_payoffs", "growth_se", "mixture",
    "bma_weights",
    "kl", "growth_loss", "holdout_logscore", "logscore_difference", "growth_curve", "zero_growth_point",
    "drawdown_probability", "effective_multiple", "ruin_probability_classical", "simulate_ruin", "n_detect",
    "n_relative",
    "kl_worst_case", "kl_inner_direct", "kl_cash_radius", "kl_robust_allocation", "wasserstein_worst_case",
    "wasserstein_robust_allocation", "kl_radius_chi2",
]


# ----------------------------------------------------------------------------------------------------------------
# Errors
class CertificationError(Exception):
    """A computed bet failed its optimality certificate (Julia `Kelly_KL.CertificationError`)."""

    def __init__(self, kkt: float, tol: float, method: str, msg: str):
        super().__init__(f"{msg} (residual {kkt:.3e} > tol {tol:.1e}, method :{method})")
        self.kkt, self.tol, self.method, self.msg = kkt, tol, method, msg


class SolverError(Exception):
    """The conic solver did not reach an optimal status (Julia `Kelly_KL.SolverError`)."""

    def __init__(self, status: str, msg: str):
        super().__init__(f"{msg} (status {status})")
        self.status, self.msg = status, msg


def _raise_mapped(e: JuliaError):
    ex = e.exception
    f = lambda name: jl.getfield(ex, jl.Symbol(name))          # field access returns plain Python values
    if jl.isa(ex, _K.CertificationError):
        raise CertificationError(float(f("kkt")), float(f("tol")), str(jl.string(f("method"))), str(f("msg"))) from None
    if jl.isa(ex, _K.SolverError):
        raise SolverError(str(f("status")), str(f("msg"))) from None
    if jl.isa(ex, jl.ArgumentError) or jl.isa(ex, jl.DomainError):
        raise ValueError(jl.sprint(jl.showerror, ex)) from None
    raise e


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except JuliaError as e:
        _raise_mapped(e)


# ----------------------------------------------------------------------------------------------------------------
# Conversions (Python -> Julia)
def _vec(x) -> Any:
    return jl.Vector[jl.Float64](np.ascontiguousarray(x, dtype=np.float64).ravel())


def _ivec(x) -> Any:
    return jl.Vector[jl.Int](np.ascontiguousarray(x, dtype=np.int64).ravel())


def _mat(x) -> Any:
    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return jl.Matrix[jl.Float64](a)


def _sym(s: str) -> Any:
    return jl.Symbol(s)


# Conversions (Julia -> Python)
def _arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


@dataclass(frozen=True)
class ScenarioModel:
    """Scenario probabilities `pi` (S,), gross returns `X` (S, J), asset names, cash flag, structure."""
    pi: np.ndarray
    X: np.ndarray
    assets: tuple
    has_cash: bool
    structure: str

    def _jl(self):
        return _call(_K.ScenarioModel, _vec(self.pi), _mat(self.X), assets=jl.Vector[jl.String](list(self.assets)),
                     has_cash=bool(self.has_cash), structure=_sym(self.structure))


def _model(m) -> ScenarioModel:
    return ScenarioModel(_arr(m.π), _arr(m.X), tuple(str(s) for s in m.assets), bool(m.has_cash), str(m.structure))


@dataclass(frozen=True)
class BetResult:
    a: np.ndarray
    growth: float
    kkt: float
    method: str
    active: tuple          # 1-based asset indices, as in Julia
    assets: tuple


def _bet(r) -> BetResult:
    return BetResult(_arr(r.a), float(r.growth), float(r.kkt), str(r.method), tuple(int(i) for i in r.active),
                     tuple(str(s) for s in r.assets))


@dataclass(frozen=True)
class RobustResult:
    a: np.ndarray
    worst_case_growth: float
    nominal_growth: float
    λ: float
    Q: np.ndarray
    support: np.ndarray
    kkt: float
    ball: str
    ε: float
    certificate: str


def _robust(r) -> RobustResult:
    return RobustResult(_arr(r.a), float(r.worst_case_growth), float(r.nominal_growth), float(r.λ), _arr(r.Q),
                        _arr(r.support), float(r.kkt), str(r.ball), float(r.ε), str(r.certificate))


def _m(m):
    return m._jl() if isinstance(m, ScenarioModel) else m


def julia_version() -> str:
    """Version of the Julia running behind the wrapper (T-PY-3)."""
    return str(jl.string(jl.VERSION))


# ----------------------------------------------------------------------------------------------------------------
# Model construction
def single_event(q, ohat, cash: bool = True) -> ScenarioModel:
    return _model(_call(_K.single_event, _vec(q), _vec(ohat), cash=cash))


def simultaneous_bets(p, w, cash: bool = True) -> ScenarioModel:
    return _model(_call(_K.simultaneous_bets, _vec(p), _vec(w), cash=cash))


def correlated_bets(P_joint, W) -> ScenarioModel:
    return _model(_call(_K.correlated_bets, _vec(P_joint), _mat(W)))


def payoffs_from_odds(o, commission: float = 0.0) -> np.ndarray:
    return _arr(_call(_K.payoffs_from_odds, _vec(o), commission=float(commission)))


def growth(m: ScenarioModel, a) -> float:
    return float(_call(_K.growth, _m(m), _vec(a)))


# Certification and optimization
def kkt_residual(X, a, pi) -> float:
    return float(_call(_K.kkt_residual, _mat(X), _vec(a), _vec(pi)))


def certify(result: BetResult, tol: float = 1e-10) -> BetResult:
    if result.kkt <= tol:
        return result
    raise CertificationError(result.kkt, tol, result.method, "uncertified bet")


def optimal_allocation(m: ScenarioModel, method: str = "auto", solver: str = "clarabel", tol: float = 1e-10) -> BetResult:
    return _bet(_call(_K.optimal_allocation, _m(m), method=_sym(method), solver=_sym(solver), tol=float(tol)))


# Uncertainty and model averaging
def beta_predictive(alpha, beta) -> np.ndarray:
    return _arr(_call(_K.beta_predictive, _vec(alpha), _vec(beta)))


def dirichlet_predictive(alpha) -> np.ndarray:
    return _arr(_call(_K.dirichlet_predictive, _vec(alpha)))


def predictive_from_draws(Pdraws, w, cash: bool = True) -> ScenarioModel:
    return _model(_call(_K.predictive_from_draws, _mat(Pdraws), _vec(w), cash=cash))


def sampled_payoffs(p, samplers, M: int, seed: int, cash: bool = True) -> ScenarioModel:
    jl_samplers = jl.Vector[jl.Any]([getattr(jl.Distributions, s[0])(*[float(v) for v in s[1:]]) for s in samplers])
    return _model(_call(_K.sampled_payoffs, _vec(p), jl_samplers, int(M), rng=jl.StableRNG(int(seed)), cash=cash))


def growth_se(m: ScenarioModel, a) -> float:
    return float(_call(_K.growth_se, _m(m), _vec(a)))


def mixture(models, weights) -> ScenarioModel:
    jm = jl.Vector[_K.ScenarioModel]([_m(m) for m in models])
    return _model(_call(_K.mixture, jm, _vec(weights)))


def bma_weights(logml) -> np.ndarray:
    return _arr(_call(_K.bma_weights, _vec(logml)))


# Diagnostics
def kl(p, q) -> float:
    return float(_call(_K.kl, _vec(p), _vec(q)))


def growth_loss(m_true: ScenarioModel, a_belief) -> float:
    return float(_call(_K.growth_loss, _m(m_true), _vec(a_belief)))


def holdout_logscore(pi, y) -> dict:
    """Mean and standard error of log pi[y_i] over 1-based held-out outcome indices y."""
    r = _call(_K.holdout_logscore, _vec(pi), _ivec(y))
    return {"mean": float(r.mean), "se": float(r.se)}


def logscore_difference(pi1, pi2, y) -> dict:
    """Paired mean and standard error of log pi1[y_i] − log pi2[y_i] (1-based y); estimates D(p‖pi2) − D(p‖pi1)."""
    r = _call(_K.logscore_difference, _vec(pi1), _vec(pi2), _ivec(y))
    return {"mean": float(r.mean), "se": float(r.se)}


def growth_curve(m: ScenarioModel, a_star, cs) -> np.ndarray:
    return _arr(_call(_K.growth_curve, _m(m), _vec(a_star), _vec(cs)))


def zero_growth_point(m: ScenarioModel, a_star) -> dict:
    r = _call(_K.zero_growth_point, _m(m), _vec(a_star))
    return {"c0": float(r.c0), "c_quad": float(r.c_quad)}


def drawdown_probability(x: float, c: float) -> float:
    return float(_call(_K.drawdown_probability, float(x), float(c)))


def effective_multiple(e_belief: float, e_true: float) -> float:
    return float(_call(_K.effective_multiple, float(e_belief), float(e_true)))


def ruin_probability_classical(p: float, a: int, N: int) -> float:
    return float(_call(_K.ruin_probability_classical, float(p), int(a), int(N)))


def simulate_ruin(m: ScenarioModel, a, seed: int, trials: int, rounds: int, w0: float = 1.0, min_stake: float = 0.0,
                  lot: float = 0.0, fixed_cost: float = 0.0, fixed_stake: float | None = None,
                  target: float = float("inf")) -> dict:
    fs = jl.nothing if fixed_stake is None else float(fixed_stake)
    r = _call(_K.simulate_ruin, _m(m), _vec(a), rng=jl.StableRNG(int(seed)), trials=int(trials), rounds=int(rounds),
              w0=float(w0), min_stake=float(min_stake), lot=float(lot), fixed_cost=float(fixed_cost),
              fixed_stake=fs, target=float(target))
    return {"p_ruin": float(r.p_ruin), "se": float(r.se), "drawdown_q": _arr(r.drawdown_q),
            "final_q": _arr(r.final_q), "min_fraction": _arr(r.min_fraction)}


def n_detect(p: float, w: float, z: float = 1.96) -> float:
    return float(_call(_K.n_detect, float(p), float(w), z=float(z)))


def n_relative(p: float, w: float, rel: float = 0.10) -> float:
    return float(_call(_K.n_relative, float(p), float(w), rel=float(rel)))


# Robust bets
def kl_worst_case(m: ScenarioModel, a, eps: float) -> tuple:
    v, lam, Q = _call(_K.kl_worst_case, _m(m), _vec(a), float(eps))
    return float(v), float(lam), _arr(Q)


def kl_inner_direct(m: ScenarioModel, a, eps: float) -> tuple:
    v, Q = _call(_K.kl_inner_direct, _m(m), _vec(a), float(eps))
    return float(v), _arr(Q)


def kl_cash_radius(m: ScenarioModel) -> tuple:
    d, Q = _call(_K.kl_cash_radius, _m(m))
    return float(d), _arr(Q)


def kl_robust_allocation(m: ScenarioModel, eps: float, solver: str = "clarabel", tol: float = 1e-10) -> RobustResult:
    return _robust(_call(_K.kl_robust_allocation, _m(m), float(eps), solver=_sym(solver), tol=float(tol)))


def _cost(cost):
    return _sym(cost) if isinstance(cost, str) else _mat(cost)


def wasserstein_worst_case(m: ScenarioModel, a, eps: float, unseen=None, cost="l1") -> tuple:
    U = _mat(np.zeros((0, np.asarray(a).size))) if unseen is None else _mat(unseen)
    v, lam, Q = _call(_K.wasserstein_worst_case, _m(m), _vec(a), float(eps), unseen=U, cost=_cost(cost))
    return float(v), float(lam), _arr(Q)


def wasserstein_robust_allocation(m: ScenarioModel, eps: float, unseen=None, cost="l1", solver: str = "clarabel",
                                  tol: float = 1e-10, gap_tol: float = 1e-7) -> RobustResult:
    J = np.asarray(m.X).shape[1] if isinstance(m, ScenarioModel) else None
    U = _mat(np.zeros((0, J))) if unseen is None else _mat(unseen)
    return _robust(_call(_K.wasserstein_robust_allocation, _m(m), float(eps), unseen=U, cost=_cost(cost),
                         solver=_sym(solver), tol=float(tol), gap_tol=float(gap_tol)))


def kl_radius_chi2(K: int, n: int, level: float = 0.95) -> float:
    return float(_call(_K.kl_radius_chi2, int(K), int(n), level=float(level)))
