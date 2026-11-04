"""Python mirror of test/cases/case_runner.jl (T-PY-1): computes a shared case's outputs through kelly_kl.
Keep in step with the Julia runner: a case kind added there must be added here."""
import numpy as np
import kelly_kl as kk


def build_model(d):
    t = d["type"]
    if t == "single_event":
        return kk.single_event(d["q"], d["ohat"], cash=d["cash"])
    if t == "simultaneous":
        return kk.simultaneous_bets(d["p"], d["w"], cash=d["cash"])
    if t == "correlated":
        return kk.correlated_bets(d["P_joint"], np.array(d["W"], dtype=float))
    if t == "scenario":
        m = kk.ScenarioModel(np.array(d["pi"], dtype=float), np.array(d["X"], dtype=float), tuple(d["assets"]),
                             d["has_cash"], d["structure"])
        m._jl()                                   # validate exactly as Julia does
        return m
    raise ValueError(f"unknown model type {t}")


def run_case(c):
    kind, inp = c["kind"], c["inputs"]
    if kind == "allocation":
        r = kk.optimal_allocation(build_model(inp["model"]))
        return {"a": r.a, "growth": r.growth}
    if kind == "kl":
        r = kk.kl_robust_allocation(build_model(inp["model"]), inp["eps"])
        return {"a": r.a, "worst_case_growth": r.worst_case_growth, "lambda": r.λ}
    if kind == "wasserstein":
        m = build_model(inp["model"])
        U = np.array(inp["unseen"], dtype=float) if "unseen" in inp else None
        r = kk.wasserstein_robust_allocation(m, inp["eps"], unseen=U, cost=inp["cost"])
        return {"a": r.a, "worst_case_growth": r.worst_case_growth, "lambda": r.λ}
    if kind == "diagnostic":
        f, x = inp["function"], inp.get("args", [])
        simple = {
            "kl": lambda: kk.kl(x[0], x[1]),
            "n_detect": lambda: kk.n_detect(x[0], x[1]),
            "n_relative": lambda: kk.n_relative(x[0], x[1]),
            "ruin_probability_classical": lambda: kk.ruin_probability_classical(x[0], int(x[1]), int(x[2])),
            "drawdown_probability": lambda: kk.drawdown_probability(x[0], x[1]),
            "kl_radius_chi2": lambda: kk.kl_radius_chi2(int(x[0]), int(x[1])),
            "bma_weights": lambda: kk.bma_weights(x[0]),
            "beta_predictive": lambda: kk.beta_predictive(x[0], x[1]),
        }
        if f in simple:
            return {"value": simple[f]()}
        m = build_model(inp["model"])
        if f == "kl_cash_radius":
            return {"value": kk.kl_cash_radius(m)[0]}
        a = kk.optimal_allocation(m).a
        if f == "zero_growth_point":
            return {"value": kk.zero_growth_point(m, a)["c0"]}
        if f == "growth_curve":
            return {"value": kk.growth_curve(m, a, x[0])}
        if f == "kl_worst_case":
            v, lam, Q = kk.kl_worst_case(m, a, x[0])
            return {"value": v, "lambda": lam, "Q": Q}
        if f == "wasserstein_worst_case":
            v, lam, Q = kk.wasserstein_worst_case(m, a, x[0], unseen=np.array(inp["unseen"], dtype=float), cost=inp["cost"])
            return {"value": v, "lambda": lam, "Q": Q}
        raise ValueError(f"unknown diagnostic {f}")
    raise ValueError(f"unknown case kind {kind}")


def case_error(got, expected):
    err = 0.0
    for k, e in expected.items():
        if k not in got:
            return float("inf")
        g, e = np.atleast_1d(np.asarray(got[k], dtype=float)), np.atleast_1d(np.asarray(e, dtype=float))
        if g.shape != e.shape:
            return float("inf")
        err = max(err, float(np.max(np.abs(g - e))))
    return err
