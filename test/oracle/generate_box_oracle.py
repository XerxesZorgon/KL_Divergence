"""Offline oracle for capped problems (ADR-014). Runs by hand only; never called from Pkg.test().

    test/oracle/.venv/Scripts/python test/oracle/generate_box_oracle.py

Writes test/oracle/box_oracle.toml (LF line endings). Exit codes: 0 ok, 1 rejection rate above 10% (nothing written),
2 requirements.txt is not committed or has uncommitted changes.

Dependencies: numpy and cvxpy only (Clarabel through cvxpy). The TOML is written by a small hand-written writer.
"""
import datetime
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import clarabel
import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REQ = "test/oracle/requirements.txt"
OUT = ROOT / "test" / "oracle" / "box_oracle.toml"

SEED = 71
TAU_O = 1e-8
MARGIN = 1e-6
N_ALLOC = 200
N_KL = 100
MAX_REJECT_RATE = 0.10
SOLVER_OPTS = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)


# ----------------------------------------------------------------------------------------------- guard
def guard():
    r = subprocess.run(["git", "ls-files", "--error-unmatch", REQ], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"{REQ} is not tracked by git; commit it before generating.\n")
        sys.exit(2)
    r = subprocess.run(["git", "status", "--porcelain", "--", REQ], cwd=ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        sys.stderr.write(f"{REQ} has uncommitted changes; commit or restore it before generating.\n")
        sys.exit(2)
    return hashlib.sha256((ROOT / REQ).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- residual (ADR-013, DESIGN 10.2)
def residual(X, w, a, u, tau=TAU_O):
    """rho_u of allocation a under weights w. Returns dict with a (clipped), r, U, Z, F (0-based), nu_star, rho_u, degenerate."""
    a = np.clip(a, 0.0, u)
    V = X @ a
    if np.any(V <= 0):
        return dict(a=a, rho_u=math.inf)
    r = X.T @ (w / V)
    J = len(a)
    U = [j for j in range(J) if math.isfinite(u[j]) and a[j] >= u[j] - tau]
    Uset = set(U)
    Z = [j for j in range(J) if j not in Uset and a[j] <= tau]
    Zset = set(Z)
    F = [j for j in range(J) if j not in Uset and j not in Zset]
    A = max((r[j] for j in F + Z), default=-math.inf)
    B = min((r[j] for j in F + U), default=math.inf)
    both = math.isfinite(A) and math.isfinite(B)
    nu_part = max(0.0, (A - B) / 2) if both else 0.0
    nu_star = (A + B) / 2 if both else (B if math.isfinite(B) else A)
    rho = max(nu_part, max(0.0, -float(np.min(a))), float(np.max(np.maximum(0.0, a - u))), abs(float(np.sum(a)) - 1.0))
    nondeg = all(r[j] <= nu_star - MARGIN for j in Z) and all(r[j] >= nu_star + MARGIN for j in U)
    return dict(a=a, r=r, U=U, Z=Z, F=F, nu_star=float(nu_star), rho_u=float(rho), degenerate=not nondeg)


# ----------------------------------------------------------------------------------------------- solvers
def _ok(prob):
    return prob.status in ("optimal", "optimal_inaccurate")


def solve_allocation(X, pi, u):
    S, J = X.shape
    a = cp.Variable(J, nonneg=True)
    t = cp.Variable(S)
    cons = [cp.sum(a) == 1, cp.constraints.ExpCone(t, np.ones(S), X @ a)]
    for j in range(J):
        if math.isfinite(u[j]):
            cons.append(a[j] <= u[j])
    prob = cp.Problem(cp.Maximize(pi @ t), cons)
    try:
        prob.solve(solver=cp.CLARABEL, **SOLVER_OPTS)
    except cp.error.SolverError:
        return None
    return np.asarray(a.value, dtype=float) if _ok(prob) and a.value is not None else None


def solve_kl(X, pi, u, eps):
    """Joint program of ADR-005 with caps: max tau - lam*eps."""
    S, J = X.shape
    a = cp.Variable(J, nonneg=True)
    t = cp.Variable(S)
    w = cp.Variable(S)
    tau = cp.Variable()
    lam = cp.Variable(nonneg=True)
    cons = [cp.sum(a) == 1,
            cp.constraints.ExpCone(t, np.ones(S), X @ a),                 # t_s <= log V_s
            cp.constraints.ExpCone(tau - t, lam * np.ones(S), w),         # lam exp((tau - t_s)/lam) <= w_s
            pi @ w <= lam]
    for j in range(J):
        if math.isfinite(u[j]):
            cons.append(a[j] <= u[j])
    prob = cp.Problem(cp.Maximize(tau - lam * eps), cons)
    try:
        prob.solve(solver=cp.CLARABEL, **SOLVER_OPTS)
    except cp.error.SolverError:
        return None
    if not _ok(prob) or a.value is None or lam.value is None:
        return None
    return np.asarray(a.value, dtype=float), float(lam.value)


def logsumexp(x):
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


# ----------------------------------------------------------------------------------------------- random problems
def random_model(rng, force_cash=None):
    S = int(rng.integers(4, 65))
    n = int(rng.integers(1, 10))
    has_cash = bool(rng.random() < 0.5) if force_cash is None else force_cash
    pi = rng.dirichlet(np.ones(S))
    bets = np.zeros((S, n))
    for s in range(S):
        while True:
            row = np.where(rng.random(n) < 0.6, 0.0, 1.0 + 4.0 * rng.random(n))
            if has_cash or np.any(row > 0):          # every row needs at least one positive entry
                bets[s] = row
                break
    X = np.hstack([np.ones((S, 1)), bets]) if has_cash else bets
    return pi, X, has_cash


def random_caps(rng, X, pi, has_cash):
    """Solve the uncapped problem and cap a random non-empty subset of bets with stake > 0.03. None if no caps can be built."""
    S, J = X.shape
    a0 = solve_allocation(X, pi, np.full(J, math.inf))
    if a0 is None:
        return None
    bets = list(range(1 if has_cash else 0, J))
    cand = [j for j in bets if a0[j] > 0.03]
    if not cand:
        return None
    k = int(rng.integers(1, len(cand) + 1))
    sub = [int(j) for j in rng.choice(cand, size=k, replace=False)]
    if not has_cash and len(sub) == J:               # keep at least one asset uncapped without cash
        sub.pop(int(rng.integers(0, len(sub))))
    if not sub:
        return None
    u = np.full(J, math.inf)
    for j in sub:
        u[j] = rng.uniform(0.02, min(0.3, 0.9 * a0[j]))
    return u


def binds(a, u):
    return any(math.isfinite(u[j]) and a[j] >= u[j] - TAU_O for j in range(len(u)))


# ----------------------------------------------------------------------------------------------- generation
def gen_allocation(rng, counts):
    cases = []
    while counts["attempted"] < N_ALLOC:
        pi, X, has_cash = random_model(rng)
        u = random_caps(rng, X, pi, has_cash)
        if u is None:
            counts["redrawn_setup"] += 1
            continue
        a = solve_allocation(X, pi, u)
        if a is not None and not binds(np.clip(a, 0.0, u), u):
            counts["redrawn_nonbinding"] += 1
            continue
        counts["attempted"] += 1
        res = residual(X, pi, a, u) if a is not None else dict(rho_u=math.inf)
        if not res["rho_u"] <= 1e-8:
            counts["rejected"] += 1
            continue
        a = res["a"]
        counts["degenerate"] += int(res["degenerate"])
        cases.append(dict(id=f"alloc-{len(cases) + 1:03d}", kind="allocation", pi=pi, X=X, upper=u, has_cash=has_cash,
                          a=a, growth=float(pi @ np.log(X @ a)), nu_star=res["nu_star"], U=res["U"], Z=res["Z"],
                          F=res["F"], rho_u=res["rho_u"], degenerate=res["degenerate"]))
    return cases


def gen_kl(rng, counts):
    cases = []
    while counts["attempted"] < N_KL:
        pi, X, has_cash = random_model(rng, force_cash=True)
        u = random_caps(rng, X, pi, has_cash)
        eps = float(math.exp(rng.uniform(math.log(1e-3), math.log(0.5))))
        if u is None:
            counts["redrawn_setup"] += 1
            continue
        sol = solve_kl(X, pi, u, eps)
        if sol is not None:
            a, lam = sol
            ac = np.clip(a, 0.0, u)
            if ac[0] >= 1 - TAU_O or lam <= 1e-9 or not binds(ac, u):
                counts["redrawn_kl"] += 1
                continue
        counts["attempted"] += 1
        if sol is None:
            counts["rejected"] += 1
            continue
        V = X @ ac
        if np.any(V <= 0):
            counts["rejected"] += 1
            continue
        lw = np.log(pi) - np.log(V) / lam
        lse = logsumexp(lw)
        Q = np.exp(lw - lse)
        wc = -lam * lse - lam * eps
        D = float(np.sum(Q * (np.log(Q) - np.log(pi))))
        res = residual(X, Q, a, u)
        if not (res["rho_u"] <= 1e-8 and abs(D - eps) <= 1e-8):
            counts["rejected"] += 1
            continue
        counts["degenerate"] += int(res["degenerate"])
        cases.append(dict(id=f"kl-{len(cases) + 1:03d}", kind="kl", pi=pi, X=X, upper=u, has_cash=True, eps=eps,
                          a=res["a"], worst_case_growth=float(wc), **{"lambda": float(lam)}, nu_star=res["nu_star"],
                          U=res["U"], Z=res["Z"], F=res["F"], rho_u=res["rho_u"], degenerate=res["degenerate"]))
    return cases


# ----------------------------------------------------------------------------------------------- TOML writer
def tf(x):
    return repr(float(x))                              # repr gives 'inf' for infinity, shortest round-trip otherwise


def tv(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return tf(v)
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(tv(x) for x in v) + "]"
    raise TypeError(type(v))


ORDER = ["id", "kind", "pi", "X", "upper", "has_cash", "eps", "a", "growth", "worst_case_growth", "lambda", "nu_star",
         "U", "Z", "F", "rho_u", "degenerate"]


def write_toml(path, header, cases):
    out = ["# Box oracle fixtures (ADR-014). Generated by test/oracle/generate_box_oracle.py; do not edit by hand.",
           "# Set indices U, Z, F are 1-based.", "[header]"]
    for k, v in header["top"].items():
        out.append(f"{k} = {tv(v)}")
    for kind in ("allocation", "kl"):
        out.append(f"\n[header.{kind}]")
        for k, v in header[kind].items():
            out.append(f"{k} = {tv(v)}")
    for c in cases:
        out.append("\n[[cases]]")
        c = dict(c)
        for key in ("U", "Z", "F"):
            c[key] = [j + 1 for j in c[key]]
        c["X"] = [list(map(float, row)) for row in c["X"]]
        for k in ORDER:
            if k in c:
                out.append(f"{k} = {tv(c[k])}")
    path.write_bytes(("\n".join(out) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------------------------------- main
def main():
    sha = guard()
    rng = np.random.default_rng(SEED)
    stats = {}
    all_cases = []
    for kind, gen, n in (("allocation", gen_allocation, N_ALLOC), ("kl", gen_kl, N_KL)):
        counts = dict(attempted=0, rejected=0, redrawn_setup=0, degenerate=0)
        counts["redrawn_nonbinding" if kind == "allocation" else "redrawn_kl"] = 0
        cases = gen(rng, counts)
        rate = counts["rejected"] / counts["attempted"]
        stats[kind] = dict(attempted=counts["attempted"], rejected=counts["rejected"], rate=rate,
                           accepted=counts["attempted"] - counts["rejected"],
                           **{k: v for k, v in counts.items() if k.startswith("redrawn") or k == "degenerate"})
        print(kind, stats[kind], flush=True)
        if rate > MAX_REJECT_RATE:
            sys.stderr.write(f"{kind}: rejection rate {rate:.3f} > {MAX_REJECT_RATE}; nothing written.\n")
            sys.exit(1)
        all_cases += cases
    header = dict(top=dict(seed=SEED, python=sys.version.split()[0], cvxpy=cp.__version__, clarabel=clarabel.__version__,
                           numpy=np.__version__, date=datetime.date.today().isoformat(), tau_o=TAU_O, margin=MARGIN,
                           requirements_sha256=sha),
                  allocation=stats["allocation"], kl=stats["kl"])
    write_toml(OUT, header, all_cases)
    print(f"wrote {OUT} ({len(all_cases)} cases), requirements_sha256 = {sha}")


if __name__ == "__main__":
    main()
