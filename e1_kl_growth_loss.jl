# E1 — Is the growth lost to a wrong distribution equal to (or bounded by) D(p‖q)?
#
# Extended Kelly II, experiment E1.  Standard library only (Random, Printf, LinearAlgebra).
#
# Setting: n mutually exclusive outcomes i = 1..n.  The true probabilities are p, the
# bettor's belief is q.  A stake on outcome i returns the gross payoff ô_i per unit if i
# occurs and 0 otherwise.  Cash (optional) returns 1 in every outcome.
#
#   wealth in outcome i:   V_i(a) = a_0 + a_i ô_i          (a_0 = cash, a_i = stake on i)
#   growth under p:        G_p(a) = Σ_i p_i log V_i(a),     a ≥ 0,  Σ_j a_j = 1
#
# The bettor chooses a_q = argmax G_q, the truth-optimal choice is a_p = argmax G_p, and the
# growth lost to the wrong belief is
#
#   L(p,q) = G_p(a_p) − G_p(a_q)  ≥ 0.
#
# Cover & Thomas (Thm 6.1.2): with no cash (full investment), a_q = q for ANY payoffs ô, so
# L = D(p‖q) exactly.  Fees that scale payoffs do not change this.  E1 checks that identity
# numerically and then measures L against D(p‖q) when cash is allowed, with a bookmaker
# margin and a commission on net winnings.
#
# Payoff model:  market probabilities m (≠ p, so an edge exists somewhere), overround ρ,
#   decimal odds        o_i = 1 / (m_i (1 + ρ))
#   after commission c  ô_i = 1 + (o_i − 1)(1 − c)
#
# Outputs:  e1_results.csv (one row per trial) and a printed summary.
# Plot with e1_plot.jl.

using Random, Printf, LinearAlgebra

# ---------------------------------------------------------------------------------------
# Cross-check solver (not used for the main results — it stalls when a weight underflows
# to zero, so `optimal_allocation` below uses the exact solution instead).
# Log-optimal allocation via Cover's multiplicative update (Cover 1984, "An algorithm for
# maximizing expected log investment return").  For returns matrix X (outcomes × assets)
# and weights a on the simplex, the update a_j ← a_j · Σ_i q_i X_ij / (X a)_i keeps a on
# the simplex and increases G_q monotonically.  Optimality (KKT): r_j = Σ_i q_i X_ij/(X a)_i
# satisfies r_j ≤ 1 for all j, with equality where a_j > 0.  We stop when max_j r_j − 1 < tol.
# ---------------------------------------------------------------------------------------
"""
    log_optimal(X, q; tol=1e-11, maxiter=2_000_000) -> (a, kkt, iters)

Maximize Σ_i q_i log((X a)_i) over the probability simplex.
"""
function log_optimal(X::AbstractMatrix, q::AbstractVector; tol=1e-11, maxiter=2_000_000)
    nA = size(X, 2)
    a = fill(1.0 / nA, nA)
    r = similar(a)
    kkt = Inf
    it = 0
    for k in 1:maxiter
        it = k
        V = X * a
        r .= X' * (q ./ V)
        kkt = maximum(r) - 1.0
        kkt < tol && break
        a .*= r
        a ./= sum(a)              # guards against round-off drift
    end
    return a, kkt, it
end

"""
    kelly_exact(q, ohat) -> a = [cash; stakes]

Exact log-optimal allocation for mutually exclusive outcomes with cash available
(Kelly 1956; Smoczynski & Tomkins 2010, "An explicit solution to the problem of
optimizing the allocations of a bettor's wealth when wagering on horse races").
Sort outcomes by expected return e_i = q_i ô_i.  Grow the bet set S greedily while
e_i > R(S), where the reserve rate is
    R(S) = (1 − Σ_{i∈S} q_i) / (1 − Σ_{i∈S} 1/ô_i).
Then stake f_i = q_i − R/ô_i on i ∈ S and keep 1 − Σ f_i in cash.
"""
function kelly_exact(q::AbstractVector, ohat::AbstractVector)
    n = length(q)
    order = sortperm(q .* ohat; rev=true)
    S = Int[]; pS = 0.0; σS = 0.0; R = 1.0
    for i in order
        q[i] * ohat[i] > R || break
        push!(S, i); pS += q[i]; σS += 1 / ohat[i]
        R = (pS >= 1 - 1e-15 || σS >= 1) ? 0.0 : (1 - pS) / (1 - σS)
    end
    f = zeros(n)
    for i in S
        f[i] = max(q[i] - R / ohat[i], 0.0)
    end
    return vcat(max(1 - sum(f), 0.0), f)
end

"""KKT residual max_j (r_j − 1), r = Xᵀ(q ./ Xa); ≤ 0 (to round-off) at the optimum."""
kkt_residual(X, a, q) = maximum(X' * (q ./ (X * a))) - 1.0

"""
    optimal_allocation(q, ohat; cash) -> (a, kkt)

No cash: a = q exactly for any payoffs (Cover & Thomas Thm 6.1.2).
Cash:    exact greedy solution `kelly_exact`.
Both are checked against the KKT conditions of the log-optimal problem.
"""
function optimal_allocation(q, ohat; cash::Bool)
    X = returns_matrix(ohat; cash)
    a = cash ? kelly_exact(q, ohat) : copy(q)
    return a, kkt_residual(X, a, q)
end

"""Returns matrix: rows = outcomes, columns = [cash (optional); stake on each outcome]."""
function returns_matrix(ohat::AbstractVector; cash::Bool)
    n = length(ohat)
    S = Matrix(Diagonal(ohat))
    return cash ? hcat(ones(n), S) : S
end

growth(X, a, p) = sum(p .* log.(X * a))
kl(p, q) = sum(p[i] > 0 ? p[i] * log(p[i] / q[i]) : 0.0 for i in eachindex(p))

dirichlet1(rng, n) = (x = randexp(rng, n); x ./ sum(x))
tilt(rng, p, σ) = (x = p .* exp.(σ .* randn(rng, length(p))); x ./ sum(x))

# ---------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------
struct Trial
    case::String; n::Int; σ::Float64; ρ::Float64; c::Float64
    kl::Float64; loss::Float64; Gp_opt::Float64; Gp_q::Float64
    cash_p::Float64; cash_q::Float64; kkt::Float64
end

function run_trial(rng, case; n, σ, τ=0.3)
    p = dirichlet1(rng, n)
    m = tilt(rng, p, τ)                        # market's view
    q = tilt(rng, p, σ)                        # bettor's (wrong) view
    ρ, c, cash = case == "A_full_investment"   ? (0.0,  0.0,  false) :
                 case == "A2_full_inv_fees"    ? (0.10, 0.05, false) :
                 case == "B_cash"              ? (0.0,  0.0,  true)  :
                 case == "B2_cash_margin"      ? (0.10, 0.0,  true)  :
                 case == "C_cash_margin_fee"   ? (0.10, 0.05, true)  :
                 case == "D_cash_superfair"    ? (-0.05, 0.0, true)  :
                 error("unknown case $case")
    o = 1.0 ./ (m .* (1 + ρ))
    ohat = 1.0 .+ (o .- 1.0) .* (1 - c)
    X = returns_matrix(ohat; cash)
    ap, kp = optimal_allocation(p, ohat; cash)
    aq, kq = optimal_allocation(q, ohat; cash)
    Gopt, Gq = growth(X, ap, p), growth(X, aq, p)
    Trial(case, n, σ, ρ, c, kl(p, q), Gopt - Gq, Gopt, Gq,
          cash ? ap[1] : 0.0, cash ? aq[1] : 0.0, max(kp, kq))
end

function main(; seed=20261101, trials_per_cell=200,
               σs=(0.05, 0.1, 0.2, 0.4, 0.8), ns=(3, 4, 6),
               out=joinpath(@__DIR__, "e1_results.csv"))
    rng = Xoshiro(seed)
    cases = ("A_full_investment", "A2_full_inv_fees", "B_cash", "B2_cash_margin",
             "C_cash_margin_fee", "D_cash_superfair")
    rows = Trial[]
    for case in cases, n in ns, σ in σs, _ in 1:trials_per_cell
        push!(rows, run_trial(rng, case; n, σ))
    end

    open(out, "w") do io
        println(io, "case,n,sigma,overround,commission,kl,loss,Gp_opt,Gp_q,cash_p,cash_q,kkt")
        for t in rows
            @printf(io, "%s,%d,%.3f,%.3f,%.3f,%.12e,%.12e,%.12e,%.12e,%.6f,%.6f,%.3e\n",
                    t.case, t.n, t.σ, t.ρ, t.c, t.kl, t.loss, t.Gp_opt, t.Gp_q,
                    t.cash_p, t.cash_q, t.kkt)
        end
    end

    println("E1 summary  (seed = $seed, $(length(rows)) trials)")
    @printf("%-20s %8s %12s %10s %10s %10s %8s\n",
            "case", "trials", "max|L−D|", "L≤D (%)", "median L/D", "max L/D", "G_p(a_q)<0")
    for case in cases
        R = filter(t -> t.case == case, rows)
        ratio = [t.loss / t.kl for t in R if t.kl > 1e-12]
        @printf("%-20s %8d %12.2e %10.1f %10.3f %10.3f %8d\n", case, length(R),
                maximum(abs(t.loss - t.kl) for t in R),
                100 * count(t -> t.loss <= t.kl + 1e-10, R) / length(R),
                sort(ratio)[cld(length(ratio), 2)], maximum(ratio),
                count(t -> t.Gp_q < 0, R))
    end
    @printf("worst KKT residual: %.2e  (≤ ~1e-12 means every allocation is optimal)\n",
            maximum(t.kkt for t in rows))
    println("wrote $out")
    return rows
end

"""
    stress(; seed, trials) -> (violations, max L/D, worst KKT)

Wide random sweep with cash: n ∈ 2:12, belief error σ ∈ [0,2], market error τ ∈ [0,1.5],
overround ρ ∈ [0,0.3], commission c ∈ [0,0.2].  Counts trials with L > D(p‖q).
"""
function stress(; seed=7, trials=50_000)
    rng = Xoshiro(seed)
    viol = 0; worst = 0.0; kmax = -Inf
    for _ in 1:trials
        n = rand(rng, 2:12); σ = 2rand(rng); τ = 1.5rand(rng); ρ = 0.3rand(rng); c = 0.2rand(rng)
        p = dirichlet1(rng, n); m = tilt(rng, p, τ); q = tilt(rng, p, σ)
        ohat = 1.0 .+ (1.0 ./ (m .* (1 + ρ)) .- 1.0) .* (1 - c)
        X = returns_matrix(ohat; cash=true)
        ap, kp = optimal_allocation(p, ohat; cash=true)
        aq, kq = optimal_allocation(q, ohat; cash=true)
        L = growth(X, ap, p) - growth(X, aq, p); D = kl(p, q)
        viol += L > D + 1e-10
        worst = max(worst, L / D); kmax = max(kmax, kp, kq)
    end
    @printf("stress: %d trials, violations of L ≤ D: %d, max L/D = %.6f, worst KKT = %.2e\n",
            trials, viol, worst, kmax)
    return viol, worst, kmax
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    stress()
end
