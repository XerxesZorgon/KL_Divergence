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
# Refactored in Task 012 to call Kelly_KL; outputs byte-identical to tag e1-original

using Random, Printf, LinearAlgebra
using Kelly_KL

# E1's published `kkt` column uses only the first KKT term; the package certifies the full residual.
e1_kkt(X, a, q) = maximum(X' * (q ./ (X * a))) - 1.0
e1_alloc(q, ohat; cash::Bool) = Kelly_KL.optimal_allocation(q, ohat; cash).a

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
    X = single_event(p, ohat; cash).X
    ap = e1_alloc(p, ohat; cash); kp = e1_kkt(X, ap, p)
    aq = e1_alloc(q, ohat; cash); kq = e1_kkt(X, aq, q)
    Gopt, Gq = growth(X, ap, p), growth(X, aq, p)
    Trial(case, n, σ, ρ, c, kl(p, q), Gopt - Gq, Gopt, Gq,
          cash ? ap[1] : 0.0, cash ? aq[1] : 0.0, max(kp, kq))
end

function main(; seed=20261101, trials_per_cell=200,
               σs=(0.05, 0.1, 0.2, 0.4, 0.8), ns=(3, 4, 6),
               out=joinpath(@__DIR__, "output", "e1_results.csv"))
    rng = Xoshiro(seed)
    cases = ("A_full_investment", "A2_full_inv_fees", "B_cash", "B2_cash_margin",
             "C_cash_margin_fee", "D_cash_superfair")
    rows = Trial[]
    for case in cases, n in ns, σ in σs, _ in 1:trials_per_cell
        push!(rows, run_trial(rng, case; n, σ))
    end

    mkpath(dirname(out))
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
        X = single_event(p, ohat; cash=true).X
        ap = e1_alloc(p, ohat; cash=true); kp = e1_kkt(X, ap, p)
        aq = e1_alloc(q, ohat; cash=true); kq = e1_kkt(X, aq, q)
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
