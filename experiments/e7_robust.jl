# E7 — Kelly vs KL-robust vs Wasserstein-robust bets when the truth has an outcome the model has never seen.
#
# Model: cash and two risky assets A, B; four scenarios with probabilities π = (0.25, 0.30, 0.25, 0.20) and gross returns
#   A: 1.40, 1.10, 0.90, 0.60      B: 1.25, 1.05, 0.95, 0.80      (both have mean 1.025).
# Truth: with probability q a crash the model has never seen (A returns 0.20, B returns 0.60); otherwise the model is exactly
# right, i.e. truth_q = (1 − q)·π ⊕ q·crash, q ∈ {0, 0.005, 0.01, 0.02, 0.05}.
# Bettors (all exact, certified bets from the library):
#   Kelly         — `optimal_allocation` on the model (the Bayesian bet when the model is the posterior predictive);
#   KL, n         — `kl_robust_allocation` with ε = `kl_radius_chi2(4, n)` for n ∈ {200, 1 000, 5 000} observations;
#   W-L1, W-L∞    — `wasserstein_robust_allocation` with the crash as an `unseen` scenario and cost `:l1` / `:linf`; radius set so
#                   truth with q = 1% lies exactly on the ball's edge: ε = 0.01·Σ_s π_s C(s, crash). This radius is a
#                   judgment ("allow up to 1% mass on a crash"), not an estimate — the crash was never observed.
# (a) Table: exact growth under each truth, and simulated Pr(wealth ever falls to half) and median final wealth over 1 000
#     rounds (1 000 paths). (b) Figure: exact growth vs q. (c) KL-radius coverage: share of 10 000 samples of size
#     n ∈ {50, 200, 1 000} whose KL ball of radius `kl_radius_chi2(4, n)` around the empirical distribution contains the truth,
#     i.e. D(π‖p̂) ≤ ε; the G-test direction D(p̂‖π) ≤ ε is reported alongside.
# Pass (TEST_PLAN §G, Task 048): (i) the Wasserstein guarantee — for q ∈ {0, 0.005, 0.01} (truth inside the ball) each W bet's
# growth under the truth ≥ its certified worst-case growth − 1e-9; (ii) the KL guarantee at q = 0 (the only truth inside a KL
# ball — an unseen outcome has D = ∞) for each KL bet; (iii) KL-radius coverage D(π‖p̂) ≤ ε within ±2 pp of 95% for
# n ∈ {200, 1 000}. n = 50 is reported, not tested: the χ² radius is asymptotic, and the prototype (20 000 samples) put its
# coverage at ≈ 93.3% — on the edge of the band, where a pass/fail would be a coin flip rather than a check.

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

const Π = [0.25, 0.30, 0.25, 0.20]
const RET = [1.40 1.25; 1.10 1.05; 0.90 0.95; 0.60 0.80]
const CRASH = [0.20 0.60]
const ASSETS = ["cash", "A", "B"]

model() = ScenarioModel(Π, hcat(ones(4), RET); assets = ASSETS, has_cash = true, structure = :correlated)
truth(q) = q == 0 ? model() :
    ScenarioModel(vcat((1 - q) .* Π, q), vcat(hcat(ones(4), RET), hcat(1.0, CRASH)); assets = ASSETS, has_cash = true,
                  structure = :correlated)

function main(; seed = 2007, out_dir = joinpath(@__DIR__, "output"), quick = false,
               article = joinpath(@__DIR__, "..", "..", "img", "e7_robust.svg"))
    t0 = time(); rng = StableRNG(seed)
    trials = quick ? 200 : 1_000; Rcov = quick ? 1_000 : 10_000
    mkpath(out_dir); m = model(); Xs = m.X; Xc = hcat(1.0, CRASH)
    qs = [0.0, 0.005, 0.01, 0.02, 0.05]

    # the bets
    bettors = Tuple{String,Vector{Float64},Float64,Float64,Symbol}[]      # (name, a, worst-case growth, radius, certificate)
    rK = optimal_allocation(m); push!(bettors, ("Kelly (model)", rK.a, rK.growth, 0.0, :kkt))
    for n in (200, 1_000, 5_000)
        ε = kl_radius_chi2(4, n); r = kl_robust_allocation(m, ε)
        push!(bettors, ("KL, n = $n", r.a, r.worst_case_growth, ε, r.certificate))
    end
    for (cost, nm) in ((:l1, "W-L1"), (:linf, "W-L∞"))
        C = cost === :l1 ? [sum(abs.(Xs[s, :] .- Xc[1, :])) for s in 1:4] : [maximum(abs.(Xs[s, :] .- Xc[1, :])) for s in 1:4]
        ε = 0.01 * sum(Π .* C)
        r = wasserstein_robust_allocation(m, ε; unseen = Xc, cost)
        push!(bettors, (nm, r.a, r.worst_case_growth, ε, r.certificate))
    end
    (D_cash, Q_cash) = kl_cash_radius(m)
    cash_rows = vcat([("kl_cash_radius", D_cash)], [("q_min_s$s", Q_cash[s]) for s in eachindex(Q_cash)])
    write_csv(joinpath(out_dir, "e7_cash_radius.csv"), ["quantity", "value"], cash_rows)
    @printf("E7: KL cash radius of the model = %.5f (a KL ball at least this large makes the robust bet all cash)\n", D_cash)

    # (a) the table
    pass = true; rows = Tuple[]
    for (nm, a, wc, ε, cert) in bettors
        @printf("E7: %-13s ε = %.5f, a = (cash %.4f, A %.4f, B %.4f), worst-case growth %+.5f, certificate %s\n",
                nm, ε, a[1], a[2], a[3], wc, cert)
        csv_nm = replace(nm, "KL, n = " => "KL n=")
        for q in qs
            t = truth(q); g = growth(t, a)
            sim = simulate_ruin(t, a; rng, trials, rounds = 1_000)
            half = mean(sim.min_fraction .≤ 0.5)
            inside = startswith(nm, "W") ? q ≤ 0.01 : (startswith(nm, "KL") ? q == 0 : false)
            ok = !inside || g ≥ wc - 1e-9
            pass &= ok
            push!(rows, (csv_nm, ε, a[1], a[2], a[3], wc, q, g, half, sim.final_q[2], inside, ok))
            @printf("E7:   q = %.3f: growth %+.5f, Pr(halved) %.3f, median final wealth %.3g%s\n", q, g, half, sim.final_q[2],
                    inside ? (ok ? "  [guarantee holds]" : "  [GUARANTEE VIOLATED]") : "")
        end
    end
    write_csv(joinpath(out_dir, "e7_table.csv"),
              ["bettor", "radius", "a_cash", "a_A", "a_B", "worst_case_growth", "q_crash", "growth", "p_halved",
               "median_final", "truth_in_ball", "guarantee_ok"], rows)

    # (b) the figure: exact growth vs q
    qgrid = collect(0.0:0.001:0.03)
    series = NamedTuple[]
    for (k, i) in enumerate((1, 3, 4, 5))
        nm, a = bettors[i][1], bettors[i][2]
        push!(series, (x = qgrid, y = [1e3 * growth(truth(q), a) for q in qgrid], label = nm, kind = :line, color_index = k))
    end
    svg = svg_plot(joinpath(out_dir, "e7_robust.svg"); series,
        xlim = (0.0, 0.03), ylim = (-6.0, 14.0), xticks = collect(0.0:0.01:0.03), yticks = collect(-6.0:2.0:14.0),
        xlabel = "Probability q of the unseen crash", ylabel = "Growth per period under the truth (×10⁻³ nats)",
        hlines = [(0.0, "")], vlines = [(0.01, "W radius")], legend_title = "E7: robust bets")

    # (c) KL-radius coverage
    cum = cumsum(Π); cov_rows = Tuple[]
    for n in (50, 200, 1_000)
        ε = kl_radius_chi2(4, n); fwd = 0; rev = 0
        for _ in 1:Rcov
            cnt = zeros(4)
            for _ in 1:n; cnt[min(searchsortedfirst(cum, rand(rng) * cum[end]), 4)] += 1; end
            ph = cnt ./ n
            fwd += kl(Π, ph) ≤ ε; rev += kl(ph, Π) ≤ ε
        end
        cf, cr = fwd / Rcov, rev / Rcov; tested = n ≥ 200; ok = !tested || abs(cf - 0.95) ≤ 0.02; pass &= ok
        push!(cov_rows, (n, ε, cf, cr, tested, ok))
        @printf("E7: KL radius coverage, n = %4d (ε = %.5f): truth in ball %.4f, G-test direction %.4f%s\n",
                n, ε, cf, cr, !tested ? " (reported only)" : (ok ? "" : "  <-- outside 93–97%"))
    end
    write_csv(joinpath(out_dir, "e7_kl_coverage.csv"), ["n", "radius", "coverage_truth_in_ball", "coverage_gtest", "tested", "pass"], cov_rows)

    quick || (mkpath(dirname(article)); cp(svg, article; force = true))
    write_meta(joinpath(out_dir, "e7_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("trials" => trials, "coverage_replications" => Rcov, "pass" => pass))
    println("E7: ", pass ? "PASS" : "FAIL")
    return (; rows, cov_rows, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
