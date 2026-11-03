# E4 — The overbetting cliff and the drawdown law.
#
# One even-money bet with win probability p = 0.525 (a 5% edge); the exact Kelly stake is b* = 2p − 1 = 0.05.
# (a) Cliff: exact long-run growth per bet when staking c·b*, c ∈ [0, 3], with the zero-growth point (≈ 2× Kelly).
# (b) Drawdown law: for c ∈ {0.5, 1, 1.5}, simulate proportional betting (`simulate_ruin`, stopped once wealth reaches 10⁹,
#     where a later fall to x·w0 has probability (x/10⁹)^(2/c−1) < 5e-4) and compare Pr(wealth ever falls to x·w0) with
#     the continuous-limit law x^(2/c−1). c is a diagnostic only (the stake a mis-stated edge produces), never a sizing knob.
# Also reported: the discrete exponent k solving p(1+f)^−k + (1−p)(1−f)^−k = 1, f = c·b* — the exact large-drawdown exponent
# of the discrete game (Lundberg), to show the continuous law's exponent is not the source of any gap.
# Pass (TEST_PLAN §G, Task 045): (i) |c0 − 2| ≤ 0.01; (ii) |simulated − law| ≤ 3 SE for every c and every x ≤ 0.5.
# Rows with x > 0.5 are reported, not tested: a drawdown of 10–30% is only a few losing bets of 2.5–7.5%, so the
# simulated wealth jumps past x (overshoot) and the probability is biased low — a discreteness effect, not a law failure.

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

function discrete_exponent(p, f)
    h(k) = p * (1 + f)^(-k) + (1 - p) * (1 - f)^(-k) - 1        # h < 0 for 0 < k < k*, h(k*) = 0
    lo, hi = 1e-9, 50.0
    for _ in 1:200
        mid = (lo + hi) / 2
        h(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function main(; seed = 2004, out_dir = joinpath(@__DIR__, "output"), quick = false,
               article_dir = joinpath(@__DIR__, "..", "..", "img"))
    t0 = time(); rng = StableRNG(seed); N = quick ? 200 : 2000
    p = 0.525
    m = ScenarioModel([p, 1 - p], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    a_star = optimal_allocation(m).a; bstar = a_star[2]
    abs(bstar - (2p - 1)) ≤ 1e-10 || error("E4: Kelly stake $bstar != 2p − 1 = $(2p - 1)")
    mkpath(out_dir)

    # (a) the cliff
    cs = collect(0.0:0.025:3.0)
    G = growth_curve(m, a_star, cs)
    zg = zero_growth_point(m, a_star)
    pass_cliff = abs(zg.c0 - 2.0) ≤ 0.01
    write_csv(joinpath(out_dir, "e4_cliff.csv"), ["c", "growth_per_bet"], [(c, g) for (c, g) in zip(cs, G)])
    cliff = svg_plot(joinpath(out_dir, "e4_cliff.svg");
        series = [(x = cs, y = 1e3 .* G, label = "Long-run growth per bet (5% edge, even money)", kind = :line)],
        xlim = (0.0, 3.0), ylim = (-4.0, 2.0), xticks = collect(0.0:0.5:3.0), yticks = collect(-4.0:1.0:2.0),
        xlabel = "Stake as a multiple c of the Kelly stake", ylabel = "Growth per bet (×10⁻³ nats)",
        hlines = [(0.0, "")], vlines = [(1.0, "Kelly"), (zg.c0, "zero growth")], legend_title = "E4: overbetting cliff")

    # (b) the drawdown law
    xs = [0.9, 0.7, 0.5, 0.3, 0.1]; cvals = (0.5, 1.0, 1.5)
    rows = Tuple[]; series = NamedTuple[]; pass_dd = true
    xgrid = collect(0.01:0.01:0.99)
    for (i, c) in enumerate(cvals)
        f = c * bstar; k = discrete_exponent(p, f)
        sim = simulate_ruin(m, [1 - f, f]; rng, trials = N, rounds = 100_000, target = 1e9)
        sim.p_ruin == 0.0 || error("E4: proportional betting was ruined (c = $c)")
        Ps = Float64[]
        for x in xs
            P = mean(sim.min_fraction .<= x); se = sqrt(max(P * (1 - P), 1 / N) / N)
            L = drawdown_probability(x, c); z = (P - L) / se
            tested = x ≤ 0.5
            tested && abs(z) > 3 && (pass_dd = false)
            push!(rows, (c, x, P, se, L, x^k, z, tested)); push!(Ps, P)
            @printf("E4: c = %.1f, x = %.1f: simulated %.4f ± %.4f, law %.4f, discrete-exponent %.4f, z = %+.2f%s\n",
                    c, x, P, se, L, x^k, z, tested ? "" : " (reported only)")
        end
        push!(series, (x = xgrid, y = [drawdown_probability(x, c) for x in xgrid], label = @sprintf("c = %.1f: law x^(2/c−1)", c),
                       kind = :line, color_index = i))
        push!(series, (x = xs, y = Ps, label = @sprintf("c = %.1f: simulated", c), kind = :points, color_index = i))
    end
    write_csv(joinpath(out_dir, "e4_drawdown.csv"),
              ["c", "x", "p_sim", "se", "p_law", "p_discrete_exponent", "z", "tested"], rows)
    dd = svg_plot(joinpath(out_dir, "e4_drawdown.svg"); series,
        xlim = (0.0, 1.0), ylim = (0.0, 1.0), xticks = collect(0.0:0.2:1.0), yticks = collect(0.0:0.2:1.0),
        xlabel = "Drawdown level x (fraction of starting wealth)", ylabel = "Pr(wealth ever falls to x)",
        legend_title = "E4: drawdown law")

    pass = pass_cliff && pass_dd
    if !quick
        mkpath(article_dir)
        cp(cliff, joinpath(article_dir, "e4_cliff.svg"); force = true)
        cp(dd, joinpath(article_dir, "e4_drawdown.svg"); force = true)
    end
    write_meta(joinpath(out_dir, "e4_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("trials" => N, "c0" => zg.c0, "kelly_stake" => bstar, "pass" => pass))
    @printf("E4: zero-growth point c0 = %.6f (quadratic approximation 2)\n", zg.c0)
    println("E4: ", pass ? "PASS" : "FAIL")
    return (; c0 = zg.c0, rows, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
