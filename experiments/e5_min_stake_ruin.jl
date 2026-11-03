# E5 — Minimum stakes create a real absorbing barrier.
#
# (a) Classical gambler's ruin: unit even-money bets with a minimum stake of 1 unit, starting bankroll a units, stopping at
#     N = 20 units. Monte Carlo (`simulate_ruin`, fixed stake) vs `ruin_probability_classical(p, a, N)` for
#     p ∈ {0.48, 0.50, 0.52} and a ∈ {2, 5, 10, 15, 18}; 10 000 trials per cell.
# (b) Kelly with a minimum stake: even-money bet, p = 0.525, Kelly stake b* = 0.05 of wealth, minimum stake 1 unit. Once
#     wealth falls below 1/b* = 20 units the Kelly stake is below the minimum, every bet is skipped and the bettor is priced
#     out. Pr(priced out) from a start of w0 units is the drawdown law at x = 20/w0 with c = 1, i.e. x itself.
#     w0 ∈ {40, 100, 200} (x = 0.5, 0.2, 0.1); 4 000 trials each, stopped at 10⁹·w0 or 100 000 rounds.
# Pass (TEST_PLAN §G): every cell of (a) and (b) within 3 SE of its formula, with SE = √(P(1−P)/trials) at the formula's P
# (the SE under the hypothesis being tested; Fix 046a — the first run used the simulated P̂ and 1 000 trials in (b)).

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

even_money(p) = ScenarioModel([p, 1 - p], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)

function main(; seed = 2005, out_dir = joinpath(@__DIR__, "output"), quick = false,
               article = joinpath(@__DIR__, "..", "..", "img", "e5_ruin.svg"))
    t0 = time(); rng = StableRNG(seed)
    Na = quick ? 1_000 : 10_000; Nb = quick ? 200 : 4_000
    N = 20; ps = (0.48, 0.50, 0.52); as = [2, 5, 10, 15, 18]
    mkpath(out_dir); pass = true

    # (a) classical gambler's ruin
    rows_a = Tuple[]; series = NamedTuple[]
    for (i, p) in enumerate(ps)
        m = even_money(p); Ps = Float64[]
        for a in as
            sim = simulate_ruin(m, [1.0, 0.0]; rng, trials = Na, rounds = 1_000_000, w0 = Float64(a), min_stake = 1.0,
                                fixed_stake = 1.0, target = Float64(N))
            P = ruin_probability_classical(p, a, N); se = sqrt(P * (1 - P) / Na); z = (sim.p_ruin - P) / se
            abs(z) > 3 && (pass = false)
            push!(rows_a, (p, a, N, sim.p_ruin, se, P, z)); push!(Ps, sim.p_ruin)
            @printf("E5a: p = %.2f, a = %2d: simulated %.4f ± %.4f, classical %.4f, z = %+.2f\n", p, a, sim.p_ruin, se, P, z)
        end
        push!(series, (x = collect(0.0:1.0:N), y = [ruin_probability_classical(p, a, N) for a in 0:N],
                       label = @sprintf("p = %.2f: classical formula", p), kind = :line, color_index = i))
        push!(series, (x = Float64.(as), y = Ps, label = @sprintf("p = %.2f: simulated", p), kind = :points, color_index = i))
    end
    write_csv(joinpath(out_dir, "e5_classical.csv"), ["p", "a", "N", "p_sim", "se", "p_classical", "z"], rows_a)

    # (b) Kelly bettor priced out by a minimum stake
    p = 0.525; m = even_money(p); bstar = optimal_allocation(m).a[2]
    abs(bstar - (2p - 1)) ≤ 1e-10 || error("E5: Kelly stake $bstar != 2p − 1")
    floor_w = 1.0 / bstar                                           # wealth below which the Kelly stake < 1 unit
    rows_b = Tuple[]
    for w0 in (40.0, 100.0, 200.0)
        sim = simulate_ruin(m, [1 - bstar, bstar]; rng, trials = Nb, rounds = 100_000, w0, min_stake = 1.0, target = 1e9 * w0)
        x = floor_w / w0
        P = mean(sim.min_fraction .< x)
        L = drawdown_probability(x, 1.0); se = sqrt(L * (1 - L) / Nb); z = (P - L) / se
        abs(z) > 3 && (pass = false)
        push!(rows_b, (w0, x, P, se, L, sim.p_ruin, z))
        @printf("E5b: start %3.0f units (x = %.2f): priced out %.4f ± %.4f, drawdown law %.4f, ruined %.4f, z = %+.2f\n",
                w0, x, P, se, L, sim.p_ruin, z)
    end
    write_csv(joinpath(out_dir, "e5_priced_out.csv"), ["w0_units", "x", "p_priced_out", "se", "p_law", "p_ruin", "z"], rows_b)

    svg = svg_plot(joinpath(out_dir, "e5_ruin.svg"); series,
        xlim = (0.0, Float64(N)), ylim = (0.0, 1.0), xticks = collect(0.0:5.0:N), yticks = collect(0.0:0.2:1.0),
        xlabel = "Starting bankroll a (units of the minimum stake; stop at N = 20)", ylabel = "Pr(ruin)",
        legend_title = "E5: minimum-stake ruin")
    quick || (mkpath(dirname(article)); cp(svg, article; force = true))
    write_meta(joinpath(out_dir, "e5_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("trials_classical" => Na, "trials_kelly" => Nb, "pass" => pass))
    println("E5: ", pass ? "PASS" : "FAIL")
    return (; rows_a, rows_b, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
