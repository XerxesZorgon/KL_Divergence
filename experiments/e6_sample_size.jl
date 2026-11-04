# E6 — How much data is needed to measure an edge?
#
# A bet pays w (gross) with probability p; the edge is e = p·w − 1. From n outcomes, ê = p̂·w − 1 with standard error
# w·√(p̂(1−p̂)/n), and the Wald 95% interval ê ± 1.96·SE.
# (a) Coverage at n = ⌈n_detect(p, w)⌉ (the n at which the interval half-width equals the edge): for w ∈ {2, 3, 10, 50} and
#     e = 0.05 (p = 1.05/w), 10 000 replications each. Also reported: the share of intervals that exclude zero (≈ 50% by
#     construction — n_detect is where an edge is detected only half the time) and the 2.5–97.5% range of the estimated Kelly
#     stake b̂* = max(ê, 0)/(w − 1) relative to the true b* = e/(w − 1).
# (b) Figure: for w = 10, the median and 2.5–97.5% range of b̂*/b* as n grows from n_detect to n_relative (1 SE = 10% of e).
# Pass (TEST_PLAN §G): (a) coverage in [94%, 96%] for every w.

using Random, Statistics, Printf, StableRNGs, Distributions, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

function replicate(rng, p, w, n, R)
    e = p * w - 1; bstar = e / (w - 1)
    cover = 0; detect = 0; ratio = zeros(R)
    for r in 1:R
        ph = rand(rng, Binomial(n, p)) / n
        eh = ph * w - 1; se = w * sqrt(ph * (1 - ph) / n)
        cover += abs(eh - e) ≤ 1.96 * se
        detect += eh - 1.96 * se > 0
        ratio[r] = max(eh, 0.0) / (w - 1) / bstar
    end
    return (coverage = cover / R, detected = detect / R, ratio = ratio)
end

function main(; seed = 2006, out_dir = joinpath(@__DIR__, "output"), quick = false,
               article = joinpath(@__DIR__, "..", "..", "img", "e6_sample_size.svg"))
    t0 = time(); rng = StableRNG(seed); R = quick ? 1_000 : 10_000; Rf = quick ? 500 : 2_000
    e = 0.05; ws = (2.0, 3.0, 10.0, 50.0)
    mkpath(out_dir); pass = true; rows = Tuple[]

    for w in ws
        p = (1 + e) / w; n = ceil(Int, n_detect(p, w)); nrel = ceil(Int, n_relative(p, w))
        res = replicate(rng, p, w, n, R)
        q = quantile(res.ratio, [0.025, 0.975])
        ok = 0.94 ≤ res.coverage ≤ 0.96; pass &= ok
        push!(rows, (w, p, n, nrel, res.coverage, res.detected, q[1], q[2], ok))
        @printf("E6: w = %4.1f, p = %.4f, n_detect = %6d, n_relative = %7d: coverage %.4f, edge detected %.3f, b̂*/b* 95%% range %.2f–%.2f%s\n",
                w, p, n, nrel, res.coverage, res.detected, q[1], q[2], ok ? "" : "  <-- outside 94–96%")
    end
    write_csv(joinpath(out_dir, "e6_coverage.csv"),
              ["w", "p", "n_detect", "n_relative", "coverage", "detected", "kelly_ratio_q025", "kelly_ratio_q975", "pass"], rows)

    # (b) Kelly-stake uncertainty vs sample size, w = 10
    w = 10.0; p = (1 + e) / w; n0 = n_detect(p, w); n1 = n_relative(p, w)
    mults = [1.0, 2.0, 4.0, 8.0, 16.0, n1 / n0]
    frows = Tuple[]; med = Float64[]; lo = Float64[]; hi = Float64[]
    for k in mults
        n = ceil(Int, k * n0); res = replicate(rng, p, w, n, Rf)
        q = quantile(res.ratio, [0.025, 0.5, 0.975])
        push!(frows, (k, n, q[1], q[2], q[3])); push!(lo, q[1]); push!(med, q[2]); push!(hi, q[3])
    end
    write_csv(joinpath(out_dir, "e6_kelly_spread.csv"), ["n_over_n_detect", "n", "q025", "median", "q975"], frows)
    svg = svg_plot(joinpath(out_dir, "e6_sample_size.svg");
        series = [(x = mults, y = med, label = "Median estimated / true Kelly stake", kind = :both),
                  (x = [mults; reverse(mults); mults[1]], y = [lo; reverse(hi); lo[1]], label = "2.5–97.5% range",
                   kind = :line, color_index = 2, dash = true)],
        xlim = (0.8, 40.0), ylim = (0.0, 2.5), xticks = [1.0, 10.0], yticks = collect(0.0:0.5:2.5), xlog = true,
        xlabel = "Sample size n as a multiple of n_detect (w = 10, 5% edge)", ylabel = "Estimated Kelly stake / true Kelly stake",
        hlines = [(1.0, "")], vlines = [(n1 / n0, "n_relative")], legend_title = "E6: how much data?")
    quick || (mkpath(dirname(article)); cp(svg, article; force = true))
    write_meta(joinpath(out_dir, "e6_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("replications" => R, "pass" => pass))
    println("E6: ", pass ? "PASS" : "FAIL")
    return (; rows, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
