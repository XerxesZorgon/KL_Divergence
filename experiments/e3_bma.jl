# E3 — Bayesian model averaging vs betting on the single best model.
#
# Three candidate models of a K-outcome event are scored on n training outcomes from the truth; the BMA bet uses the
# likelihood-weighted mixture, the alternative bets on the highest-likelihood model alone. Both are exact Kelly bets
# (cash allowed, 5% bookmaker margin); growth is evaluated under the truth.
# Pass (TEST_PLAN §G): for every training size, mean paired growth difference (BMA − best) > −2 SE.

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

simplex(rng, n) = (x = randexp(rng, n); x ./ sum(x))
tilt(rng, p, σ) = (x = p .* exp.(σ .* randn(rng, length(p))); x ./ sum(x))
draw(rng, p, n) = (c = cumsum(p); [min(searchsortedfirst(c, rand(rng) * c[end]), length(p)) for _ in 1:n])

function main(; seed = 2003, out_dir = joinpath(@__DIR__, "output"), quick = false, ns = (10, 50, 200),
               article = joinpath(@__DIR__, "..", "..", "img", "e3_bma.svg"))
    t0 = time(); rng = StableRNG(seed); R = quick ? 100 : 1000
    rows = Tuple[]; summary = Tuple[]
    for n in ns
        d = zeros(R)
        for r in 1:R
            K = rand(rng, 3:8); p = simplex(rng, K); mk = tilt(rng, p, 0.3); ohat = 1.0 ./ (mk .* 1.05)
            models = [tilt(rng, p, σ) for σ in (0.2, 0.4, 0.6)]
            y = draw(rng, p, n)
            ll = [sum(log.(πm[y])) for πm in models]
            w = bma_weights(ll)
            πbma = sum(w[i] .* models[i] for i in eachindex(models)); πbma ./= sum(πbma)
            truth = single_event(p, ohat)
            a_bma = optimal_allocation(πbma, ohat).a
            a_best = optimal_allocation(models[argmax(ll)], ohat).a
            d[r] = growth(truth, a_bma) - growth(truth, a_best)
            push!(rows, (n, r, K, d[r]))
        end
        μ = mean(d); se = std(d) / sqrt(R)
        push!(summary, (n, μ, se, mean(d .> 0), μ > -2se))
    end
    pass = all(s -> s[5], summary)
    write_csv(joinpath(out_dir, "e3_bma.csv"), ["n_train", "rep", "K", "growth_diff"], rows)
    write_csv(joinpath(out_dir, "e3_bma_summary.csv"), ["n_train", "mean_diff", "se", "frac_bma_ahead", "pass"], summary)
    xs = Float64[s[1] for s in summary]; ys = [1e3 * s[2] for s in summary]
    lo_s = [1e3 * (s[2] - 2s[3]) for s in summary]; hi_s = [1e3 * (s[2] + 2s[3]) for s in summary]
    raw = maximum(hi_s) * 1.2; step = raw > 10 ? 5.0 : 1.0; top = step * ceil(raw / step)
    svg = svg_plot(joinpath(out_dir, "e3_bma.svg");
        series = [(x = xs, y = ys, label = "Mean growth advantage of BMA over best model", kind = :both),
                  (x = xs, y = lo_s, label = "−2 SE", kind = :line, color_index = 2, dash = true),
                  (x = xs, y = hi_s, label = "+2 SE", kind = :line, color_index = 2, dash = true)],
        xlim = (5.0, 500.0), ylim = (0.0, top), xticks = [10.0, 100.0], yticks = collect(0.0:step:top), xlog = true,
        xlabel = "Training outcomes n (log scale)", ylabel = "Growth advantage per bet (×10⁻³ nats)",
        hlines = [(0.0, "")], legend_title = "E3: model averaging")
    quick || (mkpath(dirname(article)); cp(svg, article; force = true))
    write_meta(joinpath(out_dir, "e3_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("replications" => R, "pass" => pass))
    for s in summary
        @printf("E3: n = %d: mean advantage %.6f nats (SE %.6f), BMA ahead in %.1f%% of replications\n", s[1], s[2], s[3], 100s[4])
    end
    println("E3: ", pass ? "PASS" : "FAIL")
    return (; summary, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
