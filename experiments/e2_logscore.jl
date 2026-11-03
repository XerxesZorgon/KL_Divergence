# E2 — Does the holdout log-score difference estimate the KL difference?
#
# For truth p and two models π1, π2, the mean of log π1(y) − log π2(y) over held-out outcomes y ~ p has expectation
# D(p‖π2) − D(p‖π1): the unknown entropy of p cancels, so comparing log scores compares distances to the truth.
# Pass (TEST_PLAN §G, amended Task 043): the share of replications whose ±2 SE interval covers the true ΔKL lies within
# 3 binomial SE of the normal-theory 95.45%.

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

simplex(rng, n) = (x = randexp(rng, n); x ./ sum(x))
tilt(rng, p, σ) = (x = p .* exp.(σ .* randn(rng, length(p))); x ./ sum(x))
draw(rng, p, n) = (c = cumsum(p); [min(searchsortedfirst(c, rand(rng) * c[end]), length(p)) for _ in 1:n])

function main(; seed = 2002, out_dir = joinpath(@__DIR__, "output"), quick = false, n = 500,
               article = joinpath(@__DIR__, "..", "..", "img", "e2_logscore.svg"))
    t0 = time(); rng = StableRNG(seed); R = quick ? 100 : 1000
    rows = Tuple[]; covered = 0
    for r in 1:R
        K = rand(rng, 3:10); p = simplex(rng, K); π1 = tilt(rng, p, 0.3); π2 = tilt(rng, p, 0.6)
        y = draw(rng, p, n)
        est = logscore_difference(π1, π2, y)
        Δ = kl(p, π2) - kl(p, π1)
        c = abs(est.mean - Δ) ≤ 2 * est.se
        covered += c
        push!(rows, (r, K, n, Δ, est.mean, est.se, Int(c)))
    end
    coverage = covered / R; nominal = 0.9545; band = 3 * sqrt(nominal * (1 - nominal) / R)
    pass = abs(coverage - nominal) ≤ band
    write_csv(joinpath(out_dir, "e2_logscore.csv"), ["rep", "K", "n", "delta_kl", "estimate", "se", "covered"], rows)
    shown = rows[1:min(300, R)]
    xs = [r[4] for r in shown]; ys = [r[5] for r in shown]
    lim = ceil(maximum(abs, vcat(xs, ys)) * 10) / 10
    svg = svg_plot(joinpath(out_dir, "e2_logscore.svg");
        series = [(x = xs, y = ys, label = "Holdout estimate, n = $n", kind = :points),
                  (x = [-lim, lim], y = [-lim, lim], label = "Estimate = true ΔKL", kind = :line, color_index = 2)],
        xlim = (-lim, lim), ylim = (-lim, lim), xticks = range(-lim, lim; length = 5), yticks = range(-lim, lim; length = 5),
        xlabel = "True ΔKL = D(p‖π2) − D(p‖π1) (nats)", ylabel = "Mean log-score difference on held-out data",
        legend_title = "E2: log score vs KL")
    quick || (mkpath(dirname(article)); cp(svg, article; force = true))
    write_meta(joinpath(out_dir, "e2_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}("replications" => R, "coverage" => coverage, "band" => band, "pass" => pass))
    @printf("E2: coverage of ±2 SE = %.4f (nominal 0.9545 ± %.4f over %d replications) → %s\n", coverage, band, R, pass ? "PASS" : "FAIL")
    return (; coverage, band, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
