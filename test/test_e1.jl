module E1Main
    include(joinpath(@__DIR__, "..", "experiments", "e1_kl_growth_loss.jl"))
end

module E1Plot
    include(joinpath(@__DIR__, "..", "experiments", "e1_plot.jl"))
end

function csv_matches_within_tol(csv_path::AbstractString, golden_path::AbstractString)
    new_lines = readlines(csv_path)
    gold_lines = readlines(golden_path)
    length(new_lines) == length(gold_lines) || return false
    isempty(new_lines) && return false
    new_lines[1] == gold_lines[1] || return false
    for i in 2:length(new_lines)
        nf = split(new_lines[i], ",")
        gf = split(gold_lines[i], ",")
        length(nf) == length(gf) || return false
        nf[1] == gf[1] || return false
        for j in 2:length(nf)
            x = parse(Float64, nf[j])
            y = parse(Float64, gf[j])
            abs(x - y) ≤ max(1e-12 * abs(y), 1e-15) || return false
        end
    end
    return true
end

@testset "E1 (T-E1-1, T-E1-2)" begin
    # T-E1-1
    viol, worst, kmax = E1Main.stress(; trials = QUICK ? 5_000 : 50_000)
    @test viol == 0
    # L/D not tested: ill-conditioned for tiny D (seed 7, trial 22 420: D = 1.45e-9 → ratio error 1.3e-7); L ≤ D + 1e-10 is the bound
    @test kmax ≤ 1e-10

    # T-E1-2
    golden_csv = joinpath(@__DIR__, "golden", "e1_results.csv")
    golden_svg = joinpath(@__DIR__, "golden", "e1_kl_growth_loss.svg")
    mktempdir() do tmp
        csv = joinpath(tmp, "e1_results.csv")
        svg = joinpath(tmp, "e1_kl_growth_loss.svg")
        E1Main.main(; out = csv)
        E1Plot.main(; csv = csv, out = svg, article = joinpath(tmp, "article.svg"))
        if haskey(ENV, "KELLY_KL_REFERENCE_HOST")
            @test read(csv) == read(golden_csv)
            @test read(svg) == read(golden_svg)
        else
            @test csv_matches_within_tol(csv, golden_csv)
            @test_skip read(svg) == read(golden_svg)
        end
    end
end

