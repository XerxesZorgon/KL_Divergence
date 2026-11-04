# T-REP-2: the reference comparator (experiments/compare.jl) catches real changes and tolerates only rounding-level ones.

module CompareUnderTest
include(joinpath(@__DIR__, "..", "experiments", "compare.jl"))
end

@testset "reference comparator (T-REP-2)" begin
    C = CompareUnderTest
    csv = "x,y,label\n1.000000000000e+00,2.500000000000e-01,bet_1\n3.141592653590e+00,-1.000000000000e-03,W-L1\n"
    svg = "<svg viewBox=\"0 0 760 470\"><polyline points=\"70.0,40.5 120.3,88.1\" stroke=\"#2a78d6\"/><text>E7: robust</text></svg>\n"
    mktempdir() do dir
        ref, out = joinpath(dir, "ref"), joinpath(dir, "out")
        for d in (ref, out)
            mkpath(d)
            write(joinpath(d, "a.csv"), csv); write(joinpath(d, "b.svg"), svg)
        end
        write(joinpath(ref, "e9_meta.toml"), "wall_seconds = 1.0\n"); write(joinpath(out, "e9_meta.toml"), "wall_seconds = 2.0\n")
        status(exact) = Dict(f => ok for (f, ok, _) in C.compare_dirs(ref, out; exact))

        # identical files pass in both modes; meta files are ignored
        @test all(values(status(true))) && all(values(status(false)))
        @test !haskey(status(true), "e9_meta.toml")

        # one byte of an SVG (non-numeric text) changed: fails in both modes
        write(joinpath(out, "b.svg"), replace(svg, "robust" => "robusT"))
        @test !status(true)["b.svg"] && !status(false)["b.svg"]
        write(joinpath(out, "b.svg"), svg)

        # one digit of a CSV changed (relative change 4e-12 > 1e-12): fails in both modes
        write(joinpath(out, "a.csv"), replace(csv, "2.500000000000e-01" => "2.500000000010e-01"))
        @test !status(true)["a.csv"] && !status(false)["a.csv"]

        # rounding-level change (relative 1e-13): fails exact, passes tolerant
        write(joinpath(out, "a.csv"), replace(csv, "3.141592653590e+00" => "3.1415926535903e+00"))
        @test !status(true)["a.csv"] && status(false)["a.csv"]

        # an SVG coordinate moved by 0.1: fails in both modes
        write(joinpath(out, "a.csv"), csv); write(joinpath(out, "b.svg"), replace(svg, "120.3" => "120.4"))
        @test !status(true)["b.svg"] && !status(false)["b.svg"]
        write(joinpath(out, "b.svg"), svg)

        # missing output and unreferenced output both fail
        rm(joinpath(out, "a.csv")); write(joinpath(out, "extra.csv"), "z\n")
        st = status(false)
        @test !st["a.csv"] && !st["extra.csv"]
    end
end
