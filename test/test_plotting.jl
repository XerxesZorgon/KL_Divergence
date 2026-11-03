using TOML

module Plotting
include(joinpath(@__DIR__, "..", "experiments", "plotting.jl"))
end

@testset "experiments: plotting helpers" begin
    mktempdir() do dir
        svg_path = joinpath(dir, "test.svg")
        series = [
            (x = [0.0, 1.0, 2.0], y = [0.1, 0.5, 0.9], label = "Line series example", kind = :line, color_index = 1),
            (x = [0.5, 1.5], y = [0.3, 0.7], label = "Points series example with a long label", kind = :points, color_index = 1),
        ]
        Plotting.svg_plot(
            svg_path;
            series,
            xlim = (0.0, 2.0),
            ylim = (0.0, 1.0),
            xticks = [0.0, 1.0, 2.0],
            yticks = [0.0, 0.5, 1.0],
            xlabel = "x",
            ylabel = "y",
            hlines = [(0.5, "half")],
            vlines = [(1.0, "mid")],
            legend_title = "Legend",
        )
        txt = read(svg_path, String)
        @test startswith(txt, "<svg")
        @test endswith(txt, "</svg>\n")
        @test occursin("<polyline", txt)
        @test occursin("<circle", txt)

        csv_path = joinpath(dir, "test.csv")
        Plotting.write_csv(csv_path, ["case", "val"], [("A", 1.5)])
        lines = readlines(csv_path)
        @test length(lines) == 2
        @test lines[1] == "case,val"
        @test lines[2] == "A,1.500000000000e+00"

        meta_path = joinpath(dir, "test_meta.toml")
        Plotting.write_meta(meta_path; seed = 42, wall_seconds = 1.2345)
        meta = TOML.parsefile(meta_path)
        @test haskey(meta, "julia_version")
        @test haskey(meta, "git_commit")
        @test meta["seed"] == 42
        @test meta["wall_seconds"] == 1.234
    end
end

