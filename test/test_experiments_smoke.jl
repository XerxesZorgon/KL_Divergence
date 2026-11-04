# Smoke test for experiments E2–E7 (TEST_PLAN "Integration (experiments)"): each experiment's `main(quick = true)` runs
# without error into a temporary folder and writes all its outputs. E1 is covered by test_e1.jl (T-E1-1, T-E1-2).
# Quick mode uses few replications, so the experiments' statistical pass/fail is not asserted here (a 1 000-sample
# coverage check, for example, fails by chance more often than a test should); only that a Bool verdict is returned.
# The full-size verdicts are checked by the experiment runs themselves and, in M8, by reproduce_all.jl.

using TOML

const SMOKE_EXPERIMENTS = [
    ("e2_logscore.jl",       ["e2_logscore.csv", "e2_logscore.svg", "e2_meta.toml"]),
    ("e3_bma.jl",            ["e3_bma.csv", "e3_bma_summary.csv", "e3_bma.svg", "e3_meta.toml"]),
    ("e4_cliff_drawdown.jl", ["e4_cliff.csv", "e4_cliff.svg", "e4_drawdown.csv", "e4_drawdown.svg", "e4_meta.toml"]),
    ("e5_min_stake_ruin.jl", ["e5_classical.csv", "e5_priced_out.csv", "e5_ruin.svg", "e5_meta.toml"]),
    ("e6_sample_size.jl",    ["e6_coverage.csv", "e6_kelly_spread.csv", "e6_sample_size.svg", "e6_meta.toml"]),
    ("e7_robust.jl",         ["e7_table.csv", "e7_kl_coverage.csv", "e7_robust.svg", "e7_meta.toml"]),
]

@testset "experiments: smoke (quick mode)" begin
    for (file, outputs) in SMOKE_EXPERIMENTS
        @testset "$file" begin
            mod = Module(Symbol(splitext(file)[1]))
            Core.eval(mod, :(include(x) = Base.include($mod, x)))
            Base.include(mod, joinpath(@__DIR__, "..", "experiments", file))
            mktempdir() do dir
                r = redirect_stdout(devnull) do
                    Base.invokelatest(mod.main; quick = true, out_dir = dir)
                end
                @test r.pass isa Bool
                for f in outputs
                    @test isfile(joinpath(dir, f))
                end
                @test TOML.parsefile(joinpath(dir, outputs[end]))["seed"] isa Integer
            end
        end
    end
end
