# Reproduce every number and figure of Extended Kelly II (Success Criterion 1; T-REP-1).
#
#   julia --project=. experiments/reproduce_all.jl
#
# Each script is loaded into a fresh module and called through `invokelatest` (Julia 1.12 world-age rules).
#
# 1. Deletes experiments/output/ and runs E1–E8 at full size into it (each experiment also refreshes its figure in ../img/).
#    An experiment that throws is marked ERROR and the rest still run.
# 2. Compares every output file with experiments/reference/ (compare.jl): byte-identical when KELLY_KL_REFERENCE_HOST=1,
#    otherwise within relative 1e-12. `*_meta.toml` files (versions, commit, wall time) are not compared.
# 3. Prints a verdict table (each experiment's own pass criterion, TEST_PLAN §G) and a file table; exits 1 on any failure.
# References are created once, on the reference machine, by experiments/update_reference.jl after a full run.
# Takes about 15–20 minutes (E5 ≈ 10).

using Kelly_KL, Printf, TOML
include(joinpath(@__DIR__, "compare.jl"))

const EXPERIMENT_FILES = [("E2", "e2_logscore.jl"), ("E3", "e3_bma.jl"), ("E4", "e4_cliff_drawdown.jl"),
                          ("E5", "e5_min_stake_ruin.jl"), ("E6", "e6_sample_size.jl"), ("E7", "e7_robust.jl"),
                          ("E8", "e8_demere.jl")]

function load_script(file)
    mod = Module(Symbol(splitext(file)[1]))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    Base.include(mod, joinpath(@__DIR__, file))
    return mod
end

function run_e1(out_dir)
    e1 = load_script("e1_kl_growth_loss.jl")
    rows = Base.invokelatest(() -> e1.main(; out = joinpath(out_dir, "e1_results.csv")))
    viol, worst, kkt = Base.invokelatest(() -> e1.stress())
    open(joinpath(out_dir, "e1_stress.csv"), "w") do io
        println(io, "trials,violations,max_L_over_D,worst_kkt")
        @printf(io, "%d,%d,%.12e,%.3e\n", 50_000, viol, worst, kkt)
    end
    plot = load_script("e1_plot.jl")
    Base.invokelatest(() -> plot.main(; csv = joinpath(out_dir, "e1_results.csv"), out = joinpath(out_dir, "e1_kl_growth_loss.svg")))
    return viol == 0 && all(t -> t.loss ≤ t.kl + 1e-10 && t.kkt ≤ 1e-10, rows) && kkt ≤ 1e-10
end

function main(; out_dir = joinpath(@__DIR__, "output"), ref_dir = joinpath(@__DIR__, "reference"), quick = false)
    t0 = time()
    rm(out_dir; recursive = true, force = true); mkpath(out_dir)
    verdicts = Tuple{String,String,Float64}[]
    function attempt(name, f)
        t = time()
        v = try
            f() ? "PASS" : "FAIL"
        catch e
            "ERROR: " * first(sprint(showerror, e), 200)
        end
        push!(verdicts, (name, v, time() - t))
        @printf("%s: %s (%.0f s)\n", name, v, time() - t)
    end
    quick || attempt("E1", () -> run_e1(out_dir))
    for (name, file) in EXPERIMENT_FILES
        attempt(name, () -> (m = load_script(file); Base.invokelatest(() -> m.main(; out_dir, quick)).pass))
    end
    open(io -> TOML.print(io, Dict("quick" => quick, "julia_version" => string(VERSION),
                                   "wall_seconds" => round(time() - t0; digits = 1))),
         joinpath(out_dir, "run_meta.toml"), "w")

    exact = get(ENV, "KELLY_KL_REFERENCE_HOST", "0") == "1"
    files = isdir(ref_dir) ? compare_dirs(ref_dir, out_dir; exact) :
            [("experiments/reference/", false, "no references — on the reference machine run experiments/update_reference.jl")]

    println("\nExperiment verdicts")
    for (name, v, t) in verdicts; @printf("  %-4s %-6s %6.0f s\n", name, first(v, 6) == "ERROR:" ? "ERROR" : v, t); end
    @printf("\nReference comparison (%s)\n", exact ? "exact: reference machine" : "tolerant: relative 1e-12")
    for (f, ok, d) in files; @printf("  %-4s %-34s %s\n", ok ? "PASS" : "FAIL", f, d); end
    pass = all(v -> v[2] == "PASS", verdicts) && all(r -> r[2], files)
    @printf("\nreproduce_all: %s — %d experiments, %d files, %.1f min\n", pass ? "PASS" : "FAIL",
            length(verdicts), length(files), (time() - t0) / 60)
    return pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
