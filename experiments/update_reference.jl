# Replace experiments/reference/ with the current experiments/output/ (ADR-009, step 4).
#
#   julia --project=. experiments/update_reference.jl
#
# Only on the reference machine (KELLY_KL_REFERENCE_HOST=1) and only after a full (non-quick) reproduce_all.jl run whose
# experiments all passed. Changing references is a reviewed change: commit them in a task of their own.

using TOML

function main(; out_dir = joinpath(@__DIR__, "output"), ref_dir = joinpath(@__DIR__, "reference"))
    get(ENV, "KELLY_KL_REFERENCE_HOST", "0") == "1" || error("update_reference: set KELLY_KL_REFERENCE_HOST=1 (reference machine only)")
    run_meta = joinpath(out_dir, "run_meta.toml")
    isfile(run_meta) || error("update_reference: $run_meta not found — run experiments/reproduce_all.jl first")
    TOML.parsefile(run_meta)["quick"] && error("update_reference: the last run was quick; references need a full run")
    rm(ref_dir; recursive = true, force = true); mkpath(ref_dir)
    n = 0
    for (root, _, files) in walkdir(out_dir), f in files
        endswith(f, "_meta.toml") && continue
        src = joinpath(root, f); dst = joinpath(ref_dir, relpath(src, out_dir))
        mkpath(dirname(dst)); cp(src, dst); n += 1
        println("reference: ", relpath(dst, ref_dir))
    end
    println("update_reference: copied $n files to $ref_dir")
    return n
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
