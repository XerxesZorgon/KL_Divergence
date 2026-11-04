using Test, Kelly_KL, Aqua

const QUICK = get(ENV, "KELLY_KL_QUICK", "0") == "1"

@testset verbose=true "Kelly_KL" begin
    @testset "Aqua (T-AQ-1)" begin
        # StableRNGs is a package dependency only so that experiments/ and test/ (run with --project=.) can use it; Kelly_KL
        # itself never loads it, so it is exempt from the stale-dependency check (Task 056, ADR-009).
        Aqua.test_all(Kelly_KL; stale_deps = (ignore = [:StableRNGs],))
    end
    include("certify_hook.jl")
    include("test_model.jl")
    include("test_constructors.jl")
    include("test_certify.jl")
    include("test_exact.jl")
    include("test_conic_raw.jl")
    include("test_polish.jl")
    include("test_optimize.jl")
    include("test_conic.jl")
    include("test_general.jl")
    include("test_uncertainty.jl")
    include("test_errata.jl")
    include("test_diagnostics.jl")
    include("test_robust.jl")
    include("test_crosscheck.jl")
    include("test_cases.jl")
    include("test_compare.jl")
    include("test_plotting.jl")
    include("test_e1.jl")
    include("test_experiments_smoke.jl")
end
