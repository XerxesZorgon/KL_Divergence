using Test, Kelly_KL, Aqua

const QUICK = get(ENV, "KELLY_KL_QUICK", "0") == "1"

@testset verbose=true "Kelly_KL" begin
    @testset "Aqua (T-AQ-1)" begin
        # stale_deps re-enabled in M8 (T-AQ-1 full), once every dependency is used
        Aqua.test_all(Kelly_KL; stale_deps = false)
    end
    include("certify_hook.jl")
    include("test_model.jl")
    include("test_constructors.jl")
    include("test_certify.jl")
    include("test_exact.jl")
    include("test_optimize.jl")
    include("test_diagnostics.jl")
    include("test_crosscheck.jl")
    include("test_e1.jl")
end
