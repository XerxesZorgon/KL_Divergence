using Test, Kelly_KL, Aqua

const QUICK = get(ENV, "KELLY_KL_QUICK", "0") == "1"

@testset verbose=true "Kelly_KL" begin
    @testset "Aqua (T-AQ-1)" begin
        # stale_deps re-enabled in M8 (T-AQ-1 full), once every dependency is used
        Aqua.test_all(Kelly_KL; stale_deps = false)
    end
    include("test_model.jl")
end

