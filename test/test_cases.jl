# T-PY-1 (Julia side): every shared case in test/cases/cases.toml reproduces its stored expected values within its tol.
# The same file is read by python/tests/test_parity.py, so a pass here plus a pass there means Python == Julia.
using TOML
include(joinpath(@__DIR__, "cases", "case_runner.jl"))

@testset "shared cases (T-PY-1, Julia side)" begin
    cases = TOML.parsefile(joinpath(@__DIR__, "cases", "cases.toml"))["case"]
    @test length(cases) ≥ 20
    @test length(unique(c["id"] for c in cases)) == length(cases)
    for c in cases
        @test !isempty(c["expected"])
        err = case_error(run_case(c), c["expected"])
        @test err ≤ c["tol"]
        err ≤ c["tol"] || @info "shared case $(c["id"]) differs by $err"
    end
end
