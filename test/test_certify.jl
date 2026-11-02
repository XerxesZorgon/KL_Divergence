@testset "certify" begin
    X = [1.0 2.0; 1.0 0.0]
    π = [0.6, 0.4]
    m = ScenarioModel(π, X; assets = ["cash", "bet"], has_cash = true, structure = :single_event)

    # (a) Kelly optimum b = 0.2
    @test kkt_residual(X, [0.8, 0.2], π) ≤ 1e-15

    # (b) Suboptimal interior allocation
    @test kkt_residual(X, [0.9, 0.1], π) ≥ 0.09

    # (c) Negativity term
    @test kkt_residual(X, [1.1, -0.1], π) ≥ 0.1

    # (d) Sum term
    @test kkt_residual(X, [0.7, 0.2], π) ≥ 0.1

    # (e) Zero wealth in positive-probability scenario
    @test kkt_residual(X, [0.0, 1.0], π) == Inf

    # (f) certify passes on optimal and throws CertificationError on suboptimal
    res_opt = BetResult(m, [0.8, 0.2], :test)
    @test certify(res_opt) === res_opt
    res_sub = BetResult(m, [0.9, 0.1], :test)
    @test_throws CertificationError certify(res_sub)

    # showerror coverage
    io = IOBuffer()
    showerror(io, CertificationError(0.1, 1e-10, :test, "uncertified bet"))
    @test occursin("CertificationError", String(take!(io)))
    showerror(io, SolverError("INFEASIBLE", "failed"))
    @test occursin("SolverError", String(take!(io)))

    # (g) assert_certified passes on optimal result
    assert_certified(res_opt, m)
end
