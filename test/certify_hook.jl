function assert_certified(result::BetResult, m::ScenarioModel; tol = 1e-10)
    @test result.kkt ≤ tol
    @test kkt_residual(m.X, result.a, m.π) ≈ result.kkt atol = 1e-14
end
