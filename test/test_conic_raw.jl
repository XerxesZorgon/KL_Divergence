using StableRNGs, Random
using Kelly_KL: solve_conic, greedy_allocation

@testset "conic (raw)" begin
    # (a) seed 601 (StableRNG), 200 random single events with cash
    rng = StableRNG(601)
    for _ in 1:200
        q, ohat = rand_single_event_inputs(rng)
        # allocation distance is ill-conditioned near marginal bets (seed 601, trial 25: 1.0e-4 apart, growth gap 4.6e-10)
        m = single_event(q, ohat); a_c = solve_conic(m); a_g = greedy_allocation(q, ohat)
        @test growth(m, a_g) - growth(m, a_c) ≤ 1e-8
        @test maximum(abs.(a_c .- a_g)) ≤ 1e-3
    end

    # (b) simultaneous_bets with Clarabel
    m_sim = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    a = solve_conic(m_sim)
    @test abs(a[2] - 0.06450023) ≤ 1e-4
    @test abs(a[3] - 0.09915443) ≤ 1e-4

    # (c) simultaneous_bets with SCS
    a_scs = solve_conic(m_sim; solver = :scs)
    @test abs(a_scs[2] - 0.06450023) ≤ 1e-3
    @test abs(a_scs[3] - 0.09915443) ≤ 1e-3

    # (d) zero-probability rows are excluded
    m_zero = ScenarioModel([0.5, 0.5, 0.0], [2.0 0.0; 0.0 3.0; 0.0 0.0];
                           assets = ["b1", "b2"], has_cash = false, structure = :correlated)
    @test maximum(abs.(solve_conic(m_zero) .- [0.5, 0.5])) ≤ 1e-4

    # (e) invalid solver throws ArgumentError
    @test_throws ArgumentError solve_conic(m_zero; solver = :foo)
end

