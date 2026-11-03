using StableRNGs, Random
using Kelly_KL: greedy_allocation

@testset "optimize" begin
    # (a) seed 201, 1 000 random multi-outcome events with cash
    @testset "Single event with cash (seed 201)" begin
        rng = StableRNG(201)
        for _ in 1:1000
            q, ohat = rand_single_event_inputs(rng)
            r = optimal_allocation(q, ohat)
            @test r.a == greedy_allocation(q, ohat)
            @test r.method == :greedy
            assert_certified(r, single_event(q, ohat))
        end
    end

    # (b) seed 202, 1 000 cases without cash
    @testset "Single event without cash (seed 202)" begin
        rng = StableRNG(202)
        for _ in 1:1000
            n = rand(rng, 2:12)
            q = rand_simplex(rng, n)
            o = 1 .+ 9 .* rand(rng, n)
            r = optimal_allocation(q, o; cash = false)
            @test r.a == q
            @test r.method == :full_investment
            assert_certified(r, single_event(q, o; cash = false))
        end
    end

    # (c) single bet as a general single-event model
    m = ScenarioModel([0.6, 0.4], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    res_c = optimal_allocation(m)
    @test res_c.a ≈ [0.8, 0.2] atol = 1e-15
    @test res_c.method == :greedy
    assert_certified(res_c, m)

    res_conic = optimal_allocation(m; method = :conic)
    @test res_conic.method == :clarabel
    @test maximum(abs.(res_conic.a .- [0.8, 0.2])) ≤ 1e-10
    assert_certified(res_conic, m)

    # (d) conic + polish path on general models and invalid single-event shape checks
    m_sim = ScenarioModel([0.6, 0.4], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :simultaneous)
    r_sim = optimal_allocation(m_sim)
    @test r_sim.method == :clarabel
    @test maximum(abs.(r_sim.a .- [0.8, 0.2])) ≤ 1e-10
    assert_certified(r_sim, m_sim)

    m_scs = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    r_scs = optimal_allocation(m_scs; solver = :scs)
    @test r_scs.method == :scs
    assert_certified(r_scs, m_scs)

    @test_throws ArgumentError optimal_allocation(m; method = :foo)
    @test_throws ArgumentError optimal_allocation(ScenarioModel([0.6, 0.4], [1.0 2.0; 1.0 1.5]; assets = ["cash", "bet"], has_cash = true, structure = :single_event))
    @test_throws ArgumentError optimal_allocation(ScenarioModel([0.6, 0.4], [1.0 2.0 3.0; 1.0 0.0 0.0]; assets = ["cash", "bet1", "bet2"], has_cash = true, structure = :single_event))
    @test_throws ArgumentError optimal_allocation(ScenarioModel([1.0, 0.0], reshape([2.0, 0.0], 2, 1); assets = ["bet1"], has_cash = false, structure = :single_event))

    # (e) deliberately wrong stored allocation raises through certify
    @test_throws CertificationError certify(BetResult(m, [0.9, 0.1], :greedy))
end
