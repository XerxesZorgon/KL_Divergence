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

@testset "T-BOX optimal_allocation with caps (Task 070)" begin
    # T-BOX-1: single bet (p = 0.6, gross payoff 2.0) as the two-column single_event model [cash, bet]; upper = [Inf, 0.1]
    m1 = ScenarioModel([0.6, 0.4], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    r1 = optimal_allocation(m1; upper = [Inf, 0.1])
    @test r1.method === :clarabel
    @test r1.a[2] == 0.1
    @test abs(r1.a[1] - 0.9) ≤ 1e-15
    @test r1.kkt ≤ 1e-10
    _, ν1, _, _, _ = Kelly_KL.kkt_components(m1.X, r1.a, m1.π; upper = r1.upper)
    @test abs(ν1 - 98 / 99) ≤ 1e-12
    r_bet = (m1.X' * (m1.π ./ (m1.X * r1.a)))[2]
    @test abs(r_bet - 12 / 11) ≤ 1e-12

    # T-BOX-4: models from rand_correlated_model (the T-ORC-5 generator) with StableRNG(70), and single events from StableRNG(701)
    same(r, r0) = r.a == r0.a && r.growth === r0.growth && r.kkt === r0.kkt && r.method === r0.method
    function check_no_caps_and_loose(m, rng)
        J = size(m.X, 2)
        r0 = optimal_allocation(m)
        @test same(optimal_allocation(m; upper = nothing), r0)
        @test same(optimal_allocation(m; upper = fill(Inf, J)), r0)
        @test same(optimal_allocation(m; upper = 1 .+ rand(rng, J)), r0)
        bets = m.has_cash ? (2:J) : (1:J)
        u = fill(Inf, J)
        for j in bets
            v = r0.a[j] + 0.05
            u[j] = v ≥ 1 ? Inf : v
        end
        r = optimal_allocation(m; upper = u)
        _, ν, U, _, _ = Kelly_KL.kkt_components(m.X, r.a, m.π; upper = r.upper)
        @test maximum(abs.(r.a .- r0.a)) ≤ 1e-10
        @test isempty(U)
        @test abs(ν - 1) ≤ 1e-9
    end
    rng = StableRNG(70)
    for _ in 1:100
        check_no_caps_and_loose(rand_correlated_model(rng), rng)
    end
    rng = StableRNG(701); rng_u = StableRNG(7010)
    for _ in 1:100
        q, ohat = rand_single_event_inputs(rng)
        check_no_caps_and_loose(single_event(q, ohat), rng_u)
    end

    # T-BOX-16 (a), (c), (d), T-BOX-8 (e): no cash, J = 4
    rng = StableRNG(716)
    X16 = 1 .+ 3 .* rand(rng, 6, 4)
    π16 = rand_simplex(rng, 6)
    m16 = ScenarioModel(π16, X16; assets = ["b1", "b2", "b3", "b4"], has_cash = false, structure = :correlated)
    u0 = [0.3, 0.25, 0.25, 0.2]
    # upper edge is 1 + 0.999e-12: 1 + 1e-12 is not representable and u0 .* (1 + 1e-12) sums to 1 + 1.00009e-12, outside the band
    for target in (1 - 1e-12, 1 - 5e-13, 1.0, 1 + 5e-13, 1 + 0.999e-12)
        u = u0 .* target
        @test abs(sum(u) - 1) ≤ 1e-12
        r = optimal_allocation(m16; upper = u)
        @test r.method === :single_point
        @test r.a == u ./ sum(u)
        @test r.kkt ≤ 1e-11
    end
    for target in (1 + 2e-12, 1 + 1e-9)
        r = optimal_allocation(m16; upper = u0 .* target)
        @test r.method === :clarabel
        @test r.kkt ≤ 1e-10
    end
    @test_throws ArgumentError optimal_allocation(m16; upper = u0 .* (1 - 2e-12))
    # T-BOX-8 (e), solve part: Σu = 1 − 5e-13 uses the shortcut
    @test optimal_allocation(m16; upper = u0 .* (1 - 5e-13)).method === :single_point
end
