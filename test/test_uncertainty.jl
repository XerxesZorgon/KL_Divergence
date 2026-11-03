using StableRNGs, Random

@testset "uncertainty: predictives, mixture, BMA" begin
    # T-ORC-12 (posterior collapse), seed 1001, 100 cases
    rng = StableRNG(1001)
    for _ in 1:100
        n = 3
        m = 0.1 .+ 0.5 .* rand(rng, n)
        e = 0.02 .+ 0.28 .* rand(rng, n)
        w = (1 .+ e) ./ m
        p2 = beta_predictive(2.0 .* m, 2.0 .* (1 .- m))
        m2 = simultaneous_bets(p2, w)
        r_2 = optimal_allocation(m2)
        assert_certified(r_2, m2)

        p10 = beta_predictive(10.0 .* m, 10.0 .* (1 .- m))
        m10 = simultaneous_bets(p10, w)
        r_10 = optimal_allocation(m10)
        assert_certified(r_10, m10)

        p1000 = beta_predictive(1000.0 .* m, 1000.0 .* (1 .- m))
        m1000 = simultaneous_bets(p1000, w)
        r_1000 = optimal_allocation(m1000)
        assert_certified(r_1000, m1000)

        @test maximum(abs.(r_2.a .- r_1000.a)) ≤ 1e-12
        @test maximum(abs.(r_10.a .- r_1000.a)) ≤ 1e-12
    end

    # T-ORC-13 (mixture linearity), seed 1002, 100 cases
    rng = StableRNG(1002)
    for _ in 1:100
        S = rand(rng, 4:32)
        n = rand(rng, 2:6)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        models = map(1:3) do _
            xπ = randexp(rng, S)
            correlated_bets(xπ ./ sum(xπ), W)
        end
        xw = randexp(rng, 3)
        weights = xw ./ sum(xw)
        mm = mixture(models, weights)
        for _ in 1:10
            xa = randexp(rng, n + 1)
            a = xa ./ sum(xa)
            @test abs(growth(mm, a) - sum(weights[i] * growth(models[i], a) for i in 1:3)) ≤ 1e-12
        end
        assert_certified(optimal_allocation(mm), mm)
    end

    # BMA and Dirichlet predictive
    @test isapprox(dirichlet_predictive([1.0, 2.0, 3.0]), [1, 2, 3] ./ 6; atol = 1e-15)
    @test isapprox(bma_weights(log.([1.0, 2.0, 3.0])), [1, 2, 3] ./ 6; atol = 1e-15)
    @test isapprox(bma_weights([1000.0, 1001.0]), [1, ℯ] ./ (1 + ℯ); atol = 1e-15)
    @test_throws ArgumentError bma_weights([0.0, -Inf])

    # Validation
    @test_throws ArgumentError beta_predictive([1.0], [0.0])
    @test_throws ArgumentError dirichlet_predictive([0.0, 1.0])
    m_a = single_event([0.5, 0.5], [2.0, 2.0])
    m_b = single_event([0.5, 0.5], [3.0, 2.0])
    @test_throws ArgumentError mixture([m_a, m_b], [0.5, 0.5])
    @test_throws ArgumentError mixture([m_a, m_a], [0.5, 0.6])
end

@testset "uncertainty: predictive from draws" begin
    # (a) Independence limit (exact), seed 1101, 100 cases
    rng = StableRNG(1101)
    for _ in 1:100
        n = rand(rng, 2:5)
        m = 0.2 .+ 0.5 .* rand(rng, n)
        h = 0.1 .* rand(rng, n)
        w = (1.1 .+ 0.3 .* rand(rng, n)) ./ m
        M = 1 << n
        P = [(((row - 1) >> (k - 1)) & 1 == 1 ? m[k] + h[k] : m[k] - h[k]) for row in 1:M, k in 1:n]
        @test maximum(abs.(predictive_from_draws(P, w).π .- simultaneous_bets(m, w).π)) ≤ 1e-14
    end

    # (b) Dependence changes the bet
    P_corr = [0.3 0.3; 0.7 0.7]
    P_ind = [0.3 0.3; 0.3 0.7; 0.7 0.3; 0.7 0.7]
    w = [2.2, 2.2]
    mc = predictive_from_draws(P_corr, w)
    mi = predictive_from_draws(P_ind, w)
    @test isapprox(mc.π, [0.29, 0.21, 0.21, 0.29]; atol = 1e-15)
    @test isapprox(mi.π, [0.25, 0.25, 0.25, 0.25]; atol = 1e-15)
    rc = optimal_allocation(mc)
    ri = optimal_allocation(mi)
    assert_certified(rc, mc)
    assert_certified(ri, mi)
    @test abs(rc.a[2] - 0.07155) ≤ 1e-4
    @test abs(ri.a[2] - 0.08266) ≤ 1e-4

    # (c) Validation
    @test_throws ArgumentError predictive_from_draws([0.5 1.2], [2.0, 2.0])
    @test_throws ArgumentError predictive_from_draws([0.5 0.5], [2.0])
end

