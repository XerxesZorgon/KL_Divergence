using StableRNGs, Random
using Kelly_KL: solve_conic, polish_log_optimal, polish_crra, greedy_allocation

@testset "polish" begin
    # (a) seed 701, 150 simultaneous problems
    rng = StableRNG(701)
    for _ in 1:150
        n = rand(rng, 2:6)
        p = 0.05 .+ 0.85 .* rand(rng, n)
        mk = clamp.(p .* exp.(0.3 .* randn(rng, n)), 0.02, 0.95)
        w = 1 ./ (mk .* (1 + 0.05 * rand(rng)))
        m = simultaneous_bets(p, w)
        (_, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test ρ ≤ 1e-10
    end

    # (b) seed 702, 150 correlated problems
    rng = StableRNG(702)
    for _ in 1:150
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        (_, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test ρ ≤ 1e-10
    end

    # (c) seed 703, 100 single events with cash
    rng = StableRNG(703)
    for _ in 1:100
        n = rand(rng, 2:12)
        xq = randexp(rng, n); q = xq ./ sum(xq)
        xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
        ρ_m = 0.3 * rand(rng)
        ohat = payoffs_from_odds(1.0 ./ (m_ .* (1 + ρ_m)); commission = 0.2 * rand(rng))
        m = single_event(q, ohat)
        (a, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test maximum(abs.(a .- greedy_allocation(q, ohat))) ≤ 1e-10
        @test ρ ≤ 1e-10
    end

    # (d) regression case (Task 021, seed 901, case 14)
    pb = [0.99, 0.99, 0.16488926145448488, 0.99, 0.6568450548562222]
    w = [1.6638977752618886, 7.705939380077611, 1.0403264815888114, 2.09348899942351, 2.5594619949577773]
    m = simultaneous_bets(pb, w)
    a0 = [9.033673338264071e-7, 0.00038533495305392433, 0.986357007716434, 1.5860291618815984e-9, 0.013163583188946002, 9.316918820305283e-5]
    (a, ρ) = polish_log_optimal(m.X, m.π, a0)
    @test ρ ≤ 1e-10
    @test a[1] > 0
end

@testset "polish: CRRA weights" begin
    # (a) Closed form, seed 1701, 500 cases
    rng = StableRNG(1701)
    for _ in 1:500
        p = 0.05 + 0.85 * rand(rng)
        w = (1.02 + 0.3 * rand(rng)) / p
        k = rand(rng, (0.0, 0.5, 2.0, 10.0))
        (a, ρ) = polish_crra([1.0 w; 1.0 0.0], [p, 1 - p], [0.5, 0.5], k)
        R = (p * (w - 1) / (1 - p))^(1 / (1 + k))
        @test abs(a[2] - (R - 1) / (w - 1 + R)) ≤ 1e-10
        @test ρ ≤ 1e-10
    end

    # (b) Correlated models, seed 1702
    rng = StableRNG(1702)
    N_b = QUICK ? 30 : 100
    for _ in 1:N_b
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m_b = correlated_bets(π, W)
        k = exp(log(0.1) + rand(rng) * log(100))
        (a, ρ) = polish_crra(m_b.X, m_b.π, solve_conic(m_b), k)
        @test ρ ≤ 1e-10
    end

    # (c) Wrapper unchanged in behavior
    pb = [0.99, 0.99, 0.16488926145448488, 0.99, 0.6568450548562222]
    w = [1.6638977752618886, 7.705939380077611, 1.0403264815888114, 2.09348899942351, 2.5594619949577773]
    m_c = simultaneous_bets(pb, w)
    a0 = [9.033673338264071e-7, 0.00038533495305392433, 0.986357007716434, 1.5860291618815984e-9, 0.013163583188946002, 9.316918820305283e-5]
    (a_c, ρ_c) = polish_log_optimal(m_c.X, m_c.π, a0)
    @test ρ_c ≤ 1e-10
    @test a_c[1] > 0
    @test_throws ArgumentError polish_crra(m_c.X, m_c.π, a0, -1.0)
end

