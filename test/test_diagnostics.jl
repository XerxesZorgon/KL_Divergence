using StableRNGs, Random, Statistics, LinearAlgebra

@testset "diagnostics: kl" begin
    # (a)
    @test kl([0.5, 0.5], [0.25, 0.75]) ≈ 0.14384103622589045 atol = 1e-15

    # (b)
    @test kl([1.0, 0.0], [0.5, 0.5]) ≈ log(2) atol = 1e-15

    # (c)
    @test kl([0.5, 0.5], [1.0, 0.0]) == Inf

    # (d) seed 301, 1 000 random pairs
    rng = StableRNG(301)
    for _ in 1:1000
        n = rand(rng, 2:12)
        p = rand_simplex(rng, n)
        q = rand_simplex(rng, n)
        @test kl(p, p) == 0.0
        @test kl(p, q) ≥ 0
    end

    # (e) validation
    @test_throws ArgumentError kl([0.5, 0.5], [1.0])
    @test_throws ArgumentError kl([0.5, -0.5], [0.5, 0.5])
end

@testset "diagnostics: growth loss and log scores" begin
    # Identity: seed 1401, 200 cases
    rng = StableRNG(1401)
    for _ in 1:200
        n = rand(rng, 2:8)
        p = rand_simplex(rng, n)
        q = rand_simplex(rng, n)
        o = 1 .+ 9 .* rand(rng, n)
        @test abs(growth_loss(single_event(p, o; cash = false), q) - kl(p, q)) ≤ 1e-12
    end

    # Bound: seed 1402, 200 correlated cases
    rng = StableRNG(1402)
    for _ in 1:200
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        π = rand_simplex(rng, S)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        xπb = π .* exp.((2 * rand(rng)) .* randn(rng, S)); πb = xπb ./ sum(xπb)
        mt = correlated_bets(π, W)
        mb = correlated_bets(πb, W)
        rb = optimal_allocation(mb)
        L = growth_loss(mt, rb.a)
        @test -1e-10 ≤ L ≤ kl(mt.π, mb.π) + 1e-10
    end

    # T-DIA-5 (exact expectation): seed 1403, 200 cases
    rng = StableRNG(1403)
    for _ in 1:200
        K = rand(rng, 2:10)
        p = rand_simplex(rng, K)
        π1 = rand_simplex(rng, K)
        π2 = rand_simplex(rng, K)
        E = sum(p[i] * logscore_difference(π1, π2, [i]).mean for i in 1:K)
        @test abs(E - (kl(p, π2) - kl(p, π1))) ≤ 1e-12
    end

    # SE
    hs = holdout_logscore([0.5, 0.25, 0.25], [1, 2, 3, 1])
    v_ref = log.([0.5, 0.25, 0.25, 0.5])
    @test isapprox(hs.mean, mean(v_ref); atol = 1e-15)
    @test isapprox(hs.se, std(v_ref) / 2; atol = 1e-15)
    @test isnan(holdout_logscore([0.5, 0.5], [1]).se)

    # Validation
    @test_throws ArgumentError holdout_logscore([0.5, 0.5], [3])
    @test_throws ArgumentError holdout_logscore([0.5, 0.5], Int[])
    @test_throws ArgumentError logscore_difference([0.5, 0.5], [1.0], [1])
end

@testset "diagnostics: growth curve (T-DIA-3)" begin
    # T-DIA-3
    p = 0.505
    w = 2.0
    m = ScenarioModel([p, 1 - p], [1.0 w; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    r = optimal_allocation(m)
    z = zero_growth_point(m, r.a)
    @test abs(z.c0 - 2) < 0.05
    @test abs(growth_curve(m, r.a, [z.c0])[1]) ≤ 1e-12
    @test abs(z.c0 - 1.9998666684444875) ≤ 1e-9
    @test z.c_quad == 2.0

    # Curve shape
    G = growth_curve(m, r.a, [0.0, 0.5, 1.0, 1.5, 2.5])
    @test G[1] == 0.0
    @test abs(G[3] - r.growth) ≤ 1e-15
    @test G[2] < G[3]
    @test G[4] < G[3]
    @test G[5] < 0
    @test growth_curve(m, r.a, [1.0001 / r.a[2]])[1] == -Inf

    # Simultaneous model
    ms = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    rs = optimal_allocation(ms)
    zs = zero_growth_point(ms, rs.a)
    @test 1 < zs.c0
    @test abs(growth_curve(ms, rs.a, [zs.c0])[1]) ≤ 1e-12

    # Validation
    @test_throws ArgumentError growth_curve(single_event([0.5, 0.5], [2.1, 2.1]; cash = false), [0.5, 0.5], [1.0])
    @test_throws ArgumentError zero_growth_point(m, [1.0, 0.0])
end

@testset "diagnostics: ruin and sample sizes (T-DIA-1, 2, 4)" begin
    # T-DIA-1
    for p in (0.4, 0.5, 0.55), N in 2:200
        n = N - 1
        T = Tridiagonal(fill(-(1 - p), n - 1), ones(n), fill(-p, n - 1))
        rhs = zeros(n)
        rhs[1] = 1 - p
        P = vcat(1.0, T \ rhs, 0.0)
        @test maximum(abs(P[a + 1] - ruin_probability_classical(p, a, N)) for a in 0:N) ≤ 1e-12
    end
    @test ruin_probability_classical(0.5, 3, 10) == 1 - 3 / 10

    # T-DIA-2
    for x in (0.1, 0.25, 0.5, 0.9)
        @test drawdown_probability(x, 1.0) == x
        @test drawdown_probability(x, 2.0) == 1.0
        @test abs(drawdown_probability(x, 0.5) - x^3) ≤ 1e-15
    end
    @test drawdown_probability(0.5, 3.0) == 1.0

    # T-DIA-4
    @test isapprox(n_detect(0.5, 2.2), 464.8336; rtol = 1e-9)
    @test isapprox(n_relative(0.5, 2.2), 12100.0; rtol = 1e-9)

    # Effective multiple
    @test effective_multiple(0.02, 0.01) == 2.0

    # Validation
    @test_throws ArgumentError drawdown_probability(1.0, 1.0)
    @test_throws ArgumentError ruin_probability_classical(1.0, 1, 2)
    @test_throws ArgumentError ruin_probability_classical(0.5, 3, 2)
    @test_throws ArgumentError n_detect(0.5, 2.0)
    @test_throws ArgumentError effective_multiple(0.01, 0.0)
end

@testset "diagnostics: ruin simulator (T-DIA-6)" begin
    # (a) No ruin for Kelly with cash
    ms = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    r = optimal_allocation(ms)
    sim = simulate_ruin(ms, r.a; rng = StableRNG(1501), trials = QUICK ? 200 : 1000, rounds = 200)
    @test sim.p_ruin == 0.0
    @test all(>(0.0), sim.final_q)

    # (b) Classical game
    m1 = ScenarioModel([0.48, 0.52], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    sim_b = simulate_ruin(
        m1,
        [1.0, 0.0];
        rng = StableRNG(1502),
        trials = QUICK ? 10_000 : 100_000,
        rounds = 100_000,
        w0 = 10.0,
        min_stake = 1.0,
        fixed_stake = 1.0,
        target = 20.0,
    )
    P = ruin_probability_classical(0.48, 10, 20)
    @test sim_b.se > 0
    @test abs(sim_b.p_ruin - P) ≤ 3 * sim_b.se

    # (c) Overbetting raises drawdowns
    mk = ScenarioModel([0.505, 0.495], [1.0 2.0; 1.0 0.0]; assets = ["cash", "bet"], has_cash = true, structure = :single_event)
    b = 0.01
    s1 = simulate_ruin(mk, [1 - b, b]; rng = StableRNG(1503), trials = 300, rounds = 20_000)
    s25 = simulate_ruin(mk, [1 - 2.5b, 2.5b]; rng = StableRNG(1503), trials = 300, rounds = 20_000)
    @test s25.drawdown_q[1] > s1.drawdown_q[1]
    @test s25.final_q[2] < s1.final_q[2]

    # (d) Lot rounding
    sim_d = simulate_ruin(mk, [1 - b, b]; rng = StableRNG(1504), trials = 50, rounds = 100, lot = 0.1)
    @test sim_d.p_ruin == 0.0
    @test all(==(1.0), sim_d.final_q)

    # (e) Validation
    @test_throws ArgumentError simulate_ruin(mk, [1 - b, b]; rng = StableRNG(1505), trials = 0, rounds = 10)
    @test_throws ArgumentError simulate_ruin(mk, [1 - b, b]; rng = StableRNG(1505), trials = 10, rounds = 10, target = 0.5)
    @test_throws ArgumentError simulate_ruin(mk, [1.0]; rng = StableRNG(1505), trials = 10, rounds = 10)
end

