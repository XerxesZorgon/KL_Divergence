using StableRNGs, Random
using Kelly_KL: kl_cash_radius_direct, greedy_allocation, polish_crra, kl_robust_conic

@testset "robust: KL worst case (T-ORC-4, T-ORC-7)" begin
    # T-ORC-4, seed 1601
    rng = StableRNG(1601)
    N4 = QUICK ? 50 : 200
    for _ in 1:N4
        S = rand(rng, 4:39)
        n = rand(rng, 2:8)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        xa = randexp(rng, n + 1); a = xa ./ sum(xa)
        ε = exp(log(1e-4) + rand(rng) * (0 - log(1e-4)))
        (v, λ, Q) = kl_worst_case(m, a, ε)
        (vd, Qd) = kl_inner_direct(m, a, ε)
        @test abs(v - vd) ≤ 1e-7
        f = log.(m.X * a)
        @test abs(sum(Q .* f) - v) ≤ 1e-12
        if 0 < λ < Inf
            @test abs(kl(Q, m.π) - ε) ≤ 1e-10
        else
            @test kl(Q, m.π) ≤ ε + 1e-12
        end
    end

    # Point mass
    m2 = correlated_bets([0.9, 0.1], [2.0; 0.0;;])
    a2 = [0.5, 0.5]
    (v2, λ2, Q2) = kl_worst_case(m2, a2, -log(0.1) + 0.01)
    @test λ2 == 0.0
    @test Q2 == [0.0, 1.0]
    @test v2 == log(0.5)

    # ε = 0
    @test isapprox(kl_worst_case(m2, a2, 0.0)[1], 0.9 * log(1.5) + 0.1 * log(0.5); atol = 1e-15)

    # T-ORC-7, seed 1602, 20 cases
    rng = StableRNG(1602)
    for _ in 1:20
        S = rand(rng, 4:39)
        n = rand(rng, 2:8)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        xa = randexp(rng, n + 1); a = xa ./ sum(xa)
        f = log.(m.X * a)
        mean_f = sum(m.π .* f)
        var_f = sum(m.π .* (f .- mean_f) .^ 2)
        e = [abs(kl_worst_case(m, a, ε)[1] - (mean_f - sqrt(2ε * var_f))) / ε for ε in (1e-4, 1e-5, 1e-6)]
        @test e[2] / e[1] ≤ 2
        @test e[3] / e[2] ≤ 2
    end

    # Validation
    @test_throws ArgumentError kl_worst_case(m2, a2, -0.1)
    @test_throws ArgumentError kl_worst_case(m2, [0.0, 1.0], 0.1)
end

@testset "robust: KL cash radius" begin
    # (a) Analytic
    m_a = correlated_bets([0.6, 0.4], [2.0; 0.0;;])
    (r_a, Q_a) = kl_cash_radius(m_a)
    @test abs(r_a - 0.020410997260127586) ≤ 1e-12
    @test abs(Q_a[1] - 0.5) ≤ 1e-12

    # (b) Kelly-cash ⇔ zero radius, seed 1801, 200 single events with cash
    rng = StableRNG(1801)
    for _ in 1:200
        n = rand(rng, 2:12)
        xq = randexp(rng, n); q = xq ./ sum(xq)
        xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
        ρ = 0.3 * rand(rng)
        ohat = payoffs_from_odds(1.0 ./ (m_ .* (1 + ρ)); commission = 0.2 * rand(rng))
        @test (kl_cash_radius(single_event(q, ohat))[1] ≤ 1e-14) == (greedy_allocation(q, ohat)[1] == 1.0)
    end

    # (c) Certificate and oracle, seed 1802
    rng = StableRNG(1802)
    N_c = QUICK ? 50 : 200
    n_inf = 0
    for _ in 1:N_c
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        (r, Q) = kl_cash_radius(m)
        (rd, st) = kl_cash_radius_direct(m)
        if isfinite(r)
            @test maximum(m.X[:, 2:end]' * Q) - 1 ≤ 1e-10
            @test r ≥ 0
            @test abs(r - rd) ≤ 1e-7
        else
            n_inf += 1
            @test st == Kelly_KL.MOI.INFEASIBLE
        end
    end
    println("kl_cash_radius (c): $n_inf / $N_c cases with radius == Inf")

    # (d) Dominant bet
    @test kl_cash_radius(correlated_bets([0.5, 0.5], [1.5; 1.2;;]))[1] == Inf

    # (e) Validation
    @test_throws ArgumentError kl_cash_radius(single_event([0.5, 0.5], [2.0, 2.0]; cash = false))
end

@testset "robust: KL-robust bet (T-ORC-6, T-ORC-10)" begin
    # (a) Analytic single bet
    m_a = correlated_bets([0.6, 0.4], [2.0; 0.0;;])
    for (ε, f) in (
        (0.001, 0.1560563420293526),
        (0.005, 0.10143706865613433),
        (0.01, 0.06033774487079824),
        (0.015, 0.02872571386602285),
    )
        r = kl_robust_allocation(m_a, ε)
        @test abs(r.a[2] - f) ≤ 1e-10
        @test r.kkt ≤ 1e-10
    end
    r_cash = kl_robust_allocation(m_a, 0.03)
    @test r_cash.a == [1.0, 0.0]
    @test r_cash.worst_case_growth == 0.0

    # (b) T-ORC-6 and max-min property, seed 1901
    rng = StableRNG(1901)
    N_b = QUICK ? 20 : 100
    n_cash = 0
    for _ in 1:N_b
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        ε = exp(log(1e-3) + rand(rng) * (log(0.5) - log(1e-3)))
        r = kl_robust_allocation(m, ε)
        r.a[1] == 1.0 && (n_cash += 1)
        @test r.kkt ≤ 1e-10
        @test abs(r.worst_case_growth - kl_worst_case(m, r.a, ε)[1]) ≤ 1e-10
        if 0 < r.λ < Inf
            @test abs(kl(r.Q, m.π) - ε) ≤ 1e-10
            J = size(m.X, 2)
            ac, ρc = polish_crra(m.X, m.π, fill(1 / J, J), 1 / r.λ)
            @test maximum(abs.(ac .- r.a)) ≤ 1e-8
        end
        J = size(m.X, 2)
        for _ in 1:5
            xb = randexp(rng, J); b = xb ./ sum(xb)
            @test kl_worst_case(m, b, ε)[1] ≤ r.worst_case_growth + 1e-10
        end
    end
    println("kl_robust_allocation (b): $n_cash / $N_b models returned all cash")

    # (c) T-ORC-10 (KL): seed 1902, 20 correlated models
    rng = StableRNG(1902)
    for _ in 1:20
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        @test maximum(abs.(kl_robust_allocation(m, 0.0).a .- optimal_allocation(m).a)) ≤ 1e-9
    end

    # (d) Validation
    @test_throws ArgumentError kl_robust_allocation(m_a, -0.1)
end

