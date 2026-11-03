using StableRNGs, Random
using Kelly_KL: kl_cash_radius_direct, greedy_allocation, polish_crra, kl_robust_conic, wasserstein_inner_lp

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

@testset "robust: Wasserstein worst case (T-ORC-9 ii)" begin
    # (a) Breakpoint scan vs LP, seed 2001
    rng = StableRNG(2001)
    N_a = QUICK ? 60 : 300
    for _ in 1:N_a
        S = rand(rng, 3:29)
        n = rand(rng, 1:6)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        T0 = rand(rng, 0:3)
        unseen = hcat(ones(T0), [rand(rng) < 0.3 ? 1 + 4 * rand(rng) : 0.0 for t in 1:T0, k in 1:n])
        xa = randexp(rng, n + 1); a = xa ./ sum(xa)
        ε = exp(log(1e-3) + rand(rng) * (log(3.0) - log(1e-3)))
        (v, λ, Q) = wasserstein_worst_case(m, a, ε; unseen)
        @test abs(v - wasserstein_inner_lp(m, a, ε; unseen)) ≤ 1e-7
        @test abs(sum(Q) - 1) ≤ 1e-12
        @test abs(sum(Q .* log.(vcat(m.X, unseen) * a)) - v) ≤ 1e-12
    end

    # (b) Unseen catastrophe
    m_b = correlated_bets([0.5, 0.5], [3.0 0.0; 0.0 2.0])
    a_b = [0.2, 0.4, 0.4]
    unseen_b = [1.0 0.0 0.0]
    @test abs(wasserstein_worst_case(m_b, a_b, 0.0; unseen = unseen_b)[1] - (0.5 * log(1.4) + 0.5 * log(1.0))) ≤ 1e-12
    @test abs(wasserstein_worst_case(m_b, a_b, 10.0; unseen = unseen_b)[1] - log(0.2)) ≤ 1e-12

    # (c) cost = :linf and user S × T matrix
    @test abs(wasserstein_worst_case(m_b, a_b, 0.5; unseen = unseen_b, cost = :linf)[1] -
              wasserstein_inner_lp(m_b, a_b, 0.5; unseen = unseen_b, cost = :linf)) ≤ 1e-7
    C_user = [0.0 2.0 1.5; 2.0 0.0 1.0]
    @test abs(wasserstein_worst_case(m_b, a_b, 0.5; unseen = unseen_b, cost = C_user)[1] -
              wasserstein_inner_lp(m_b, a_b, 0.5; unseen = unseen_b, cost = C_user)) ≤ 1e-7

    # (d) Validation
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, 0.5; unseen = zeros(1, 2))
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, 0.5; unseen = [1.0 -0.1 0.0])
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, -1.0; unseen = unseen_b)
    @test_throws ArgumentError wasserstein_worst_case(m_b, [0.0, 0.5, 0.5], 0.5; unseen = unseen_b)
end

@testset "robust: Wasserstein-robust bet (T-ORC-9 i, T-ORC-10, T-ORC-11)" begin
    # (a) T-ORC-9 (i) and certificates, seed 2101
    rng = StableRNG(2101)
    N_a = QUICK ? 30 : 100
    n_kkt = 0
    n_gap = 0
    max_gap = 0.0
    for _ in 1:N_a
        S = rand(rng, 3:19)
        n = rand(rng, 1:5)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        T0 = rand(rng, 0:3)
        unseen = hcat(ones(T0), [rand(rng) < 0.3 ? 1 + 4 * rand(rng) : 0.0 for t in 1:T0, k in 1:n])
        ε = exp(log(1e-3) + rand(rng) * (log(2.0) - log(1e-3)))
        r = wasserstein_robust_allocation(m, ε; unseen)
        @test abs(r.worst_case_growth - wasserstein_worst_case(m, r.a, ε; unseen)[1]) ≤ 1e-10
        @test r.certificate == :kkt ? r.kkt ≤ 1e-10 : (r.certificate == :duality_gap && r.kkt ≤ 1e-7)
        if r.certificate == :kkt
            n_kkt += 1
        elseif r.certificate == :duality_gap
            n_gap += 1
            max_gap = max(max_gap, r.kkt)
        end
        J = size(m.X, 2)
        for _ in 1:5
            xb = randexp(rng, J); b = xb ./ sum(xb)
            @test wasserstein_worst_case(m, b, ε; unseen)[1] ≤ r.worst_case_growth + (r.certificate == :kkt ? 1e-10 : 1e-7)
        end
    end
    println("wasserstein_robust_allocation (a): :kkt = $n_kkt, :duality_gap = $n_gap, max_gap = $max_gap")

    # (b) T-ORC-10 (W): seed 2102, 20 models
    rng = StableRNG(2102)
    for _ in 1:20
        S = rand(rng, 3:19)
        n = rand(rng, 1:5)
        xπ = randexp(rng, S); π = xπ ./ sum(xπ)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        m = correlated_bets(π, W)
        @test maximum(abs.(wasserstein_robust_allocation(m, 0.0).a .- optimal_allocation(m).a)) ≤ 1e-9
    end

    # (c) T-ORC-11 (unseen catastrophe forces cash)
    m_c = correlated_bets([0.5, 0.5], [3.0 0.0; 0.0 2.0])
    unseen_c = [1.0 0.0 0.0]
    @test optimal_allocation(m_c).a[1] ≤ 1e-12
    for ε in (0.01, 1.0)
        r = wasserstein_robust_allocation(m_c, ε; unseen = unseen_c)
        @test r.a[1] > 0
        @test isfinite(r.worst_case_growth)
    end
    @test kl_cash_radius(m_c)[1] == Inf
    @test kl_robust_allocation(m_c, 0.01).a[1] ≤ 1e-9

    # (d) KL results carry :kkt
    @test kl_robust_allocation(correlated_bets([0.6, 0.4], [2.0; 0.0;;]), 0.01).certificate == :kkt
end



