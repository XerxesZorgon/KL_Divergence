using StableRNGs, Random

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

