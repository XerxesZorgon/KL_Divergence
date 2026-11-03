using StableRNGs, Random, LinearAlgebra
using Kelly_KL: full_investment_allocation, greedy_allocation

@testset "exact" begin
    # T-ORC-1 (greedy), seed 101, 1 000 cases
    @testset "T-ORC-1 (greedy)" begin
        rng = StableRNG(101)
        for _ in 1:1000
            p = rand(rng)
            w = 1 + 49 * rand(rng)
            a = greedy_allocation([p], [w])
            expected = max((p * w - 1) / (w - 1), 0.0)
            @test abs(a[2] - expected) ≤ 1e-12
            @test abs(a[1] - (1 - expected)) ≤ 1e-12
            m = ScenarioModel([p, 1 - p], [1.0 w; 1.0 0.0];
                              assets = ["cash", "bet"],
                              has_cash = true,
                              structure = :single_event)
            assert_certified(BetResult(m, a, :greedy), m)
        end
    end

    # T-ORC-2, seed 102, 1 000 cases
    @testset "T-ORC-2" begin
        rng = StableRNG(102)
        for _ in 1:1000
            n = rand(rng, 2:12)
            p = rand_simplex(rng, n)
            q = rand_simplex(rng, n)
            o = 1 .+ 9 .* rand(rng, n)
            @test full_investment_allocation(q) == q
            X = Matrix(Diagonal(o))
            @test abs(growth(X, q, p) - (sum(p .* log.(o)) + sum(p .* log.(p)) - sum(p .* log.(p ./ q)))) ≤ 1e-12
            @test abs((growth(X, p, p) - growth(X, q, p)) - sum(p .* log.(p ./ q))) ≤ 1e-12
            m = single_event(q, o; cash = false)
            assert_certified(BetResult(m, q, :full_investment), m)
        end
    end

    # Greedy certification on multi-outcome events, seed 103, 1 000 cases
    @testset "Greedy certification on multi-outcome events" begin
        rng = StableRNG(103)
        for _ in 1:1000
            q, ohat = rand_single_event_inputs(rng)
            m = single_event(q, ohat)
            assert_certified(BetResult(m, greedy_allocation(q, ohat), :greedy), m)
        end
    end

    # Validation
    @testset "Validation" begin
        @test_throws ArgumentError greedy_allocation([0.6, 0.6], [2.0, 2.0])
        @test_throws ArgumentError greedy_allocation([0.5], [0.0])
    end
end

