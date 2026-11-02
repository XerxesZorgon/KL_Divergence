using StableRNGs, Random

@testset "constructors" begin
    # (a) simultaneous_bets exact values
    m = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    @test m.π ≈ [0.525, 0.225, 0.175, 0.075] atol = 1e-15
    @test m.X == [1.0 0.0 0.0; 1.0 4.0 0.0; 1.0 0.0 6.0; 1.0 4.0 6.0]
    @test m.assets == ["cash", "bet_1", "bet_2"]
    @test m.structure == :simultaneous

    # (b) seed 501 (StableRNG), 200 cases, n ∈ 1:8
    rng = StableRNG(501)
    for _ in 1:200
        n = rand(rng, 1:8)
        p = rand(rng, n)
        w = 1.0 .+ 9.0 .* rand(rng, n)
        mb = simultaneous_bets(p, w)
        @test length(mb.π) == 2^n
        @test abs(sum(mb.π) - 1) ≤ 1e-12
        for k in 1:n
            @test sum(mb.π[s] for s in 1:2^n if mb.X[s, k + 1] > 0) ≈ p[k] atol = 1e-12
        end
    end

    # (c) simultaneous_bets without cash throws DomainError on losing scenario
    @test_throws DomainError simultaneous_bets([0.3], [4.0]; cash = false)

    # (d) n > 16 throws ArgumentError
    @test_throws ArgumentError simultaneous_bets(fill(0.5, 17), fill(2.0, 17))

    # (e) correlated_bets
    mc = correlated_bets([0.5, 0.3, 0.2], [2.0 0.0; 0.0 3.0; 2.0 3.0])
    @test mc.X == [1.0 2.0 0.0; 1.0 0.0 3.0; 1.0 2.0 3.0]
    @test mc.structure == :correlated
    @test_throws ArgumentError correlated_bets([0.5, 0.6], [2.0; 0.0;;])
end
