using StableRNGs, Random

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
        xp = randexp(rng, n); p = xp ./ sum(xp)
        xq = randexp(rng, n); q = xq ./ sum(xq)
        @test kl(p, p) == 0.0
        @test kl(p, q) ≥ 0
    end

    # (e) validation
    @test_throws ArgumentError kl([0.5, 0.5], [1.0])
    @test_throws ArgumentError kl([0.5, -0.5], [0.5, 0.5])
end
