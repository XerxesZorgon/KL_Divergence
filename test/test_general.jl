using StableRNGs, Random

function _check_orc3(mt, mb)
    rt = optimal_allocation(mt)
    rb = optimal_allocation(mb)
    L = growth(mt, rt.a) - growth(mt, rb.a)
    D = kl(mt.π, mb.π)
    @test -1e-10 ≤ L ≤ D + 1e-10
    assert_certified(rt, mt)
    assert_certified(rb, mb)
end

@testset "general models (T-ORC-3, T-ORC-5)" begin
    N3 = QUICK ? 100 : 1000
    N5 = QUICK ? 50 : 500

    # T-ORC-3 (simultaneous), seed 901
    rng = StableRNG(901)
    for _ in 1:N3
        n = rand(rng, 2:6)
        p = 0.05 .+ 0.85 .* rand(rng, n)
        mk = clamp.(p .* exp.(0.3 .* randn(rng, n)), 0.02, 0.95)
        w = 1 ./ (mk .* (1 + 0.05 * rand(rng)))
        pb = clamp.(p .* exp.((2 * rand(rng)) .* randn(rng, n)), 0.01, 0.99)
        _check_orc3(simultaneous_bets(p, w), simultaneous_bets(pb, w))
    end

    # T-ORC-3 (correlated), seed 902
    rng = StableRNG(902)
    for _ in 1:N3
        S = rand(rng, 4:64)
        n = rand(rng, 2:11)
        π = rand_simplex(rng, S)
        W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
        xπb = π .* exp.((2 * rand(rng)) .* randn(rng, S)); πb = xπb ./ sum(xπb)
        _check_orc3(correlated_bets(π, W), correlated_bets(πb, W))
    end

    # T-ORC-5, seed 903
    rng = StableRNG(903)
    for _ in 1:N5
        m = rand_correlated_model(rng, 4:256, 1:11)
        assert_certified(optimal_allocation(m), m)
    end
end

