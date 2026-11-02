using StableRNGs, Random, LinearAlgebra
using Kelly_KL: greedy_allocation, cover_update

@testset "crosscheck (T-E1-3)" begin
    # (a) Agreement on non-degenerate events
    rng = StableRNG(401)
    accepted = 0
    for _ in 1:100_000
        accepted == 1000 && break
        n = rand(rng, 2:6)
        xq = randexp(rng, n); q = xq ./ sum(xq)
        xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
        ohat = 1.0 ./ (m_ .* (1 + 0.1 * rand(rng)))
        X = single_event(q, ohat).X
        g = greedy_allocation(q, ohat)
        r = X' * (q ./ (X * g))
        strict_ok = all(j -> g[j] > 1e-12 ? g[j] > 1e-3 : r[j] < 1 - 1e-3, eachindex(g))
        strict_ok || continue
        accepted += 1
        ac, kc, itc = cover_update(X, q; tol = 1e-14)
        @test kc < 1e-14
        @test itc < 2_000_000
        @test maximum(abs.(ac .- g)) ≤ 1e-8
    end
    @test accepted == 1000

    # (b) Recorded stall case (E1 stress, seed 7, trial 17 107)
    q = [0.00020838121287119872, 1.1209261096267569e-5, 0.0005783775956495249,
         3.371889426870182e-5, 0.9665317526828919, 4.267801849974459e-7,
         0.030729606897642796, 6.818056191985896e-7, 0.0019058448697754468]
    ohat = [5.643849259957775, 76.48229011241091, 19.753496285271467,
            19.936675832894974, 2.6187380177308954, 80.02336032728618,
            8.745265092155345, 3.148397861287974, 7.223847829808504]
    X = hcat(ones(9), Matrix(Diagonal(ohat)))
    a, k, it = cover_update(X, q)
    @test k > 1e-3
    @test it == 2_000_000
    @test kkt_residual(X, greedy_allocation(q, ohat), q) ≤ 1e-10
end

