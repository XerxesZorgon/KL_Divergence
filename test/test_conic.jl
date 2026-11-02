using StableRNGs, Random
using Kelly_KL: greedy_allocation

@testset "conic oracles (T-ORC-1c, T-ORC-8, T-ERR-2)" begin
    # T-ORC-1 (conic path), seed 801, 1 000 cases
    rng = StableRNG(801)
    for _ in 1:1000
        p = rand(rng)
        w = 1 + 49 * rand(rng)
        m = ScenarioModel([p, 1 - p], [1.0 w; 1.0 0.0];
                          assets = ["cash", "bet"],
                          has_cash = true,
                          structure = :single_event)
        r = optimal_allocation(m; method = :conic)
        @test abs(r.a[2] - max((p * w - 1) / (w - 1), 0.0)) ≤ 1e-10
        assert_certified(r, m)
    end

    # T-ORC-8 (greedy = conic), seed 802, 1 000 accepted non-degenerate single events
    rng = StableRNG(802)
    accepted = 0
    for _ in 1:100_000
        accepted == 1000 && break
        n = rand(rng, 2:6)
        xq = randexp(rng, n); q = xq ./ sum(xq)
        xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
        ohat = 1.0 ./ (m_ .* (1 + 0.1 * rand(rng)))
        m_ev = single_event(q, ohat)
        X = m_ev.X
        g = greedy_allocation(q, ohat)
        r_kkt = X' * (q ./ (X * g))
        strict_ok = all(j -> g[j] > 1e-12 ? g[j] > 1e-3 : r_kkt[j] < 1 - 1e-3, eachindex(g))
        strict_ok || continue
        accepted += 1
        r = optimal_allocation(q, ohat; method = :conic)
        @test maximum(abs.(r.a .- g)) ≤ 1e-10
        assert_certified(r, m_ev)
    end
    @test accepted == 1000

    # T-ERR-2 (parent two-bet example)
    m = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    r = optimal_allocation(m)
    # parent article prints 0.0991 by truncation; correctly rounded it is 0.0992 (Task 020)
    @test floor.(r.a[2:3] .* 1e4) ./ 1e4 == [0.0645, 0.0991]
    @test round.(r.a[2:3]; digits = 4) == [0.0645, 0.0992]
    @test maximum(abs.(r.a .- [0.8363453271373047, 0.06450023479783634, 0.09915443806486002])) ≤ 1e-10
    assert_certified(r, m)
end

