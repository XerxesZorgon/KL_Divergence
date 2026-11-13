using Random
using Kelly_KL: greedy_allocation

function assert_certified(result::BetResult, m::ScenarioModel; tol = 1e-10)
    @test result.kkt ≤ tol
    @test kkt_residual(m.X, result.a, m.π; upper = result.upper) ≈ result.kkt atol = 1e-14
end

rand_simplex(rng, n) = (x = randexp(rng, n); x ./ sum(x))

function rand_single_event_inputs(rng, n_range = 2:12)
    n = rand(rng, n_range)
    q = rand_simplex(rng, n)
    xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
    ρ = 0.3 * rand(rng)
    ohat = payoffs_from_odds(1.0 ./ (m_ .* (1 + ρ)); commission = 0.2 * rand(rng))
    return q, ohat
end

function rand_correlated_model(rng, S_range = 4:64, n_range = 2:11)
    S = rand(rng, S_range)
    n = rand(rng, n_range)
    π = rand_simplex(rng, S)
    W = [rand(rng) < 0.4 ? 1 + 4 * rand(rng) : 0.0 for s in 1:S, k in 1:n]
    return correlated_bets(π, W)
end

function rand_nondegen_single_event(rng)
    n = rand(rng, 2:6)
    q = rand_simplex(rng, n)
    xm = q .* exp.(0.5 .* randn(rng, n)); m_ = xm ./ sum(xm)
    ohat = 1.0 ./ (m_ .* (1 + 0.1 * rand(rng)))
    m_ev = single_event(q, ohat)
    g = greedy_allocation(q, ohat)
    r_kkt = m_ev.X' * (q ./ (m_ev.X * g))
    ok = all(j -> g[j] > 1e-12 ? g[j] > 1e-3 : r_kkt[j] < 1 - 1e-3, eachindex(g))
    return ok, q, ohat, m_ev, g
end

function rand_unseen(rng, n)
    T0 = rand(rng, 0:3)
    return hcat(ones(T0), [rand(rng) < 0.3 ? 1 + 4 * rand(rng) : 0.0 for t in 1:T0, k in 1:n])
end


