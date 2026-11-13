using StableRNGs, Random
using Kelly_KL: solve_conic, polish_log_optimal, polish_crra, greedy_allocation

const _POLISH_REG_M = simultaneous_bets(
    [0.99, 0.99, 0.16488926145448488, 0.99, 0.6568450548562222],
    [1.6638977752618886, 7.705939380077611, 1.0403264815888114, 2.09348899942351, 2.5594619949577773],
)
const _POLISH_REG_A0 = [9.033673338264071e-7, 0.00038533495305392433, 0.986357007716434, 1.5860291618815984e-9, 0.013163583188946002, 9.316918820305283e-5]

@testset "polish" begin
    # (a) seed 701, 150 simultaneous problems
    rng = StableRNG(701)
    for _ in 1:150
        n = rand(rng, 2:6)
        p = 0.05 .+ 0.85 .* rand(rng, n)
        mk = clamp.(p .* exp.(0.3 .* randn(rng, n)), 0.02, 0.95)
        w = 1 ./ (mk .* (1 + 0.05 * rand(rng)))
        m = simultaneous_bets(p, w)
        (_, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test ρ ≤ 1e-10
    end

    # (b) seed 702, 150 correlated problems
    rng = StableRNG(702)
    for _ in 1:150
        m = rand_correlated_model(rng)
        (_, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test ρ ≤ 1e-10
    end

    # (c) seed 703, 100 single events with cash
    rng = StableRNG(703)
    for _ in 1:100
        q, ohat = rand_single_event_inputs(rng)
        m = single_event(q, ohat)
        (a, ρ) = polish_log_optimal(m.X, m.π, solve_conic(m))
        @test maximum(abs.(a .- greedy_allocation(q, ohat))) ≤ 1e-10
        @test ρ ≤ 1e-10
    end

    # (d) regression case (Task 021, seed 901, case 14)
    (a, ρ) = polish_log_optimal(_POLISH_REG_M.X, _POLISH_REG_M.π, _POLISH_REG_A0)
    @test ρ ≤ 1e-10
    @test a[1] > 0
end

@testset "polish: CRRA weights" begin
    # (a) Closed form, seed 1701, 500 cases
    rng = StableRNG(1701)
    for _ in 1:500
        p = 0.05 + 0.85 * rand(rng)
        w = (1.02 + 0.3 * rand(rng)) / p
        k = rand(rng, (0.0, 0.5, 2.0, 10.0))
        (a, ρ) = polish_crra([1.0 w; 1.0 0.0], [p, 1 - p], [0.5, 0.5], k)
        R = (p * (w - 1) / (1 - p))^(1 / (1 + k))
        @test abs(a[2] - (R - 1) / (w - 1 + R)) ≤ 1e-10
        @test ρ ≤ 1e-10
    end

    # (b) Correlated models, seed 1702
    rng = StableRNG(1702)
    N_b = QUICK ? 30 : 100
    for _ in 1:N_b
        m_b = rand_correlated_model(rng)
        k = exp(log(0.1) + rand(rng) * log(100))
        (a, ρ) = polish_crra(m_b.X, m_b.π, solve_conic(m_b), k)
        @test ρ ≤ 1e-10
    end

    # (c) Wrapper unchanged in behavior
    (a_c, ρ_c) = polish_log_optimal(_POLISH_REG_M.X, _POLISH_REG_M.π, _POLISH_REG_A0)
    @test ρ_c ≤ 1e-10
    @test a_c[1] > 0
    @test_throws ArgumentError polish_crra(_POLISH_REG_M.X, _POLISH_REG_M.π, _POLISH_REG_A0, -1.0)
end


@testset "T-BOX-10 polish with caps (Task 069)" begin
    draw_caps(rng, J) = [j == 1 ? Inf : (rand(rng) < 0.5 ? 0.02 + 0.3 * rand(rng) : Inf) for j in 1:J]

    # (a) release: warm start at the cap, optimum is interior
    Xa = [1.0 2.0; 1.0 0.0]; πa = [0.6, 0.4]; ua = [Inf, 0.3]
    a, ρ = Kelly_KL.polish_log_optimal(Xa, πa, [0.7, 0.3]; upper = ua)
    @test a[2] < ua[2] - 1e-9
    @test abs(a[2] - 0.2) ≤ 1e-9
    @test ρ ≤ 1e-10

    # (b) add with ν < r_j < 1
    rng = StableRNG(69)
    used = 0
    mb = nothing; ub = nothing; a0b = nothing; a_starb = nothing; jb = 0
    for d in 1:1000
        m = rand_correlated_model(rng)
        m.has_cash || continue
        u = draw_caps(rng, size(m.X, 2))
        a_star, _ = polish_log_optimal(m.X, m.π, solve_conic(m; upper = u); upper = u)
        for j in 2:length(a_star)
            (a_star[j] > 1e-6 && a_star[j] < u[j] - 1e-6) || continue
            a0 = copy(a_star); a0[1] += a0[j]; a0[j] = 0.0
            _, ν0, _, _, _ = Kelly_KL.kkt_components(m.X, a0, m.π; upper = u)
            rj = (m.X' * (m.π ./ (m.X * a0)))[j]
            if ν0 < rj < 1
                used = d; mb, ub, a0b, a_starb, jb = m, u, a0, a_star, j
                break
            end
        end
        used > 0 && break
    end
    println("T-BOX-10 (b): precondition met at draw $used")
    @test used > 0
    if used > 0
        a, ρ = polish_log_optimal(mb.X, mb.π, a0b; upper = ub)
        @test a[jb] > 0
        @test ρ ≤ 1e-10
        @test abs(growth(mb, a) - growth(mb, a_starb)) ≤ 1e-10
    end

    # (c) random robustness
    rng = StableRNG(691)
    n = 0
    while n < 100
        m = rand_correlated_model(rng)
        m.has_cash || continue
        n += 1
        u = draw_caps(rng, size(m.X, 2))
        a_c = solve_conic(m; upper = u)
        for k in (0.0, 1.5)
            a, ρ = polish_crra(m.X, m.π, a_c, k; upper = u)
            @test ρ ≤ 1e-10
            @test all(0 .≤ a .≤ u)
            @test abs(sum(a) - 1) ≤ 1e-14
            k == 0.0 && @test growth(m, a) ≥ growth(m, a_c) - 1e-12
        end
    end
end
