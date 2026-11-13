using StableRNGs, Random
using Kelly_KL: solve_conic, greedy_allocation

@testset "conic (raw)" begin
    # (a) seed 601 (StableRNG), 200 random single events with cash
    rng = StableRNG(601)
    for _ in 1:200
        q, ohat = rand_single_event_inputs(rng)
        # allocation distance is ill-conditioned near marginal bets (seed 601, trial 25: 1.0e-4 apart, growth gap 4.6e-10)
        m = single_event(q, ohat); a_c = solve_conic(m); a_g = greedy_allocation(q, ohat)
        @test growth(m, a_g) - growth(m, a_c) ≤ 1e-8
        @test maximum(abs.(a_c .- a_g)) ≤ 1e-3
    end

    # (b) simultaneous_bets with Clarabel
    m_sim = simultaneous_bets([0.3, 0.25], [4.0, 6.0])
    a = solve_conic(m_sim)
    @test abs(a[2] - 0.06450023) ≤ 1e-4
    @test abs(a[3] - 0.09915443) ≤ 1e-4

    # (c) simultaneous_bets with SCS
    a_scs = solve_conic(m_sim; solver = :scs)
    @test abs(a_scs[2] - 0.06450023) ≤ 1e-3
    @test abs(a_scs[3] - 0.09915443) ≤ 1e-3

    # (d) zero-probability rows are excluded
    m_zero = ScenarioModel([0.5, 0.5, 0.0], [2.0 0.0; 0.0 3.0; 0.0 0.0];
                           assets = ["b1", "b2"], has_cash = false, structure = :correlated)
    @test maximum(abs.(solve_conic(m_zero) .- [0.5, 0.5])) ≤ 1e-4

    # (e) invalid solver throws ArgumentError
    @test_throws ArgumentError solve_conic(m_zero; solver = :foo)

    # (f) uncapped path unchanged (exact equality)
    a_ref = solve_conic(m_sim)
    @test a_ref == solve_conic(m_sim; upper = nothing)
    @test a_ref == solve_conic(m_sim; upper = fill(Inf, size(m_sim.X, 2)))

    # (g) 200 random capped problems: raw output lies in A_u
    rng68 = StableRNG(68)
    n_bind = 0
    n_done = 0
    while n_done < 200
        mc = rand_correlated_model(rng68)
        mc.has_cash || continue
        n_done += 1
        Jc = size(mc.X, 2)
        uc = fill(Inf, Jc)
        for j in 2:Jc
            rand(rng68) < 0.5 && (uc[j] = 0.02 + 0.3 * rand(rng68))
        end
        ac = solve_conic(mc; upper = uc)
        @test all(ac .≥ -1e-8)
        @test all(ac .≤ uc .+ 1e-8)
        @test abs(sum(ac) - 1) ≤ 1e-8
        any(isfinite.(uc) .& (ac .≥ uc .- 1e-6)) && (n_bind += 1)
    end
    println("T-conic (g): binding-cap cases = $n_bind / 200")
    @test n_bind ≥ 1

    # (h) error cases
    @test_throws ArgumentError solve_conic(m_sim; upper = [0.5, Inf, Inf])    # cap on cash
    @test_throws ArgumentError solve_conic(m_zero; upper = [0.5, 0.5])        # near-point, no cash, Σu = 1
end

