using StableRNGs, Random, Distributions
using Kelly_KL: kl_cash_radius_direct, greedy_allocation, polish_crra, wasserstein_inner_lp
import JuMP
import MathOptInterface as MOI

@testset "robust: KL worst case (T-ORC-4, T-ORC-7)" begin
    # T-ORC-4, seed 1601
    rng = StableRNG(1601)
    N4 = QUICK ? 50 : 200
    for _ in 1:N4
        m = rand_correlated_model(rng, 4:39, 2:8)
        a = rand_simplex(rng, size(m.X, 2))
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
        m = rand_correlated_model(rng, 4:39, 2:8)
        a = rand_simplex(rng, size(m.X, 2))
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

@testset "robust: KL cash radius" begin
    # (a) Analytic
    m_a = correlated_bets([0.6, 0.4], [2.0; 0.0;;])
    (r_a, Q_a) = kl_cash_radius(m_a)
    @test abs(r_a - 0.020410997260127586) ≤ 1e-12
    @test abs(Q_a[1] - 0.5) ≤ 1e-12

    # (b) Kelly-cash ⇔ zero radius, seed 1801, 200 single events with cash
    rng = StableRNG(1801)
    for _ in 1:200
        q, ohat = rand_single_event_inputs(rng)
        @test (kl_cash_radius(single_event(q, ohat))[1] ≤ 1e-14) == (greedy_allocation(q, ohat)[1] == 1.0)
    end

    # (c) Certificate and oracle, seed 1802
    rng = StableRNG(1802)
    N_c = QUICK ? 50 : 200
    n_inf = 0
    for _ in 1:N_c
        m = rand_correlated_model(rng)
        (r, Q) = kl_cash_radius(m)
        (rd, st) = kl_cash_radius_direct(m)
        if isfinite(r)
            @test maximum(m.X[:, 2:end]' * Q) - 1 ≤ 1e-10
            @test r ≥ 0
            @test abs(r - rd) ≤ 1e-7
        else
            n_inf += 1
            @test st == Kelly_KL.MOI.INFEASIBLE
        end
    end
    println("kl_cash_radius (c): $n_inf / $N_c cases with radius == Inf")

    # (d) Dominant bet
    @test kl_cash_radius(correlated_bets([0.5, 0.5], [1.5; 1.2;;]))[1] == Inf

    # (e) Validation
    @test_throws ArgumentError kl_cash_radius(single_event([0.5, 0.5], [2.0, 2.0]; cash = false))
end

@testset "T-BOX-6 cash radius with caps, first clause (Task 073)" begin
    rng = StableRNG(73)
    models = ScenarioModel[]; draws = 0; skipped_cert = 0
    while length(models) < 100 && draws < 2000
        draws += 1
        m = rand_correlated_model(rng)
        m.has_cash || continue
        r = try
            kl_cash_radius(m)[1]
        catch e
            e isa CertificationError ? (skipped_cert += 1; continue) : rethrow()
        end
        (isfinite(r) && r > 1e-3) || continue
        push!(models, m)
    end
    println("T-BOX-6 clause 1: draws = ", draws, ", skipped_cert = ", skipped_cert, ", collected = ", length(models))
    @test length(models) == 100
    for m in models
        J = size(m.X, 2)
        u = fill(Inf, J)
        for j in 2:J
            u[j] = 0.02 + 0.28 * rand(rng)
        end
        (r0, Q0) = kl_cash_radius(m)
        (r1, Q1) = kl_cash_radius(m; upper = u)
        # Tuples of arrays are never `===` (arrays compare by identity), so `r1 === r0` (scalar, bitwise)
        # together with `Q1 == Q0` (elementwise) is the bitwise check.
        @test r1 === r0
        @test Q1 == Q0
    end
    if !isempty(models)
        m = first(models); J = size(m.X, 2)
        @test_throws ArgumentError kl_cash_radius(m; upper = [0.5; fill(Inf, J - 1)])
        @test_throws ArgumentError kl_cash_radius(m; upper = fill(Inf, J - 1))
    end
end

@testset "robust: KL-robust bet (T-ORC-6, T-ORC-10)" begin
    # (a) Analytic single bet
    m_a = correlated_bets([0.6, 0.4], [2.0; 0.0;;])
    for (ε, f) in (
        (0.001, 0.1560563420293526),
        (0.005, 0.10143706865613433),
        (0.01, 0.06033774487079824),
        (0.015, 0.02872571386602285),
    )
        r = kl_robust_allocation(m_a, ε)
        @test abs(r.a[2] - f) ≤ 1e-10
        @test r.kkt ≤ 1e-10
    end
    r_cash = kl_robust_allocation(m_a, 0.03)
    @test r_cash.a == [1.0, 0.0]
    @test r_cash.worst_case_growth == 0.0

    # (b) T-ORC-6 and max-min property, seed 1901
    rng = StableRNG(1901)
    N_b = QUICK ? 20 : 100
    n_cash = 0
    for _ in 1:N_b
        m = rand_correlated_model(rng)
        ε = exp(log(1e-3) + rand(rng) * (log(0.5) - log(1e-3)))
        r = kl_robust_allocation(m, ε)
        r.a[1] == 1.0 && (n_cash += 1)
        @test r.kkt ≤ 1e-10
        @test abs(r.worst_case_growth - kl_worst_case(m, r.a, ε)[1]) ≤ 1e-10
        J = size(m.X, 2)
        if 0 < r.λ < Inf
            @test abs(kl(r.Q, m.π) - ε) ≤ 1e-10
            ac, _ = polish_crra(m.X, m.π, fill(1 / J, J), 1 / r.λ)
            @test maximum(abs.(ac .- r.a)) ≤ 1e-8
        end
        for _ in 1:5
            b = rand_simplex(rng, J)
            @test kl_worst_case(m, b, ε)[1] ≤ r.worst_case_growth + 1e-10
        end
    end
    println("kl_robust_allocation (b): $n_cash / $N_b models returned all cash")

    # (c) T-ORC-10 (KL): seed 1902, 20 correlated models
    rng = StableRNG(1902)
    for _ in 1:20
        m = rand_correlated_model(rng)
        @test maximum(abs.(kl_robust_allocation(m, 0.0).a .- optimal_allocation(m).a)) ≤ 1e-9
    end

    # (d) Validation
    @test_throws ArgumentError kl_robust_allocation(m_a, -0.1)
end

@testset "robust: Wasserstein worst case (T-ORC-9 ii)" begin
    # (a) Breakpoint scan vs LP, seed 2001
    rng = StableRNG(2001)
    N_a = QUICK ? 60 : 300
    for _ in 1:N_a
        m = rand_correlated_model(rng, 3:29, 1:6)
        J = size(m.X, 2)
        unseen = rand_unseen(rng, J - 1)
        a = rand_simplex(rng, J)
        ε = exp(log(1e-3) + rand(rng) * (log(3.0) - log(1e-3)))
        (v, λ, Q) = wasserstein_worst_case(m, a, ε; unseen)
        @test abs(v - wasserstein_inner_lp(m, a, ε; unseen)) ≤ 1e-7
        @test abs(sum(Q) - 1) ≤ 1e-12
        @test abs(sum(Q .* log.(vcat(m.X, unseen) * a)) - v) ≤ 1e-12
    end

    # (b) Unseen catastrophe
    m_b = correlated_bets([0.5, 0.5], [3.0 0.0; 0.0 2.0])
    a_b = [0.2, 0.4, 0.4]
    unseen_b = [1.0 0.0 0.0]
    @test abs(wasserstein_worst_case(m_b, a_b, 0.0; unseen = unseen_b)[1] - (0.5 * log(1.4) + 0.5 * log(1.0))) ≤ 1e-12
    @test abs(wasserstein_worst_case(m_b, a_b, 10.0; unseen = unseen_b)[1] - log(0.2)) ≤ 1e-12

    # (c) cost = :linf and user S × T matrix
    @test abs(wasserstein_worst_case(m_b, a_b, 0.5; unseen = unseen_b, cost = :linf)[1] -
              wasserstein_inner_lp(m_b, a_b, 0.5; unseen = unseen_b, cost = :linf)) ≤ 1e-7
    C_user = [0.0 2.0 1.5; 2.0 0.0 1.0]
    @test abs(wasserstein_worst_case(m_b, a_b, 0.5; unseen = unseen_b, cost = C_user)[1] -
              wasserstein_inner_lp(m_b, a_b, 0.5; unseen = unseen_b, cost = C_user)) ≤ 1e-7

    # (d) Validation
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, 0.5; unseen = zeros(1, 2))
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, 0.5; unseen = [1.0 -0.1 0.0])
    @test_throws ArgumentError wasserstein_worst_case(m_b, a_b, -1.0; unseen = unseen_b)
    @test_throws ArgumentError wasserstein_worst_case(m_b, [0.0, 0.5, 0.5], 0.5; unseen = unseen_b)
end

@testset "robust: Wasserstein-robust bet (T-ORC-9 i, T-ORC-10, T-ORC-11)" begin
    # (a) T-ORC-9 (i) and certificates, seed 2101
    rng = StableRNG(2101)
    N_a = QUICK ? 30 : 100
    n_kkt = 0
    n_gap = 0
    max_gap = 0.0
    for _ in 1:N_a
        m = rand_correlated_model(rng, 3:19, 1:5)
        J = size(m.X, 2)
        unseen = rand_unseen(rng, J - 1)
        ε = exp(log(1e-3) + rand(rng) * (log(2.0) - log(1e-3)))
        r = wasserstein_robust_allocation(m, ε; unseen)
        @test abs(r.worst_case_growth - wasserstein_worst_case(m, r.a, ε; unseen)[1]) ≤ 1e-10
        @test r.certificate == :kkt ? r.kkt ≤ 1e-10 : (r.certificate == :duality_gap && r.kkt ≤ 1e-7)
        if r.certificate == :kkt
            n_kkt += 1
        elseif r.certificate == :duality_gap
            n_gap += 1
            max_gap = max(max_gap, r.kkt)
        end
        for _ in 1:5
            b = rand_simplex(rng, J)
            @test wasserstein_worst_case(m, b, ε; unseen)[1] ≤ r.worst_case_growth + (r.certificate == :kkt ? 1e-10 : 1e-7)
        end
    end
    println("wasserstein_robust_allocation (a): :kkt = $n_kkt, :duality_gap = $n_gap, max_gap = $max_gap")

    # (b) T-ORC-10 (W): seed 2102, 20 models
    rng = StableRNG(2102)
    for _ in 1:20
        m = rand_correlated_model(rng, 3:19, 1:5)
        @test maximum(abs.(wasserstein_robust_allocation(m, 0.0).a .- optimal_allocation(m).a)) ≤ 1e-9
    end

    # (c) T-ORC-11 (unseen catastrophe forces cash)
    m_c = correlated_bets([0.5, 0.5], [3.0 0.0; 0.0 2.0])
    unseen_c = [1.0 0.0 0.0]
    @test optimal_allocation(m_c).a[1] ≤ 1e-12
    for ε in (0.01, 1.0)
        r = wasserstein_robust_allocation(m_c, ε; unseen = unseen_c)
        @test r.a[1] > 0
        @test isfinite(r.worst_case_growth)
    end
    @test kl_cash_radius(m_c)[1] == Inf
    @test kl_robust_allocation(m_c, 0.01).a[1] ≤ 1e-9

    # (d) KL results carry :kkt
    @test kl_robust_allocation(correlated_bets([0.6, 0.4], [2.0; 0.0;;]), 0.01).certificate == :kkt
end

@testset "robust: KL radius (T-DIA-7)" begin
    @test abs(kl_radius_chi2(2, 100) - 0.01920729410347062) ≤ 1e-14
    @test abs(kl_radius_chi2(5, 500; level = 0.99) - 0.013276704135987622) ≤ 1e-14
    for K in (2, 3, 10), n in (10, 1000)
        @test kl_radius_chi2(K, n) == quantile(Distributions.Chisq(K - 1), 0.95) / (2n)
    end
    @test_throws ArgumentError kl_radius_chi2(1, 10)
    @test_throws ArgumentError kl_radius_chi2(3, 0)
    @test_throws ArgumentError kl_radius_chi2(3, 10; level = 1.0)
end





@testset "Fix 074a safeguarded λ search" begin
    # model 84 of the T-BOX-6 draw (StableRNG(73)), uncapped; copied with full precision
    π84 = [0.07136346342659079, 0.007476906034825119, 0.7663115227991835, 0.15484810773940055]
    X84 = [1.0 2.683524870969485 0.0 0.0 1.9720652904384073 0.0; 1.0 0.0 0.0 0.0 0.0 0.0; 1.0 4.446691684308016 1.5473295361924109 0.0 3.3232896413528072 3.2240975149918247; 1.0 0.0 0.0 0.0 2.2969988383777684 2.7320440246679993]
    m84 = ScenarioModel(π84, X84; assets = ["cash", "bet_1", "bet_2", "bet_3", "bet_4", "bet_5"], has_cash = true, structure = :correlated)
    D_min = kl_cash_radius(m84)[1]
    p84 = m84.π[m84.π .> 0]
    function tight_lambda(m, ε)                       # true λ* from the conic program at tight tolerance (diagnostic step 4)
        pos = findall(>(0.0), m.π); p = m.π[pos]; Xp = m.X[pos, :]; S = length(pos); J = size(m.X, 2)
        model = Kelly_KL._tight_clarabel_model()
        JuMP.@variable(model, a[1:J] >= 0); JuMP.@variable(model, λ >= 0); JuMP.@variable(model, t)
        JuMP.@variable(model, w[1:S]); JuMP.@variable(model, z[1:S])
        JuMP.@constraint(model, sum(a) == 1)
        JuMP.@constraint(model, sum(p[s] * z[s] for s in 1:S) <= λ)
        for s in 1:S
            JuMP.@constraint(model, [t - w[s], λ, z[s]] in MOI.ExponentialCone())
            JuMP.@constraint(model, [w[s], 1.0, sum(Xp[s, j] * a[j] for j in 1:J)] in MOI.ExponentialCone())
        end
        JuMP.@objective(model, Max, t - λ * ε)
        JuMP.optimize!(model)
        return (JuMP.value(λ), JuMP.objective_value(model))   # (λ_c, v_c = t − λ_c ε)
    end
    for f in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
        ε = f * D_min
        r = kl_robust_allocation(m84, ε)
        λ_c, v_c = tight_lambda(m84, ε)
        rel = abs(r.λ - λ_c) / λ_c
        dv = abs(r.worst_case_growth - v_c)
        gap = abs(r.worst_case_growth - kl_worst_case(m84, r.a, ε)[1])
        println("Fix 074a model 84, ε/D_min = ", f, ": λ* = ", r.λ, ", rel(λ) = ", rel, ", |Δv| = ", dv, ", kl_worst_case gap = ", gap, ", ρ = ", r.kkt)
        @test r.kkt ≤ 1e-10
        @test abs(Kelly_KL.kl(r.Q, p84) - ε) ≤ 1e-10
        # coarse guard: the λ search failure this fix addresses was ~1e6 off; λ_c from the conic is noisy at ~1e-6 because the
        # objective is flat in λ (envelope); λ* accuracy is implied by the two certificates above (slope dD/dlnλ ≈ 0.67 ⇒ ~1.5e-10).
        @test rel ≤ 1e-4
        @test dv ≤ 1e-8
        @test gap ≤ 1e-10
    end

    # unchanged path: control models; report how often the fallback ran
    rng = StableRNG(742); cnt = 0; fb0 = Kelly_KL._KL_FALLBACK_COUNT[]
    while cnt < 20
        m = rand_correlated_model(rng)
        m.has_cash || continue
        Dm = try kl_cash_radius(m)[1] catch e; e isa CertificationError ? continue : rethrow() end
        (isfinite(Dm) && Dm > 1e-3) || continue
        cnt += 1
        r = kl_robust_allocation(m, 0.5 * Dm)
        @test r.kkt ≤ 1e-10
    end
    println("Fix 074a control models: fallback ran on ", Kelly_KL._KL_FALLBACK_COUNT[] - fb0, "/20")
end
