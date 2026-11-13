@testset "certify" begin
    X = [1.0 2.0; 1.0 0.0]
    π = [0.6, 0.4]
    m = ScenarioModel(π, X; assets = ["cash", "bet"], has_cash = true, structure = :single_event)

    # (a) Kelly optimum b = 0.2
    @test kkt_residual(X, [0.8, 0.2], π) ≤ 1e-15

    # (b) Suboptimal interior allocation
    @test kkt_residual(X, [0.9, 0.1], π) ≥ 0.09

    # (c) Negativity term
    @test kkt_residual(X, [1.1, -0.1], π) ≥ 0.1

    # (d) Sum term
    @test kkt_residual(X, [0.7, 0.2], π) ≥ 0.1

    # (e) Zero wealth in positive-probability scenario
    @test kkt_residual(X, [0.0, 1.0], π) == Inf

    # (f) certify passes on optimal and throws CertificationError on suboptimal
    res_opt = BetResult(m, [0.8, 0.2], :test)
    @test certify(res_opt) === res_opt
    res_sub = BetResult(m, [0.9, 0.1], :test)
    @test_throws CertificationError certify(res_sub)

    # showerror coverage
    io = IOBuffer()
    showerror(io, CertificationError(0.1, 1e-10, :test, "uncertified bet"))
    @test occursin("CertificationError", String(take!(io)))
    showerror(io, SolverError("INFEASIBLE", "failed"))
    @test occursin("SolverError", String(take!(io)))

    # (g) assert_certified passes on optimal result
    assert_certified(res_opt, m)
end


@testset "T-BOX ν residual (Task 066)" begin
    using StableRNGs
    kc = Kelly_KL.kkt_components

    # T-BOX-5: frozen v0.1 residual, bitwise agreement without caps
    function ref_v01(X, a, π, τ_a)
        V = X * a
        any(s -> π[s] > 0 && !(V[s] > 0), eachindex(π)) && return Inf
        r = X' * [π[s] > 0 ? π[s] / V[s] : 0.0 for s in eachindex(π)]
        return max(maximum(r .- 1), maximum(abs, r[a .> τ_a] .- 1; init = 0.0), max(0.0, -minimum(a)), abs(sum(a) - 1))
    end
    simplex(rng, n) = (w = rand(rng, n) .+ 1e-3; w ./ sum(w))
    rng = StableRNG(66)
    for i in 1:500
        if i ≤ 250
            S = rand(rng, 2:10); J = rand(rng, 2:6)
            X = hcat(ones(S), 3 .* rand(rng, S, J - 1))
            π = simplex(rng, S); a = simplex(rng, J)
        else
            K = rand(rng, 2:6)
            q = simplex(rng, K); ohat = 1 .+ 3 .* rand(rng, K)
            mm = single_event(q, ohat; cash = true)
            X, π = mm.X, mm.π
            a = optimal_allocation(mm).a
            J = length(a)
        end
        ref = ref_v01(X, a, π, 1e-12)
        @test kkt_residual(X, a, π) === ref
        @test kkt_residual(X, a, π; upper = fill(Inf, J)) === ref
    end

    # T-BOX-12 / T-BOX-3 (v), A = -∞: all assets capped, a = u
    X3 = [3.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 6.0]
    π3 = [0.5, 0.3, 0.2]
    u3 = [0.2, 0.3, 0.5]
    ρ, ν, U, Z, F = kc(X3, u3, π3; upper = u3)
    @test U == [1, 2, 3]
    @test Z == Int[] && F == Int[]
    @test ρ == 0.0
    @test abs(ν - 0.4) ≤ 1e-15

    # T-BOX-3 (v), B = +∞: all zero, ν* = A
    Xb = [1.0 2.0; 1.0 0.5]
    πb = [0.5, 0.5]
    ab = [1e-13, 1e-13]
    ρ, ν, U, Z, F = kc(Xb, ab, πb; upper = [Inf, 0.5])
    rb = Xb' * (πb ./ (Xb * ab))
    @test Z == [1, 2]
    @test ν == maximum(rb)
    @test ρ == abs(sum(ab) - 1)

    # T-BOX-1 hand case
    Xh = [1.0 2.0; 1.0 0.0]
    πh = [0.6, 0.4]
    ρ, ν, U, Z, F = kc(Xh, [0.9, 0.1], πh; upper = [Inf, 0.1])
    @test U == [2]
    @test F == [1]
    @test ρ ≤ 1e-15
    @test abs(ν - 98 / 99) ≤ 1e-12
    r_bet = 0.6 * 2 / 1.1
    @test abs(r_bet - 12 / 11) ≤ 1e-12
end
