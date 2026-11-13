using StableRNGs, Random

@testset "T-BOX-8 validation (Task 064)" begin
    nu = Kelly_KL._normalize_upper
    mc = single_event([0.6, 0.4], [2.0, 2.5]; cash = true)           # J = 3
    mn = single_event([0.3, 0.3, 0.4], [4.0, 4.0, 3.0]; cash = false)  # J = 3
    msg(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError" end

    # (a) no cash, Σ u_j < 1 − BOX_FEAS_TOL
    for u in ([0.3, 0.3, 0.3], [0.3, 0.3, 0.4 - 2e-12])
        @test_throws ArgumentError nu(mn, u)
        @test occursin("Σ u_j", msg(() -> nu(mn, u)))
    end
    # (b) zero, negative, NaN caps
    for u in ([Inf, 0.0, 0.5], [Inf, -0.1, 0.5], [Inf, NaN, 0.5])
        @test_throws ArgumentError nu(mc, u)
    end
    # (c) wrong length
    @test_throws ArgumentError nu(mc, [Inf, 0.5])
    # (d) finite cap on cash
    @test_throws ArgumentError nu(mc, [0.5, 0.2, 0.2])
    # (e) within tolerance: accepted unchanged
    ue = [0.3, 0.3, 0.4 - 5e-13]
    @test nu(mn, ue) == ue

    # normalization
    @test nu(mc, [Inf, 0.2, 1.5]) == [Inf, 0.2, Inf]
    @test nu(mc, [1.0, 1.0, 2.0]) === nothing
    @test nu(mc, nothing) === nothing
end


@testset "T-BOX-11 projection (Task 065)" begin
    rng = StableRNG(65)
    for _ in 1:1000
        J = rand(rng, 2:8)
        x = 2 .* randn(rng, J)
        u = Float64[]
        while true
            u = [rand(rng) < 0.5 ? Inf : 0.05 + 0.95 * rand(rng) for _ in 1:J]
            sum(u) ≥ 1.05 && break
        end
        a = Kelly_KL._project_capped(x, u)
        # (1) feasibility
        @test all(a .≥ 0)
        @test all(a .≤ u)
        @test abs(sum(a) - 1) ≤ 1e-15
        # (2) idempotence
        @test maximum(abs.(Kelly_KL._project_capped(a, u) .- a)) ≤ 1e-15
        # (3) independent bisection
        f(θ) = sum(clamp.(x .- θ, 0.0, u)) - 1
        lo, hi = minimum(x) - 1, maximum(x)
        for _ in 1:200
            mid = (lo + hi) / 2
            f(mid) > 0 ? (lo = mid) : (hi = mid)
        end
        θb = (lo + hi) / 2
        @test maximum(abs.(a .- clamp.(x .- θb, 0.0, u))) ≤ 1e-12
    end
end
