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

