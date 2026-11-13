using SHA

const _BOX_HEADER, _BOX_ALLOC, _BOX_KL = load_box_oracle()

# @test with the fixture case id, the quantity and the observed value printed on failure (a plain @test cannot name the case).
function _chk(ok::Bool, id, what, observed = nothing)
    ok || println("  FIXTURE CASE ", id, ": ", what, observed === nothing ? "" : "; observed " * string(observed))
    @test ok
end

@testset "T-BOX-15 fixture integrity" begin
    h = _BOX_HEADER
    for kind in ("allocation", "kl")
        hk = h[kind]
        @test hk["rate"] ≤ 0.10
        @test hk["rejected"] / hk["attempted"] == hk["rate"]
        @test hk["attempted"] - hk["rejected"] == length(kind == "allocation" ? _BOX_ALLOC : _BOX_KL)
    end
    for k in ("seed", "python", "cvxpy", "clarabel")
        @test haskey(h, k)
    end
    @test bytes2hex(sha256(read(joinpath(@__DIR__, "oracle", "requirements.txt")))) == h["requirements_sha256"]

    for c in _BOX_ALLOC
        ρ, ν, U, Z, F = Kelly_KL.kkt_components(c.X, c.a, c.pi; τ_a = 1e-8, upper = c.upper)
        _chk(ρ ≤ 1e-8, c.id, "oracle ρ_u ≤ 1e-8 (τ_o)", ρ)
        _chk((U, Z, F) == (c.U, c.Z, c.F), c.id, "recomputed U/Z/F equal stored", ((U, Z, F), (c.U, c.Z, c.F)))
        r = c.X' * (c.pi ./ (c.X * c.a))
        nondeg = all(r[j] ≤ c.nu_star - 1e-6 for j in c.Z) && all(r[j] ≥ c.nu_star + 1e-6 for j in c.U)
        _chk(c.degenerate == !nondeg, c.id, "stored degenerate flag reproduces", (c.degenerate, !nondeg))
    end

    for c in _BOX_KL
        V = c.X * c.a
        lw = log.(c.pi) .- log.(V) ./ c.lambda
        mx = maximum(lw)
        lw .-= mx .+ log(sum(exp.(lw .- mx)))
        Q = exp.(lw)
        D = sum(Q[s] * (lw[s] - log(c.pi[s])) for s in eachindex(Q) if Q[s] > 0)
        ρ, _, _, _, _ = Kelly_KL.kkt_components(c.X, c.a, Q; τ_a = 1e-8, upper = c.upper)
        _chk(ρ ≤ 1e-8, c.id, "saddle ρ_u ≤ 1e-8 (τ_o)", ρ)
        _chk(abs(D - c.eps) ≤ 1e-8, c.id, "|D(Q*‖π) − ε| ≤ 1e-8", abs(D - c.eps))
    end
    println("T-BOX-15: KL fixtures eps range = ", extrema(c.eps for c in _BOX_KL), "; ", length(_BOX_ALLOC), " allocation, ", length(_BOX_KL), " kl cases")
end

@testset "T-BOX-2 / T-BOX-3 capped problems vs fixtures" begin
    n_nondeg = 0; n_i = 0; n_ii = 0; n_iii = 0; n_iv = 0; max_gratio = 0.0
    for c in _BOX_ALLOC
        m = model_of(c)
        r = optimal_allocation(m; upper = c.upper)
        # T-BOX-2
        _chk(r.kkt ≤ 1e-10, c.id, "library ρ_u ≤ 1e-10", r.kkt)
        # case-specific growth tolerance (ADR-014 amendment 2026-11-13): the oracle point is only ρ_u(oracle)-optimal
        gtol = 1e-10 + 3 * c.rho_u
        gratio = abs(r.growth - c.growth) / gtol
        max_gratio = max(max_gratio, gratio)
        _chk(abs(r.growth - c.growth) ≤ gtol, c.id, "|G − G_oracle| ≤ 1e-10 + 3·ρ_u(oracle) = $gtol", abs(r.growth - c.growth))
        _, ν, U, Z, F = Kelly_KL.kkt_components(m.X, r.a, m.π; upper = r.upper)
        if !c.degenerate
            n_nondeg += 1
            _chk(maximum(abs.(r.a .- c.a)) ≤ 1e-8, c.id, "‖a − a_oracle‖∞ ≤ 1e-8", maximum(abs.(r.a .- c.a)))
            _chk((U, Z, F) == (c.U, c.Z, c.F), c.id, "library U/Z/F equal stored", ((U, Z, F), (c.U, c.Z, c.F)))
        end
        # T-BOX-3
        rv = m.X' * (m.π ./ (m.X * r.a))
        n_i += 1
        for j in F
            _chk(abs(rv[j] - ν) ≤ 1e-10, c.id, "(i) |r_j − ν*| ≤ 1e-10 at free j = $j", abs(rv[j] - ν))
        end
        if !c.degenerate
            n_ii += 1
            _chk(abs(ν - c.nu_star) ≤ 1e-8, c.id, "(ii) |ν* − ν*_oracle| ≤ 1e-8", abs(ν - c.nu_star))
        end
        if !isempty(F) && sum(r.a[F]) ≥ 0.01
            n_iii += 1
            ν_cf = (1 - sum(r.upper[j] * rv[j] for j in U; init = 0.0)) / (1 - sum(r.upper[j] for j in U; init = 0.0))
            _chk(abs(ν_cf - ν) ≤ 1e-9, c.id, "(iii) closed-form ν agrees within 1e-9", abs(ν_cf - ν))
        end
        if any(j -> rv[j] > 1, U)
            n_iv += 1
            _chk(ν < 1, c.id, "(iv) ν* < 1 when a capped asset has r_j > 1", ν)
        end
    end
    println("T-BOX-2: ", length(_BOX_ALLOC), " cases (", n_nondeg, " non-degenerate); T-BOX-3: (i) ", n_i, " (ii) ", n_ii, " (iii) ", n_iii, " (iv) ", n_iv, "; largest |ΔG| / (1e-10 + 3ρ_u) = ", max_gratio)
end
