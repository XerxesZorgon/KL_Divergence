struct RobustResult
    a::Vector{Float64}
    worst_case_growth::Float64
    nominal_growth::Float64
    λ::Float64
    Q::Vector{Float64}
    support::Matrix{Float64}
    kkt::Float64
    ball::Symbol
    ε::Float64
    certificate::Symbol
end

function _validate_kl_inner_inputs(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real, name::AbstractString)
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("$name requires finite ε ≥ 0, got $ε"))
    length(a) == size(m.X, 2) || throw(ArgumentError("$name: length(a) = $(length(a)) != size(m.X, 2) = $(size(m.X, 2))"))
    pos = findall(>(0.0), m.π)
    V = m.X[pos, :] * collect(Float64, a)
    all(x -> isfinite(x) && x > 0.0, V) || throw(ArgumentError("$name: all positive-probability scenarios must have V > 0"))
    return pos, V
end

"""
    kl_worst_case(m::ScenarioModel, a, ε) -> (value, λ, Q)

Worst-case expected log growth over the KL ball `{Q : D(Q‖π) ≤ ε}`; `Q* ∝ π·e^{−f/λ*}`;
a KL ball cannot put mass on scenarios `π` gives zero probability.
"""
function kl_worst_case(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real)
    pos, V = _validate_kl_inner_inputs(m, a, ε, "kl_worst_case")
    p = m.π[pos]; f = log.(V); fmin = minimum(f); g = f .- fmin
    (ε == 0 || maximum(g) ≤ 1e-15) && return (sum(p .* f), Inf, p)
    mins = g .≤ 0.0; D0 = -log(sum(p[mins]))                 # D(Q‖π) of the limit λ → 0
    if ε ≥ D0                                               # the ball contains the worst scenario(s) outright
        Q = p .* mins; Q ./= sum(Q); return (fmin, 0.0, Q)
    end
    Qof(λ) = (lw = log.(p) .- g ./ λ; lw .-= maximum(lw); Q = exp.(lw); Q ./ sum(Q))
    D(λ) = kl(Qof(λ), p)
    λ = sqrt(sum(p .* g .^ 2) / (2ε)) + 1e-300                # small-ε scale as a start
    lo = λ; while D(lo) ≤ ε; lo /= 2; end
    hi = λ; while D(hi) > ε; hi *= 2; end
    l, h = log(lo), log(hi)
    for _ in 1:300
        mid = (l + h) / 2
        D(exp(mid)) > ε ? (l = mid) : (h = mid)
        h - l ≤ 1e-15 && break
    end
    λ = exp((l + h) / 2); Q = Qof(λ)
    return (sum(Q .* f), λ, Q)
end

function kl_inner_direct(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real)
    pos, V = _validate_kl_inner_inputs(m, a, ε, "kl_inner_direct")
    p = m.π[pos]; f = log.(V); S_pos = length(pos)
    model = _tight_clarabel_model()
    @variable(model, Q[1:S_pos] >= 0)
    @variable(model, r[1:S_pos])
    @constraint(model, sum(Q) == 1)
    @constraint(model, sum(r) <= Float64(ε))
    for s in 1:S_pos
        @constraint(model, [-r[s], Q[s], p[s]] in MOI.ExponentialCone())
    end
    @objective(model, Min, sum(Q[s] * f[s] for s in 1:S_pos))
    JuMP.optimize!(model)
    _check_optimal(model, "kl_inner_direct conic solve failed")
    Q_val = _simplex_value(Q)
    return (sum(Q_val .* f), Q_val)
end

"""
    kl_cash_radius(m::ScenarioModel; maxit = 2000, tol = 1e-13) -> (radius, Q)

KL distance from `m` to the nearest "no-edge" world (`E_Q[X_j] ≤ 1` for every bet column),
measuring the KL robustness margin of the edge. `radius = 0` means Kelly already holds only
cash; `radius = Inf` means no `Q` in any KL ball removes every edge, so the KL-robust bet
never goes to all cash. Projected Newton on the dual; `Inf` is returned only with a proof;
an undecidable, nearly degenerate model throws `CertificationError`.
"""
function kl_cash_radius(m::ScenarioModel; maxit = 2000, tol = 1e-13)
    m.has_cash || throw(ArgumentError("kl_cash_radius needs a model with cash"))
    pos = findall(>(0.0), m.π); p = m.π[pos]
    Y = m.X[pos, 2:end] .- 1.0; n = size(Y, 2); μ = zeros(n)
    φ(μ) = (z = -(Y * μ) .+ log.(p); mz = maximum(z); -(mz + log(sum(exp.(z .- mz)))))   # concave dual; = D_min at the optimum
    Qof(μ) = (lw = log.(p) .- Y * μ; lw .-= maximum(lw); Q = exp.(lw); Q ./ sum(Q))   # Q ∝ π·exp(−Σ μ_j (X_j − 1))
    kkt(μ, g) = max(maximum(abs, g[μ .> 0]; init = 0.0), maximum(g[μ .== 0]; init = 0.0))
    φmax = log(1 / minimum(p))                     # D(Q‖π) ≤ log(1/π_min) for every Q
    for _ in 1:maxit                               # projected Newton (Bertsekas) on max φ(μ), μ ≥ 0
        Q = Qof(μ); g = Y' * Q; k0 = kkt(μ, g)
        k0 ≤ tol && break
        F = findall(j -> μ[j] > 0 || g[j] > 0, 1:n)
        YF = Y[:, F]; c = YF' * Q
        H = -(YF' * (YF .* Q) .- c * c')           # = −Cov_Q(Y_F)
        d = zeros(n); d[F] = (H - (1e-12 + norm(g[F])) * I) \ (-g[F])   # Levenberg: ascent even when H is singular
        t = 1.0; φ0 = φ(μ)
        while t > 1e-16
            μn = max.(μ .+ t .* d, 0.0)
            if k0 > 1e-6
                φ(μn) ≥ φ0 + 1e-4 * dot(g, μn .- μ) && break      # far from the optimum: Armijo on φ
            else
                kkt(μn, Y' * Qof(μn)) < k0 && break               # near it φ is flat to rounding: accept a smaller KKT residual
            end
            t /= 2
        end
        μn = max.(μ .+ t .* d, 0.0); μn == μ && break; μ = μn
        # exact proofs that no no-edge world exists: the dual exceeds every possible D, or μ is a bet mix beating cash everywhere
        (φ(μ) > φmax + 1e-9 || minimum(Y * μ) > 0) && return (Inf, Qof(μ))
    end
    all(==(0.0), μ) && return (0.0, p)             # no bet has an edge under π
    Q = Qof(μ)
    minimum(Y * μ) > 0 && return (Inf, Q)
    cert = maximum(Y' * Q)
    cert > 1e-12 && throw(CertificationError(cert, 1e-12, :kl, "cash radius undecided (nearly degenerate model)"))
    return (max(kl(Q, p), 0.0), Q)
end

function kl_cash_radius_direct(m::ScenarioModel)
    m.has_cash || throw(ArgumentError("kl_cash_radius_direct needs a model with cash"))
    pos = findall(>(0.0), m.π); p = m.π[pos]; S_pos = length(pos); J = size(m.X, 2)
    model = _tight_clarabel_model()
    @variable(model, Q[1:S_pos] >= 0)
    @variable(model, r[1:S_pos])
    @constraint(model, sum(Q) == 1)
    for j in 2:J
        @constraint(model, sum(Q[s] * m.X[pos[s], j] for s in 1:S_pos) <= 1.0)
    end
    for s in 1:S_pos
        @constraint(model, [-r[s], Q[s], p[s]] in MOI.ExponentialCone())
    end
    @objective(model, Min, sum(r))
    JuMP.optimize!(model)
    st = JuMP.termination_status(model)
    return st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ? (JuMP.objective_value(model), st) : (Inf, st)
end

function kl_robust_conic(m::ScenarioModel, ε::Real; solver::Symbol = :clarabel)
    pos = findall(>(0.0), m.π); p = m.π[pos]; Xp = m.X[pos, :]; S_pos = length(pos); J = size(m.X, 2)
    model = _conic_model(solver)
    @variable(model, a[1:J] >= 0)
    @variable(model, λ >= 0)
    @variable(model, t)
    @variable(model, w[1:S_pos])
    @variable(model, z[1:S_pos])
    @constraint(model, sum(a) == 1)
    @constraint(model, sum(p[s] * z[s] for s in 1:S_pos) <= λ)
    for s in 1:S_pos
        @constraint(model, [t - w[s], λ, z[s]] in MOI.ExponentialCone())
        @constraint(model, [w[s], 1.0, sum(Xp[s, j] * a[j] for j in 1:J)] in MOI.ExponentialCone())
    end
    @objective(model, Max, t - λ * Float64(ε))
    JuMP.optimize!(model)
    _check_optimal(model, "kl_robust_conic solve failed (:$solver)")
    return (_simplex_value(a), JuMP.value(λ))
end

"""
    kl_robust_allocation(m::ScenarioModel, ε; solver = :clarabel, tol = CERT_TOL) -> RobustResult

The bet maximizing worst-case expected log growth over the KL ball `{Q : D(Q‖π) ≤ ε}`;
equals the CRRA bet with `γ = 1 + 1/λ*`; all cash exactly when `ε ≥ kl_cash_radius(m)`;
a model without cash whose `λ* → 0` throws `CertificationError` (v1 limit).
"""
function kl_robust_allocation(m::ScenarioModel, ε::Real; solver::Symbol = :clarabel, tol = CERT_TOL)
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("kl_robust_allocation requires finite ε ≥ 0, got $ε"))
    pos = findall(>(0.0), m.π); p = m.π[pos]; Xp = m.X[pos, :]; J = size(m.X, 2); εf = Float64(ε)
    if εf == 0.0
        r = optimal_allocation(m; solver, tol)
        return RobustResult(r.a, r.growth, r.growth, Inf, p, Xp, r.kkt, :kl, 0.0, :kkt)
    end
    if m.has_cash
        (D_min, Qc) = kl_cash_radius(m)
        if εf ≥ D_min
            a = [1.0; zeros(J - 1)]
            ρ = kkt_residual(Xp, a, Qc)
            ρ ≤ tol || throw(CertificationError(ρ, tol, :kl, "cash certificate"))
            return RobustResult(a, 0.0, 0.0, 0.0, Qc, Xp, ρ, :kl, εf, :kkt)
        end
    end
    (a, λ0) = kl_robust_conic(m, εf; solver)
    function Dλ(λ)                                  # re-optimize the CRRA bet at λ, warm-started, and return D(Q‖π)
        a .= polish_crra(m.X, m.π, a, 1 / λ; tol)[1]
        Q, _ = crra_weights(Xp, p, a, 1 / λ)
        return kl(Q, p)
    end
    lo, hi = λ0 / 2, 2λ0
    n = 0; while Dλ(lo) ≤ εf; lo /= 2; (n += 1) > 60 && throw(CertificationError(Inf, tol, :kl, "no lower bracket for λ*")); end
    n = 0; while Dλ(hi) > εf; hi *= 2; (n += 1) > 60 && throw(CertificationError(Inf, tol, :kl, "no upper bracket for λ*")); end
    l, h = log(lo), log(hi)
    for _ in 1:200
        mid = (l + h) / 2
        Dλ(exp(mid)) > εf ? (l = mid) : (h = mid)
        h - l < 1e-13 && break
    end
    λ = exp((l + h) / 2)
    a, ρ = polish_crra(m.X, m.π, a, 1 / λ; tol)
    Q, V = crra_weights(Xp, p, a, 1 / λ)
    ρ ≤ tol || throw(CertificationError(ρ, tol, :kl, "saddle-point KKT residual $ρ > $tol"))
    d_err = abs(kl(Q, p) - εf)
    d_err ≤ tol || throw(CertificationError(d_err, tol, :kl, "KL boundary residual $d_err > $tol"))
    return RobustResult(a, sum(Q .* log.(V)), growth(m, a), λ, Q, Xp, ρ, :kl, εf, :kkt)
end

function wasserstein_support(m::ScenarioModel, unseen, cost)
    unseen isa AbstractMatrix || throw(ArgumentError("unseen must be a matrix"))
    size(unseen, 2) == size(m.X, 2) || throw(ArgumentError("size(unseen, 2) = $(size(unseen, 2)) != size(m.X, 2) = $(size(m.X, 2))"))
    all(x -> x isa Real && isfinite(x) && x ≥ 0.0, unseen) || throw(ArgumentError("unseen entries must be finite and ≥ 0"))
    pos = findall(>(0.0), m.π)
    Xs = m.X[pos, :]
    p = m.π[pos]
    Xt = vcat(Xs, Matrix{Float64}(unseen))
    S, T = size(Xs, 1), size(Xt, 1)
    if cost === :l1
        C = [sum(abs.(Xs[s, :] .- Xt[t, :])) for s in 1:S, t in 1:T]
    elseif cost === :linf
        C = [maximum(abs.(Xs[s, :] .- Xt[t, :])) for s in 1:S, t in 1:T]
    elseif cost isa AbstractMatrix{<:Real}
        size(cost) == (S, T) || throw(ArgumentError("cost matrix must have size ($S, $T), got $(size(cost))"))
        all(x -> isfinite(x) && x ≥ 0.0, cost) || throw(ArgumentError("cost matrix entries must be finite and ≥ 0"))
        all(s -> cost[s, s] == 0.0, 1:S) || throw(ArgumentError("cost matrix must satisfy C[s, s] == 0"))
        C = Matrix{Float64}(cost)
    else
        throw(ArgumentError("cost must be :l1, :linf, or an S×T matrix"))
    end
    return (Xs, Xt, C, p)
end

function _validate_wasserstein_inner(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real, unseen, cost, name::AbstractString)
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("$name requires finite ε ≥ 0, got $ε"))
    length(a) == size(m.X, 2) || throw(ArgumentError("length(a) = $(length(a)) != size(m.X, 2) = $(size(m.X, 2))"))
    Xs, Xt, C, p = wasserstein_support(m, unseen, cost)
    V = Xt * collect(Float64, a)
    all(x -> isfinite(x) && x > 0.0, V) || throw(ArgumentError("all support scenarios must have Xt * a > 0"))
    return Xt, C, p, V
end

"""
    wasserstein_worst_case(m::ScenarioModel, a, ε; unseen = zeros(0, size(m.X, 2)), cost = :l1) -> (value, λ, Q)

Worst-case expected log growth when an adversary may move probability mass along the support
at total transport cost `≤ ε`; unlike the KL ball it can move mass onto `unseen` scenarios the
model gives zero probability.
"""
function wasserstein_worst_case(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real;
                                unseen = zeros(0, size(m.X, 2)), cost = :l1)
    Xt, C, p, _ = _validate_wasserstein_inner(m, a, ε, unseen, cost, "wasserstein_worst_case")
    f = log.(Xt * a); S, T = size(C)
    cur = zeros(Int, S); events = Tuple{Float64,Int,Int,Int}[]       # (λ, s, old t, new t)
    for s in 1:S
        fmin = minimum(f); cand = findall(t -> f[t] == fmin, 1:T)
        ℓ = cand[argmin(C[s, cand])]; cur[s] = ℓ; λc = 0.0
        while C[s, ℓ] > 0
            best, bt = Inf, 0
            for t in 1:T
                C[s, t] < C[s, ℓ] || continue
                λt = max((f[t] - f[ℓ]) / (C[s, ℓ] - C[s, t]), λc)
                if λt < best || (λt == best && C[s, t] < C[s, bt]); best, bt = λt, t; end
            end
            push!(events, (best, s, ℓ, bt)); ℓ = bt; λc = best
        end
    end
    budget = sum(p[s] * C[s, cur[s]] for s in 1:S)
    if budget ≤ ε                              # λ* = 0: the budget reaches every scenario's worst outcome
        Q = zeros(T); for s in 1:S; Q[cur[s]] += p[s]; end
        return (sum(Q .* f), 0.0, Q)
    end
    sort!(events, by = e -> e[1])
    asg = copy(cur); i = 1
    while i ≤ length(events)
        λb = events[i][1]; j = i
        while j ≤ length(events) && events[j][1] == λb; j += 1; end
        newbudget = budget + sum(p[events[k][2]] * (C[events[k][2], events[k][4]] - C[events[k][2], events[k][3]]) for k in i:j-1)
        if newbudget ≤ ε
            θ = (ε - newbudget) / (budget - newbudget)   # share of the switching mass left on the old outcome: budget binds exactly
            Q = zeros(T); sw = Set(events[k][2] for k in i:j-1)
            for s in 1:S; s in sw || (Q[asg[s]] += p[s]); end
            for k in i:j-1; s, o, n = events[k][2], events[k][3], events[k][4]; Q[o] += θ * p[s]; Q[n] += (1 - θ) * p[s]; end
            return (sum(Q .* f), λb, Q)
        end
        for k in i:j-1; asg[events[k][2]] = events[k][4]; end
        budget = newbudget; i = j
    end
end

function wasserstein_inner_lp(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real;
                              unseen = zeros(0, size(m.X, 2)), cost = :l1)
    _, C, p, V = _validate_wasserstein_inner(m, a, ε, unseen, cost, "wasserstein_inner_lp")
    f = log.(V); S, T = size(C)
    model = _tight_clarabel_model()
    @variable(model, γ[1:S, 1:T] >= 0)
    for s in 1:S
        @constraint(model, sum(γ[s, t] for t in 1:T) == p[s])
    end
    @constraint(model, sum(γ[s, t] * C[s, t] for s in 1:S, t in 1:T) <= Float64(ε))
    @objective(model, Min, sum(γ[s, t] * f[t] for s in 1:S, t in 1:T))
    JuMP.optimize!(model)
    _check_optimal(model, "wasserstein_inner_lp solve failed")
    return JuMP.objective_value(model)
end

function wasserstein_robust_conic(m::ScenarioModel, ε::Real, Xs::Matrix{Float64}, Xt::Matrix{Float64},
                                  C::Matrix{Float64}, p::Vector{Float64}; solver::Symbol = :clarabel)
    S, T = size(C); J = size(m.X, 2)
    model = _conic_model(solver)
    @variable(model, a[1:J] >= 0)
    @variable(model, λ >= 0)
    @variable(model, u[1:S])
    @constraint(model, sum(a) == 1)
    @constraint(model, con[s = 1:S, t = 1:T],
                [u[s] - λ * C[s, t], 1.0, sum(Xt[t, j] * a[j] for j in 1:J)] in MOI.ExponentialCone())
    @objective(model, Max, sum(p[s] * u[s] for s in 1:S) - λ * Float64(ε))
    JuMP.optimize!(model)
    _check_optimal(model, "wasserstein_robust_conic solve failed (:$solver)")
    a_val = _simplex_value(a)
    Γ = [abs(JuMP.dual(con[s, t])[1]) for s in 1:S, t in 1:T]
    for s in 1:S
        rs = sum(@view(Γ[s, :]))
        rs > 0.0 ? (Γ[s, :] .*= p[s] / rs) : (Γ[s, s] = p[s])
    end
    cost_Γ = sum(Γ .* C)
    if cost_Γ > ε
        θ = Float64(ε) / cost_Γ
        I0 = zeros(S, T)
        for s in 1:S; I0[s, s] = p[s]; end
        Γ = θ .* Γ .+ (1 - θ) .* I0
    end
    return (a_val, Γ)
end

"""
    wasserstein_robust_allocation(m::ScenarioModel, ε; unseen = zeros(0, size(m.X, 2)), cost = :l1, solver = :clarabel, tol = CERT_TOL, gap_tol = GAP_TOL) -> RobustResult

Maximizes worst-case growth when an adversary may move mass (including onto `unseen` scenarios)
at transport cost `≤ ε`; tied worst cases are certified by a duality gap `≤ gap_tol` instead of
a KKT residual (ADR-006, exception).
"""
function wasserstein_robust_allocation(m::ScenarioModel, ε::Real;
                                       unseen = zeros(0, size(m.X, 2)), cost = :l1,
                                       solver::Symbol = :clarabel, tol = CERT_TOL, gap_tol = GAP_TOL)
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("wasserstein_robust_allocation requires finite ε ≥ 0, got $ε"))
    (Xs, Xt, C, p) = wasserstein_support(m, unseen, cost)
    εf = Float64(ε)
    if εf == 0.0
        r = optimal_allocation(m; solver, tol)
        return RobustResult(r.a, r.growth, r.growth, Inf, p, Xs, r.kkt, :wasserstein, 0.0, :kkt)
    end
    (a0, Γ) = wasserstein_robust_conic(m, εf, Xs, Xt, C, p; solver)
    wc(a) = wasserstein_worst_case(m, a, εf; unseen, cost)
    a = copy(a0); Q = wc(a)[3]; tried = [copy(a0)]
    for _ in 1:10
        a = polish_log_optimal(Xt, max.(Q, 1e-300), a; tol)[1]   # 1e-300 floor keeps every support row feasible
        push!(tried, copy(a))
        v, λ, Qn = wc(a)
        if maximum(abs.(Qn .- Q)) ≤ 1e-12
            act = Qn .> 0
            ρ = kkt_residual(Xt[act, :], a, Qn[act])
            ρ ≤ tol && return RobustResult(a, v, growth(m, a), λ, Qn, Xt, ρ, :wasserstein, εf, :kkt)
            break
        end
        Q = Qn
    end
    Qd = vec(sum(Γ, dims = 1))
    mQ = ScenarioModel(Qd, Xt; assets = m.assets, has_cash = m.has_cash && all(==(1.0), Xt[:, 1]), structure = :correlated)
    U = optimal_allocation(mQ; solver, tol).growth
    a = tried[argmax([wc(b)[1] for b in tried])]
    (L, λ, Qw) = wc(a)
    gap = U - L
    (-1e-12 ≤ gap ≤ gap_tol) || throw(CertificationError(gap, gap_tol, :wasserstein, "duality gap"))
    return RobustResult(a, L, growth(m, a), λ, Qw, Xt, gap, :wasserstein, εf, :duality_gap)
end

"""
    kl_radius_chi2(K::Integer, n::Integer; level = 0.95) -> Float64

Under the hypothesis that the model is right, `2n·D(p̂‖π)` is asymptotically `χ²_{K−1}` (G-test),
so a KL ball of this radius around the empirical distribution of `n` observations over `K`
outcomes covers the truth with probability ≈ `level`; asymptotic — E7 checks coverage by
simulation; verify the Ben-Tal et al. (2013) direction convention before citing.
"""
function kl_radius_chi2(K::Integer, n::Integer; level::Real = 0.95)
    K ≥ 2 || throw(ArgumentError("kl_radius_chi2 requires K ≥ 2, got $K"))
    n ≥ 1 || throw(ArgumentError("kl_radius_chi2 requires n ≥ 1, got $n"))
    (isfinite(level) && 0.0 < level < 1.0) || throw(ArgumentError("kl_radius_chi2 requires 0 < level < 1, got $level"))
    return Distributions.quantile(Distributions.Chisq(K - 1), Float64(level)) / (2 * n)
end




