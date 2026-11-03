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
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_feas", 1e-12)
    JuMP.set_optimizer_attribute(model, "max_iter", 500)
    @variable(model, Q[1:S_pos] >= 0)
    @variable(model, r[1:S_pos])
    @constraint(model, sum(Q) == 1)
    @constraint(model, sum(r) <= Float64(ε))
    for s in 1:S_pos
        @constraint(model, [-r[s], Q[s], p[s]] in MOI.ExponentialCone())
    end
    @objective(model, Min, sum(Q[s] * f[s] for s in 1:S_pos))
    JuMP.optimize!(model)
    st = JuMP.termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || throw(SolverError(string(st), "kl_inner_direct conic solve failed"))
    Q_val = max.(JuMP.value.(Q), 0.0)
    Q_val ./= sum(Q_val)
    return (sum(Q_val .* f), Q_val)
end

"""
    kl_cash_radius(m::ScenarioModel; newton_max = 100, outer_max = 20, tol = 1e-12) -> (radius, Q)

KL distance from `m` to the nearest "no-edge" world (`E_Q[X_j] ≤ 1` for every bet column),
measuring the KL robustness margin of the edge. `radius = 0` means Kelly already holds only
cash; `radius = Inf` means no `Q` in any KL ball removes every edge, so the KL-robust bet
never goes to all cash.
"""
function kl_cash_radius(m::ScenarioModel; newton_max = 100, outer_max = 20, tol = 1e-12)
    m.has_cash || throw(ArgumentError("kl_cash_radius needs a model with cash"))
    pos = findall(>(0.0), m.π); p = m.π[pos]
    Y = m.X[pos, 2:end] .- 1.0; n = size(Y, 2); μ = zeros(n)
    Qof(μ) = (lw = log.(p) .- Y * μ; lw .-= maximum(lw); Q = exp.(lw); Q ./ sum(Q))   # Q ∝ π·exp(−Σ μ_j (X_j − 1))
    for _ in 1:outer_max
        for _ in 1:newton_max
            A = findall(>(0.0), μ); isempty(A) && break
            Q = Qof(μ); g = Y' * Q                       # gradient of the concave dual = E_Q[X_j − 1]
            YA = Y[:, A]; c = YA' * Q
            H = -(YA' * (YA .* Q) .- c * c')             # Hessian = −Cov_Q
            d = pinv(H) * (-g[A])
            tmax, iblk = 1.0, 0
            for i in eachindex(A)
                if d[i] < 0 && μ[A[i]] + d[i] ≤ 0.0
                    ti = μ[A[i]] / -d[i]; ti < tmax && ((tmax, iblk) = (ti, i))
                end
            end
            μ[A] .+= tmax .* d; iblk > 0 && (μ[A[iblk]] = 0.0); μ .= max.(μ, 0.0)
            iblk == 0 && maximum(abs, tmax .* d) < 1e-15 && break
        end
        Q = Qof(μ); g = Y' * Q
        add = [j for j in 1:n if μ[j] == 0.0 && g[j] > tol]
        isempty(add) && maximum(abs, g[μ .> 0]; init = 0.0) ≤ tol && break
        for j in add; μ[j] = 1e-6; end
    end
    all(==(0.0), μ) && return (0.0, p)          # no bet has an edge under π: the model itself is the nearest no-edge world
    Q = Qof(μ)
    maximum(Y' * Q; init = -Inf) > tol && return (Inf, Q)   # no-edge world not reached: some bet mix beats cash everywhere
    return (max(kl(Q, p), 0.0), Q)
end

function kl_cash_radius_direct(m::ScenarioModel)
    m.has_cash || throw(ArgumentError("kl_cash_radius_direct needs a model with cash"))
    pos = findall(>(0.0), m.π); p = m.π[pos]; S_pos = length(pos)
    J = size(m.X, 2)
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_feas", 1e-12)
    JuMP.set_optimizer_attribute(model, "max_iter", 500)
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
    solver in (:clarabel, :scs) || throw(ArgumentError("solver must be :clarabel or :scs, got :$solver"))
    pos = findall(>(0.0), m.π); p = m.π[pos]; Xp = m.X[pos, :]; S_pos = length(pos); J = size(m.X, 2)
    model = JuMP.Model(solver === :clarabel ? Clarabel.Optimizer : SCS.Optimizer)
    JuMP.set_silent(model)
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
    st = JuMP.termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || throw(SolverError(string(st), "kl_robust_conic solve failed (:$solver)"))
    a_val = max.(JuMP.value.(a), 0.0)
    return (a_val ./ sum(a_val), JuMP.value(λ))
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
        return RobustResult(r.a, r.growth, r.growth, Inf, p, Xp, r.kkt, :kl, 0.0)
    end
    if m.has_cash
        (D_min, Qc) = kl_cash_radius(m)
        if εf ≥ D_min
            a = [1.0; zeros(J - 1)]
            ρ = kkt_residual(Xp, a, Qc)
            ρ ≤ tol || throw(CertificationError(ρ, tol, :kl, "cash certificate"))
            return RobustResult(a, 0.0, 0.0, 0.0, Qc, Xp, ρ, :kl, εf)
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
    return RobustResult(a, sum(Q .* log.(V)), growth(m, a), λ, Q, Xp, ρ, :kl, εf)
end

function wasserstein_support(m::ScenarioModel, unseen, cost)
    unseen isa AbstractMatrix{<:Real} || throw(ArgumentError("unseen must be a real matrix"))
    size(unseen, 2) == size(m.X, 2) || throw(ArgumentError("size(unseen, 2) = $(size(unseen, 2)) != size(m.X, 2) = $(size(m.X, 2))"))
    all(x -> isfinite(x) && x ≥ 0.0, unseen) || throw(ArgumentError("unseen entries must be finite and ≥ 0"))
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

"""
    wasserstein_worst_case(m::ScenarioModel, a, ε; unseen = zeros(0, size(m.X, 2)), cost = :l1) -> (value, λ, Q)

Worst-case expected log growth when an adversary may move probability mass along the support
at total transport cost `≤ ε`; unlike the KL ball it can move mass onto `unseen` scenarios the
model gives zero probability.
"""
function wasserstein_worst_case(m::ScenarioModel, a::AbstractVector{<:Real}, ε::Real;
                                unseen = zeros(0, size(m.X, 2)), cost = :l1)
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("wasserstein_worst_case requires finite ε ≥ 0, got $ε"))
    length(a) == size(m.X, 2) || throw(ArgumentError("length(a) = $(length(a)) != size(m.X, 2) = $(size(m.X, 2))"))
    Xs, Xt, C, p = wasserstein_support(m, unseen, cost)
    V = Xt * collect(Float64, a)
    all(x -> isfinite(x) && x > 0.0, V) || throw(ArgumentError("all support scenarios must have Xt * a > 0"))
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
    (isfinite(ε) && ε ≥ 0.0) || throw(ArgumentError("wasserstein_inner_lp requires finite ε ≥ 0, got $ε"))
    length(a) == size(m.X, 2) || throw(ArgumentError("length(a) = $(length(a)) != size(m.X, 2) = $(size(m.X, 2))"))
    Xs, Xt, C, p = wasserstein_support(m, unseen, cost)
    V = Xt * collect(Float64, a)
    all(x -> isfinite(x) && x > 0.0, V) || throw(ArgumentError("all support scenarios must have Xt * a > 0"))
    f = log.(V); S, T = size(C)
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_feas", 1e-12)
    JuMP.set_optimizer_attribute(model, "max_iter", 500)
    @variable(model, γ[1:S, 1:T] >= 0)
    for s in 1:S
        @constraint(model, sum(γ[s, t] for t in 1:T) == p[s])
    end
    @constraint(model, sum(γ[s, t] * C[s, t] for s in 1:S, t in 1:T) <= Float64(ε))
    @objective(model, Min, sum(γ[s, t] * f[t] for s in 1:S, t in 1:T))
    JuMP.optimize!(model)
    st = JuMP.termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || throw(SolverError(string(st), "wasserstein_inner_lp solve failed"))
    return JuMP.objective_value(model)
end


