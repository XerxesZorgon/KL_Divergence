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

