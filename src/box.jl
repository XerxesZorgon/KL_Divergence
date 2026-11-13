# Per-asset upper caps (box constraints): validation and normalization (Task 064).

const BOX_FEAS_TOL = 1e-12

"""
    _normalize_upper(m::ScenarioModel, upper) -> Union{Nothing, Vector{Float64}}

Validate per-asset caps `upper` for `m` and normalize them: caps ≥ 1 become `Inf`; returns `nothing` when
`upper === nothing` or every cap is `Inf`. Internal; not exported.
"""
function _normalize_upper(m::ScenarioModel, upper)::Union{Nothing, Vector{Float64}}
    upper === nothing && return nothing
    upper isa AbstractVector{<:Real} || throw(ArgumentError("upper must be a vector of reals, got $(typeof(upper))"))
    J = size(m.X, 2)
    length(upper) == J || throw(ArgumentError("length(upper) = $(length(upper)) but the model has $J assets"))
    for (j, v) in enumerate(upper)
        (isnan(v) || v ≤ 0) && throw(ArgumentError("upper[$j] = $v; caps must be > 0"))
    end
    u = Vector{Float64}(upper)
    u[u .≥ 1] .= Inf
    m.has_cash && isfinite(u[1]) && throw(ArgumentError("caps on cash are not supported (upper[1] = $(u[1]))"))
    if !m.has_cash && sum(u) < 1 - BOX_FEAS_TOL
        throw(ArgumentError("infeasible caps: Σ u_j = $(sum(u)) < 1 with no cash"))
    end
    all(isinf, u) && return nothing
    return u
end


"""
    _project_capped(x::AbstractVector{<:Real}, u::Vector{Float64}) -> Vector{Float64}

Euclidean projection of `x` onto A_u = {a : 0 ≤ a_j ≤ u_j, Σ a_j = 1} (`Inf` = uncapped). Precondition: `u` comes from
`_normalize_upper` (not `nothing`) and `sum(u) ≥ 1`; not re-validated. Internal; not exported.

The result is `clamp.(x .- θ, 0, u)` with θ the root of the non-increasing piecewise-linear `S(θ) = Σ clamp(x_j - θ, 0, u_j)`
`= 1`, found exactly by evaluating `S` at the sorted breakpoints and interpolating linearly inside the bracketing interval.
"""
function _project_capped(x::AbstractVector{<:Real}, u::Vector{Float64})::Vector{Float64}
    xf = Vector{Float64}(x)
    S(θ) = sum(clamp.(xf .- θ, 0.0, u))
    bp = sort!(vcat(xf, [xf[j] - u[j] for j in eachindex(u) if isfinite(u[j])]))
    s = S.(bp)
    θ = if s[1] < 1
        # root lies left of every breakpoint, where only uncapped assets move (slope -n_inf)
        bp[1] - (1 - s[1]) / count(isinf, u)
    else
        k = findlast(≥(1), s)               # s non-increasing; s[end] = 0 < 1 so k < length(bp)
        s[k] == 1 ? bp[k] : bp[k] + (s[k] - 1) * (bp[k + 1] - bp[k]) / (s[k] - s[k + 1])
    end
    a = clamp.(xf .- θ, 0.0, u)
    F = findall(j -> 0 < a[j] < u[j], eachindex(a))
    isempty(F) || (a[F] .+= (1 - sum(a)) / length(F))
    return a
end

# Near-point caps (ADR-016): no cash and |Σu − 1| ≤ BOX_FEAS_TOL, so A_u is empty or a single point. `u` is normalized or `nothing`.
_near_point(m::ScenarioModel, u) = u !== nothing && !m.has_cash && abs(sum(u) - 1) ≤ BOX_FEAS_TOL
