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

