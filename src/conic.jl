function _conic_model(solver::Symbol)
    solver in (:clarabel, :scs) || throw(ArgumentError("solver must be :clarabel or :scs, got :$solver"))
    model = JuMP.Model(solver === :clarabel ? Clarabel.Optimizer : SCS.Optimizer)
    JuMP.set_silent(model)
    return model
end

function _tight_clarabel_model()
    model = _conic_model(:clarabel)
    JuMP.set_optimizer_attribute(model, "tol_gap_abs", DIRECT_TOL)
    JuMP.set_optimizer_attribute(model, "tol_gap_rel", DIRECT_TOL)
    JuMP.set_optimizer_attribute(model, "tol_feas", DIRECT_TOL)
    JuMP.set_optimizer_attribute(model, "max_iter", DIRECT_MAX_ITER)
    return model
end

function _check_optimal(model, msg::AbstractString)
    st = JuMP.termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || throw(SolverError(string(st), msg))
    return st
end

function _simplex_value(a)
    a_val = max.(JuMP.value.(a), 0.0)
    return a_val ./ sum(a_val)
end

"""
    solve_conic(m::ScenarioModel; solver = :clarabel, upper = nothing) -> Vector{Float64}

Solve the exponential-cone program for the log-optimal allocation (unpolished, not certified). `upper` gives per-asset caps a_j ≤ upper[j]
(one entry per column of `m.X`; `Inf` or any entry ≥ 1 means uncapped; no cap on cash); the caps are added as constraints and the result is projected
onto the capped simplex. Without cash and with |Σ u_j − 1| ≤ 1e-12 the feasible set is a single point and an `ArgumentError` directs the caller to
`optimal_allocation` (ADR-016).
"""
function solve_conic(m::ScenarioModel; solver::Symbol = :clarabel, upper = nothing)
    u = _normalize_upper(m, upper)
    if u !== nothing && !m.has_cash && abs(sum(u) - 1) ≤ BOX_FEAS_TOL
        throw(ArgumentError("near-point caps (|Σu − 1| ≤ 1e-12): use optimal_allocation (ADR-016)"))
    end
    J = size(m.X, 2)
    pos = findall(>(0.0), m.π)                       # scenarios with π_s = 0 are excluded
    model = _conic_model(solver)
    @variable(model, a[1:J] >= 0)
    @variable(model, t[1:length(pos)])
    @constraint(model, sum(a) == 1)
    if u !== nothing
        for j in 1:J
            isfinite(u[j]) && @constraint(model, a[j] <= u[j])
        end
    end
    for (k, s) in enumerate(pos)
        @constraint(model, [t[k], 1.0, sum(m.X[s, j] * a[j] for j in 1:J)] in MOI.ExponentialCone())  # t_s ≤ log V_s(a)
    end
    @objective(model, Max, sum(m.π[s] * t[k] for (k, s) in enumerate(pos)))
    JuMP.optimize!(model)
    _check_optimal(model, "conic solve failed (:$solver)")
    u === nothing && return _simplex_value(a)
    return _project_capped(max.(JuMP.value.(a), 0.0), u)
end

