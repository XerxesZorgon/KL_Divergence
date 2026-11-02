function solve_conic(m::ScenarioModel; solver::Symbol = :clarabel)
    solver in (:clarabel, :scs) || throw(ArgumentError("solver must be :clarabel or :scs, got :$solver"))
    S, J = size(m.X)
    pos = findall(>(0.0), m.π)                       # scenarios with π_s = 0 are excluded
    model = JuMP.Model(solver === :clarabel ? Clarabel.Optimizer : SCS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, a[1:J] >= 0)
    @variable(model, t[1:length(pos)])
    @constraint(model, sum(a) == 1)
    for (k, s) in enumerate(pos)
        @constraint(model, [t[k], 1.0, sum(m.X[s, j] * a[j] for j in 1:J)] in MOI.ExponentialCone())  # t_s ≤ log V_s(a)
    end
    @objective(model, Max, sum(m.π[s] * t[k] for (k, s) in enumerate(pos)))
    JuMP.optimize!(model)
    st = JuMP.termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || throw(SolverError(string(st), "conic solve failed (:$solver)"))
    a_val = max.(JuMP.value.(a), 0.0)
    return a_val ./ sum(a_val)
end

