function single_event_inputs(m::ScenarioModel)
    bets = m.has_cash ? (2:size(m.X, 2)) : (1:size(m.X, 2))
    s_of_bet = Vector{Int}(undef, length(bets))
    for (k, j) in enumerate(bets)
        pos_rows = findall(>(0.0), @view m.X[:, j])
        length(pos_rows) == 1 || throw(ArgumentError("single_event shape: column $j (\"$(m.assets[j])\") has $(length(pos_rows)) positive entries, expected 1"))
        sj = pos_rows[1]
        sj in @view(s_of_bet[1:k-1]) && throw(ArgumentError("single_event shape: column $j (\"$(m.assets[j])\") pays in scenario $sj, which is already used by another bet column"))
        s_of_bet[k] = sj
    end
    return [m.π[s] for s in s_of_bet], [m.X[s_of_bet[k], bets[k]] for k in eachindex(bets)]
end

"""
    optimal_allocation(m::ScenarioModel; method::Symbol = :auto, solver::Symbol = :clarabel, tol = CERT_TOL) -> BetResult
    optimal_allocation(q, ohat; cash::Bool = true, method::Symbol = :auto, solver::Symbol = :clarabel, tol = CERT_TOL) -> BetResult

Compute and certify the log-optimal allocation for `m`; every returned bet is certified (KKT ≤ tol).
Dispatch rules:
- `method = :auto` (default) on `:single_event` uses the exact closed-form paths (`:greedy` with cash, `:full_investment` without cash; `solver` is ignored).
- Any other `m.structure`, or `method = :conic`, solves the exponential-cone program via `solver ∈ (:clarabel, :scs)`, polishes on the active set, and certifies with `result.method == solver`.
"""
function optimal_allocation(m::ScenarioModel; method::Symbol = :auto, solver::Symbol = :clarabel, tol = CERT_TOL)
    method in (:auto, :conic) || throw(ArgumentError("method must be :auto or :conic, got :$method"))
    if method === :auto && m.structure === :single_event
        q, ohat = single_event_inputs(m)
        if m.has_cash
            return certify(BetResult(m, greedy_allocation(q, ohat), :greedy); tol = tol)
        end
        length(q) == length(m.π) || throw(ArgumentError("single_event without cash requires length(bets)=$(length(q)) == length(π)=$(length(m.π)) (column 1)"))
        return certify(BetResult(m, full_investment_allocation(q), :full_investment); tol = tol)
    end
    a_raw = solve_conic(m; solver = solver)
    a, _ = polish_log_optimal(m.X, m.π, a_raw; tol = tol)
    return certify(BetResult(m, a, solver); tol = tol)
end

optimal_allocation(q, ohat; cash::Bool = true, method::Symbol = :auto, solver::Symbol = :clarabel, tol = CERT_TOL) =
    optimal_allocation(single_event(q, ohat; cash = cash); method = method, solver = solver, tol = tol)

