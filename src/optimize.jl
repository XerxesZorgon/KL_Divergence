"""
    optimal_allocation(m::ScenarioModel; solver = :clarabel, tol = 1e-10) -> BetResult
    optimal_allocation(q, ohat; cash::Bool = true, tol = 1e-10) -> BetResult

Compute and certify the log-optimal allocation for `m`.
For `:single_event` models, exact closed-form paths (`:greedy` with cash, `:full_investment` without cash)
are used; `solver` is accepted and ignored on these exact paths.
"""
function optimal_allocation(m::ScenarioModel; solver = :clarabel, tol = 1e-10)
    if m.structure != :single_event
        throw(ArgumentError("structure :$(m.structure) requires the conic path (milestone M2)"))
    end

    S, J = size(m.X)
    bets = m.has_cash ? collect(2:J) : collect(1:J)
    s_of_bet = Vector{Int}(undef, length(bets))
    seen_s = Set{Int}()

    for (k, j) in enumerate(bets)
        pos_rows = findall(>(0.0), @view m.X[:, j])
        if length(pos_rows) != 1
            throw(ArgumentError("single_event shape: column $j (\"$(m.assets[j])\") has $(length(pos_rows)) positive entries, expected 1"))
        end
        sj = pos_rows[1]
        if sj in seen_s
            throw(ArgumentError("single_event shape: column $j (\"$(m.assets[j])\") pays in scenario $sj, which is already used by another bet column"))
        end
        push!(seen_s, sj)
        s_of_bet[k] = sj
    end

    q = [m.π[s_of_bet[k]] for k in eachindex(bets)]
    ohat = [m.X[s_of_bet[k], bets[k]] for k in eachindex(bets)]

    if m.has_cash
        a = greedy_allocation(q, ohat)
        method = :greedy
    else
        if length(bets) != length(m.π)
            throw(ArgumentError("single_event without cash requires length(bets)=$(length(bets)) == length(π)=$S (column 1)"))
        end
        a = full_investment_allocation(q)
        method = :full_investment
    end

    return certify(BetResult(m, a, method); tol = tol)
end

optimal_allocation(q, ohat; cash::Bool = true, tol = 1e-10) =
    optimal_allocation(single_event(q, ohat; cash = cash); tol = tol)
