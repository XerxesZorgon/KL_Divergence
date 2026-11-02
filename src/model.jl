function require_nonneg_finite(v, label)
    i = findfirst(x -> !(isfinite(x) && x >= 0.0), v)
    i === nothing || throw(ArgumentError("$label[$i] = $(v[i]) must be finite and ≥ 0 at index $i"))
end

function require_pos_finite(v, label)
    i = findfirst(x -> !(isfinite(x) && x > 0.0), v)
    i === nothing || throw(ArgumentError("$label[$i] = $(v[i]) must be finite and > 0 at index $i"))
end

struct ScenarioModel
    π::Vector{Float64}
    X::Matrix{Float64}
    assets::Vector{String}
    has_cash::Bool
    structure::Symbol

    function ScenarioModel(π::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
                           assets::AbstractVector{<:AbstractString}, has_cash::Bool, structure::Symbol)
        S, J = size(X)
        (S == length(π) && S >= 1) || throw(ArgumentError("rule 1 (dimensions): size(X,1)=$S != length(π)=$(length(π)) or < 1 (index 1)"))
        (J == length(assets) && J >= 1) || throw(ArgumentError("rule 1 (dimensions): size(X,2)=$J != length(assets)=$(length(assets)) or < 1 (index 1)"))
        assets_vec = String[String(a) for a in assets]
        dup = findfirst(j -> assets_vec[j] in @view(assets_vec[1:j-1]), 2:J)
        dup === nothing || throw(ArgumentError("rule 1 (unique assets): duplicate asset \"$(assets_vec[dup + 1])\" at index $(dup + 1)"))
        π_vec = Vector{Float64}(π)
        require_nonneg_finite(π_vec, "rule 2 (probability non-negative): invalid π")
        s_π = sum(π_vec)
        (isfinite(s_π) && abs(s_π - 1.0) <= PROB_TOL) || throw(ArgumentError("rule 2 (probability sum): |sum(π) - 1| = $(abs(s_π - 1.0)) > $PROB_TOL (index 1)"))
        X_mat = Matrix{Float64}(X)
        bad_x = findfirst(val -> !(isfinite(val) && val >= 0.0), X_mat)
        bad_x === nothing || throw(ArgumentError("rule 3 (returns finite and ≥ 0): invalid X[$(bad_x[1]), $(bad_x[2])] = $(X_mat[bad_x]) at index $(Tuple(bad_x))"))
        if has_cash
            assets_vec[1] == "cash" || throw(ArgumentError("rule 4 (cash asset): assets[1] = \"$(assets_vec[1])\" != \"cash\" at index 1"))
            bad_c = findfirst(!=(1.0), @view X_mat[:, 1])
            bad_c === nothing || throw(ArgumentError("rule 4 (cash return): X[$bad_c, 1] = $(X_mat[bad_c, 1]) != 1.0 at index ($bad_c, 1)"))
        end
        structure in VALID_STRUCTURES || throw(ArgumentError("rule 5 (structure): invalid structure = :$structure (index 1)"))
        bad_s = findfirst(s -> π_vec[s] > 0.0 && !any(>(0.0), @view X_mat[s, :]), 1:S)
        bad_s === nothing || throw(DomainError(bad_s, "rule 6 (feasibility): scenario row $bad_s with π[$bad_s] = $(π_vec[bad_s]) > 0 has no positive return entry"))
        return new(π_vec, X_mat, assets_vec, has_cash, structure)
    end
end

function single_event(q::AbstractVector{<:Real}, ohat::AbstractVector{<:Real}; cash::Bool = true)
    n = length(q)
    (n == length(ohat) && n >= 1) || throw(ArgumentError("single_event: length(q)=$n != length(ohat)=$(length(ohat)) or < 1 (index 1)"))
    require_pos_finite(ohat, "single_event: ohat")
    X = cash ? hcat(ones(n), Matrix(Diagonal(ohat))) : Matrix(Diagonal(ohat))
    outcome_names = ["outcome_$i" for i in 1:n]
    return ScenarioModel(q, X; assets = cash ? vcat(["cash"], outcome_names) : outcome_names, has_cash = cash, structure = :single_event)
end

function simultaneous_bets(p::AbstractVector{<:Real}, w::AbstractVector{<:Real}; cash::Bool = true)
    n = length(p)
    n == length(w) || throw(ArgumentError("simultaneous_bets: length(p)=$n != length(w)=$(length(w)) (index 1)"))
    1 <= n <= 16 || throw(ArgumentError("simultaneous_bets: n = $n: n > 16 gives more than 65 536 scenarios"))
    bad_p = findfirst(x -> !(isfinite(x) && 0.0 <= x <= 1.0), p)
    bad_p === nothing || throw(ArgumentError("simultaneous_bets: p[$bad_p] = $(p[bad_p]) must be in [0, 1] at index $bad_p"))
    require_pos_finite(w, "simultaneous_bets: w")
    S = 1 << n
    wins(s, k) = ((s - 1) >> (k - 1)) & 1 == 1
    π = [prod(wins(s, k) ? Float64(p[k]) : 1.0 - Float64(p[k]) for k in 1:n) for s in 1:S]
    X_bets = [wins(s, k) ? Float64(w[k]) : 0.0 for s in 1:S, k in 1:n]
    X = cash ? hcat(ones(S), X_bets) : X_bets
    bet_names = ["bet_$k" for k in 1:n]
    return ScenarioModel(π, X; assets = cash ? vcat(["cash"], bet_names) : bet_names, has_cash = cash, structure = :simultaneous)
end

function correlated_bets(P_joint::AbstractVector{<:Real}, W::AbstractMatrix{<:Real})
    S, n = size(W)
    return ScenarioModel(P_joint, hcat(ones(S), W); assets = vcat(["cash"], ["bet_$k" for k in 1:n]), has_cash = true, structure = :correlated)
end

function payoffs_from_odds(o::AbstractVector{<:Real}; commission::Real = 0.0)
    length(o) >= 1 || throw(ArgumentError("payoffs_from_odds: empty odds vector (index 1)"))
    require_pos_finite(o, "payoffs_from_odds: o")
    (isfinite(commission) && 0.0 <= commission < 1.0) || throw(ArgumentError("payoffs_from_odds: commission = $commission must be in [0, 1)"))
    return 1.0 .+ (o .- 1.0) .* (1 - commission)
end

growth(X, a, π) = sum(π .* log.(X * a))
growth(m::ScenarioModel, a) = growth(m.X, a, m.π)

