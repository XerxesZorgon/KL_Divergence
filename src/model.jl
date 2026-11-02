struct ScenarioModel
    π::Vector{Float64}
    X::Matrix{Float64}
    assets::Vector{String}
    has_cash::Bool
    structure::Symbol

    function ScenarioModel(π::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
                           assets::AbstractVector{<:AbstractString},
                           has_cash::Bool,
                           structure::Symbol)
        S, J = size(X)
        # Rule 1: dimensions and unique assets
        if S != length(π) || S < 1
            throw(ArgumentError("rule 1 (dimensions): size(X,1)=$S != length(π)=$(length(π)) or < 1 (index 1)"))
        end
        if J != length(assets) || J < 1
            throw(ArgumentError("rule 1 (dimensions): size(X,2)=$J != length(assets)=$(length(assets)) or < 1 (index 1)"))
        end
        seen = Set{String}()
        for (j, name) in enumerate(assets)
            s = String(name)
            if s in seen
                throw(ArgumentError("rule 1 (unique assets): duplicate asset \"$s\" at index $j"))
            end
            push!(seen, s)
        end

        # Rule 2: π_s ≥ 0 and |sum(π) - 1| ≤ 1e-12 (stored as given, no renormalization)
        π_vec = Vector{Float64}(π)
        for (s, p) in enumerate(π_vec)
            if !(isfinite(p) && p >= 0.0)
                throw(ArgumentError("rule 2 (probability non-negative): invalid π[$s] = $p at index $s"))
            end
        end
        s_π = sum(π_vec)
        if !(isfinite(s_π) && abs(s_π - 1.0) <= 1e-12)
            throw(ArgumentError("rule 2 (probability sum): |sum(π) - 1| = $(abs(s_π - 1.0)) > 1e-12 (index 1)"))
        end

        # Rule 3: all X entries finite and ≥ 0
        X_mat = Matrix{Float64}(X)
        for j in 1:J, s in 1:S
            val = X_mat[s, j]
            if !(isfinite(val) && val >= 0.0)
                throw(ArgumentError("rule 3 (returns finite and ≥ 0): invalid X[$s, $j] = $val at index ($s, $j)"))
            end
        end

        # Rule 4: has_cash ⇒ X[:,1] == ones(S) and assets[1] == "cash"
        assets_vec = String[String(a) for a in assets]
        if has_cash
            if assets_vec[1] != "cash"
                throw(ArgumentError("rule 4 (cash asset): assets[1] = \"$(assets_vec[1])\" != \"cash\" at index 1"))
            end
            for s in 1:S
                if X_mat[s, 1] != 1.0
                    throw(ArgumentError("rule 4 (cash return): X[$s, 1] = $(X_mat[s, 1]) != 1.0 at index ($s, 1)"))
                end
            end
        end

        # Rule 5: structure ∈ (:single_event, :simultaneous, :correlated, :sampled, :mixture)
        if !(structure in (:single_event, :simultaneous, :correlated, :sampled, :mixture))
            throw(ArgumentError("rule 5 (structure): invalid structure = :$structure (index 1)"))
        end

        # Rule 6: feasibility: every row s with π_s > 0 has at least one positive entry
        for s in 1:S
            if π_vec[s] > 0.0
                has_pos = false
                for j in 1:J
                    if X_mat[s, j] > 0.0
                        has_pos = true
                        break
                    end
                end
                if !has_pos
                    throw(DomainError(s, "rule 6 (feasibility): scenario row $s with π[$s] = $(π_vec[s]) > 0 has no positive return entry"))
                end
            end
        end

        return new(π_vec, X_mat, assets_vec, has_cash, structure)
    end
end

function single_event(q::AbstractVector{<:Real}, ohat::AbstractVector{<:Real}; cash::Bool = true)
    n = length(q)
    if n != length(ohat) || n < 1
        throw(ArgumentError("single_event: length(q)=$n != length(ohat)=$(length(ohat)) or < 1 (index 1)"))
    end
    for (i, v) in enumerate(ohat)
        if !(isfinite(v) && v > 0.0)
            throw(ArgumentError("single_event: ohat[$i] = $v must be finite and > 0 at index $i"))
        end
    end
    X = cash ? hcat(ones(n), Matrix(Diagonal(ohat))) : Matrix(Diagonal(ohat))
    outcome_names = ["outcome_$i" for i in 1:n]
    assets = cash ? vcat(["cash"], outcome_names) : outcome_names
    return ScenarioModel(q, X; assets = assets, has_cash = cash, structure = :single_event)
end

function payoffs_from_odds(o::AbstractVector{<:Real}; commission::Real = 0.0)
    if length(o) < 1
        throw(ArgumentError("payoffs_from_odds: empty odds vector (index 1)"))
    end
    for (i, v) in enumerate(o)
        if !(isfinite(v) && v > 0.0)
            throw(ArgumentError("payoffs_from_odds: o[$i] = $v must be finite and > 0 at index $i"))
        end
    end
    if !(isfinite(commission) && 0.0 <= commission < 1.0)
        throw(ArgumentError("payoffs_from_odds: commission = $commission must be in [0, 1)"))
    end
    return 1.0 .+ (o .- 1.0) .* (1 - commission)
end

growth(X, a, π) = sum(π .* log.(X * a))
growth(m::ScenarioModel, a) = growth(m.X, a, m.π)

