full_investment_allocation(q) = collect(Float64, q)

function greedy_allocation(q::AbstractVector{<:Real}, ohat::AbstractVector{<:Real})
    if length(q) != length(ohat) || length(q) < 1
        throw(ArgumentError("greedy_allocation: length(q)=$(length(q)) != length(ohat)=$(length(ohat)) or < 1 (index 1)"))
    end
    for (i, p) in enumerate(q)
        if !(isfinite(p) && p >= 0.0)
            throw(ArgumentError("greedy_allocation: q[$i] = $p must be finite and ≥ 0 at index $i"))
        end
    end
    sq = sum(q)
    if !(isfinite(sq) && sq <= 1.0 + 1e-12)
        throw(ArgumentError("greedy_allocation: sum(q) = $sq > 1 + 1e-12 (index 1)"))
    end
    for (i, v) in enumerate(ohat)
        if !(isfinite(v) && v > 0.0)
            throw(ArgumentError("greedy_allocation: ohat[$i] = $v must be finite and > 0 at index $i"))
        end
    end

    n = length(q)
    order = sortperm(q .* ohat; rev=true)
    S = Int[]; pS = 0.0; σS = 0.0; R = 1.0
    for i in order
        q[i] * ohat[i] > R || break
        push!(S, i); pS += q[i]; σS += 1 / ohat[i]
        R = (pS >= 1 - 1e-15 || σS >= 1) ? 0.0 : (1 - pS) / (1 - σS)
    end
    f = zeros(n)
    for i in S
        f[i] = max(q[i] - R / ohat[i], 0.0)
    end
    return vcat(max(1 - sum(f), 0.0), f)
end

