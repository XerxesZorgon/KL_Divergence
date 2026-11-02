"""
    kl(p, q) -> Float64

Kullback–Leibler divergence D(p‖q) = Σ_i p_i log(p_i / q_i) in nats (0·log 0 = 0; +Inf when p_i > 0 = q_i).
Remaining diagnostics arrive in M4.
"""
function kl(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    if length(p) != length(q) || length(p) < 1
        throw(ArgumentError("kl: length(p)=$(length(p)) != length(q)=$(length(q)) or < 1 (index 1)"))
    end
    for (i, v) in enumerate(p)
        if !(isfinite(v) && v >= 0.0)
            throw(ArgumentError("kl: p[$i] = $v must be finite and ≥ 0 at index $i"))
        end
    end
    for (i, v) in enumerate(q)
        if !(isfinite(v) && v >= 0.0)
            throw(ArgumentError("kl: q[$i] = $v must be finite and ≥ 0 at index $i"))
        end
    end
    return sum(p[i] > 0 ? p[i] * log(p[i] / q[i]) : 0.0 for i in eachindex(p))
end
