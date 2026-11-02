"""
    kl(p, q) -> Float64

Kullback–Leibler divergence D(p‖q) = Σ_i p_i log(p_i / q_i) in nats (0·log 0 = 0; +Inf when p_i > 0 = q_i).
Remaining diagnostics arrive in M4.
"""
function kl(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    (length(p) == length(q) && length(p) >= 1) || throw(ArgumentError("kl: length(p)=$(length(p)) != length(q)=$(length(q)) or < 1 (index 1)"))
    require_nonneg_finite(p, "kl: p")
    require_nonneg_finite(q, "kl: q")
    return sum(p[i] > 0 ? p[i] * log(p[i] / q[i]) : 0.0 for i in eachindex(p))
end

