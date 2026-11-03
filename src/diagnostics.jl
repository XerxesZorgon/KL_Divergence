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

function growth_loss(m_true::ScenarioModel, a_belief::AbstractVector{<:Real})
    r = optimal_allocation(m_true)
    return r.growth - growth(m_true, a_belief)
end

function _validate_logscore_prob(π::AbstractVector{<:Real}, name::AbstractString)
    length(π) ≥ 1 || throw(ArgumentError("$name: π must be non-empty"))
    require_nonneg_finite(π, name)
    abs(sum(π) - 1.0) ≤ PROB_TOL || throw(ArgumentError("$name: probabilities must sum to 1 within $PROB_TOL"))
end

function _validate_logscore_inputs(π::AbstractVector{<:Real}, y::AbstractVector{<:Integer}, name::AbstractString)
    _validate_logscore_prob(π, "$name: π")
    K = length(π)
    length(y) ≥ 1 || throw(ArgumentError("$name: y must have length ≥ 1"))
    all(yi -> 1 ≤ yi ≤ K, y) || throw(ArgumentError("$name: outcome indices y must be in 1:$K"))
end

function holdout_logscore(π::AbstractVector{<:Real}, y::AbstractVector{<:Integer})
    _validate_logscore_inputs(π, y, "holdout_logscore")
    v = [log(Float64(π[yi])) for yi in y]
    n = length(v)
    return (mean = mean(v), se = n ≥ 2 ? std(v) / sqrt(n) : NaN)
end

"""
    logscore_difference(π1, π2, y)

Paired mean and standard error of `log(π1[y_i]) - log(π2[y_i])` over held-out outcome indices `y`.
Its expectation over outcomes drawn from the truth `p` is `D(p‖π2) − D(p‖π1)` — positive when `π1`
is closer to the truth (the unknown entropy cancels).
"""
function logscore_difference(π1::AbstractVector{<:Real}, π2::AbstractVector{<:Real}, y::AbstractVector{<:Integer})
    length(π1) == length(π2) || throw(ArgumentError("logscore_difference: π1 and π2 must have equal length"))
    _validate_logscore_inputs(π1, y, "logscore_difference")
    _validate_logscore_prob(π2, "logscore_difference: π2")
    v = [log(Float64(π1[yi])) - log(Float64(π2[yi])) for yi in y]
    n = length(v)
    return (mean = mean(v), se = n ≥ 2 ? std(v) / sqrt(n) : NaN)
end

function growth_curve(m::ScenarioModel, a_star::AbstractVector{<:Real}, cs::AbstractVector{<:Real})
    m.has_cash || throw(ArgumentError("growth_curve requires m.has_cash == true"))
    length(a_star) == size(m.X, 2) || throw(ArgumentError("growth_curve: length(a_star) != size(m.X, 2)"))
    b = Float64.(a_star[2:end])
    sb = sum(b)
    pos = findall(>(0.0), m.π)
    Xp = m.X[pos, :]
    πp = m.π[pos]
    out = Vector{Float64}(undef, length(cs))
    for (idx, c_raw) in enumerate(cs)
        c = Float64(c_raw)
        a_c = vcat(1.0 - c * sb, c .* b)
        Vp = Xp * a_c
        out[idx] = any(≤(0.0), Vp) ? -Inf : sum(πp .* log.(Vp))
    end
    return out
end

"""
    zero_growth_point(m::ScenarioModel, a_star)

Find `c0 > 1` where scaling the Kelly stakes `a_star[2:end]` by `c0` gives zero long-run growth
`G(c0) = 0`; beyond `c0` wealth tends to 0. Returns `(c0 = c0, c_quad = 2.0)`.
"""
function zero_growth_point(m::ScenarioModel, a_star::AbstractVector{<:Real})
    G(c) = growth_curve(m, a_star, [c])[1]
    G(1.0) > 0.0 || throw(ArgumentError("no positive growth at c = 1"))
    lo = 1.0
    hi = 2.0
    while G(hi) > 0.0
        lo = hi
        hi *= 2.0
        hi > 1e6 && throw(ArgumentError("zero_growth_point bracket exceeded 1e6"))
    end
    while hi - lo > 1e-14 * hi
        mid = 0.5 * (lo + hi)
        if G(mid) > 0.0
            lo = mid
        else
            hi = mid
        end
    end
    return (c0 = 0.5 * (lo + hi), c_quad = 2.0)
end

