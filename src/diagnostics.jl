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

"""
    drawdown_probability(x, c)

Continuous-limit probability that wealth ever falls to fraction `x` of its starting value
when betting `c` times the Kelly stakes (Thorp 2006 — to be checked in E4); a diagnostic,
never a sizing rule (ADR-011).
"""
function drawdown_probability(x::Real, c::Real)
    (isfinite(x) && 0.0 < x < 1.0) || throw(ArgumentError("drawdown_probability requires 0 < x < 1, got $x"))
    (isfinite(c) && c > 0.0) || throw(ArgumentError("drawdown_probability requires c > 0, got $c"))
    return c ≥ 2.0 ? 1.0 : Float64(x)^(2.0 / Float64(c) - 1.0)
end

function effective_multiple(e_belief::Real, e_true::Real)
    (isfinite(e_belief) && isfinite(e_true) && e_true > 0.0) ||
        throw(ArgumentError("effective_multiple requires finite e_belief and e_true > 0"))
    return Float64(e_belief) / Float64(e_true)
end

function ruin_probability_classical(p::Real, a::Integer, N::Integer)
    (isfinite(p) && 0.0 < p < 1.0) || throw(ArgumentError("ruin_probability_classical requires 0 < p < 1, got $p"))
    N ≥ 1 || throw(ArgumentError("ruin_probability_classical requires N ≥ 1, got $N"))
    0 ≤ a ≤ N || throw(ArgumentError("ruin_probability_classical requires 0 ≤ a ≤ N, got a = $a, N = $N"))
    pf = Float64(p)
    q = 1.0 - pf
    return pf == 0.5 ? 1.0 - a / N : ((q / pf)^a - (q / pf)^N) / (1.0 - (q / pf)^N)
end

function _validate_sample_size_inputs(p::Real, w::Real, name::AbstractString)
    (isfinite(p) && 0.0 < p < 1.0) || throw(ArgumentError("$name requires 0 < p < 1, got $p"))
    (isfinite(w) && w > 0.0) || throw(ArgumentError("$name requires w > 0, got $w"))
    e = Float64(p) * Float64(w) - 1.0
    e ≠ 0.0 || throw(ArgumentError("$name: zero edge"))
    return Float64(p), Float64(w), e
end

function n_detect(p::Real, w::Real; z::Real = 1.96)
    (isfinite(z) && z > 0.0) || throw(ArgumentError("n_detect requires z > 0"))
    pf, wf, e = _validate_sample_size_inputs(p, w, "n_detect")
    return Float64(z)^2 * pf * (1.0 - pf) * wf^2 / e^2
end

function n_relative(p::Real, w::Real; rel::Real = 0.10)
    (isfinite(rel) && rel > 0.0) || throw(ArgumentError("n_relative requires rel > 0"))
    pf, wf, e = _validate_sample_size_inputs(p, w, "n_relative")
    return pf * (1.0 - pf) * wf^2 / (Float64(rel) * e)^2
end

"""
    simulate_ruin(m::ScenarioModel, a; rng, trials, rounds, w0 = 1.0, min_stake = 0.0, lot = 0.0, fixed_cost = 0.0, fixed_stake = nothing, target = Inf)

Monte Carlo ruin simulator: proportional staking (`a` times current wealth on each bet; cash is never staked)
unless `fixed_stake` is given (then a fixed stake on the first bet asset — the classical gambler's-ruin game);
`lot` rounds stakes down; stakes below `min_stake` are skipped; `fixed_cost` is charged per placed bet;
ruin when wealth ≤ 0 or falls below `min_stake`; a trial stops at ruin, at `target`, or after `rounds`.
`drawdown_q` are quantiles of the maximum drawdown from the running peak; `min_fraction` holds each trial's
lowest wealth as a fraction of `w0` (for the drawdown law, E4).
"""
function simulate_ruin(m::ScenarioModel, a; rng, trials::Integer, rounds::Integer, w0 = 1.0, min_stake = 0.0, lot = 0.0,
                       fixed_cost = 0.0, fixed_stake = nothing, target = Inf)
    S, J = size(m.X)
    (trials ≥ 1 && rounds ≥ 1) || throw(ArgumentError("trials and rounds must be ≥ 1"))
    length(a) == J || throw(ArgumentError("length(a) = $(length(a)) != number of assets $J"))
    (w0 > 0 && min_stake ≥ 0 && lot ≥ 0 && fixed_cost ≥ 0 && target > w0) ||
        throw(ArgumentError("need w0 > 0, min_stake ≥ 0, lot ≥ 0, fixed_cost ≥ 0, target > w0"))
    (fixed_stake === nothing || fixed_stake > 0) || throw(ArgumentError("fixed_stake must be > 0"))
    bets = m.has_cash ? (2:J) : (1:J); cum = cumsum(m.π)
    ruined = 0; maxdd = zeros(trials); final = zeros(trials); minf = zeros(trials)
    for t in 1:trials
        W = w0; peak = w0; dd = 0.0; ruin = false; lo = w0
        for _ in 1:rounds
            s = min(searchsortedfirst(cum, rand(rng) * cum[end]), S)
            if fixed_stake === nothing
                stakes = [a[j] * W for j in bets]
                lot > 0 && (stakes = floor.(stakes ./ lot) .* lot)
            else
                stakes = zeros(length(bets)); stakes[1] = min(fixed_stake, W)
            end
            stakes[stakes .< min_stake] .= 0.0                     # bets below the minimum are skipped
            placed = count(>(0.0), stakes)
            W = W - sum(stakes) - fixed_cost * placed + sum(stakes[i] * m.X[s, bets[i]] for i in eachindex(bets))
            peak = max(peak, W); dd = max(dd, 1 - W / peak); lo = min(lo, W)
            if W ≤ 0 || (min_stake > 0 && W < min_stake); ruin = true; break; end
            W ≥ target && break
        end
        ruined += ruin; maxdd[t] = dd; final[t] = max(W, 0.0); minf[t] = max(lo, 0.0) / w0
    end
    p = ruined / trials
    return (p_ruin = p, se = sqrt(p * (1 - p) / trials), drawdown_q = quantile(maxdd, [0.5, 0.9, 0.99]),
            final_q = quantile(final, [0.1, 0.5, 0.9]), min_fraction = minf)
end

