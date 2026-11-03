"""
    beta_predictive(α, β)

Posterior predictive win probabilities `α ./ (α .+ β)` for independent binary events
with Beta(α_k, β_k) posteriors. For independent binary events the Bayes-optimal bet
uses exactly these posterior means — prior width alone does not change the bet (DESIGN §5.7).
"""
function beta_predictive(α::AbstractVector{<:Real}, β::AbstractVector{<:Real})
    n = length(α)
    n ≥ 1 && length(β) == n || throw(ArgumentError("α and β must have equal length ≥ 1"))
    (all(isfinite, α) && all(>(0), α) && all(isfinite, β) && all(>(0), β)) ||
        throw(ArgumentError("α and β must be finite and > 0"))
    αf = collect(Float64, α)
    βf = collect(Float64, β)
    return αf ./ (αf .+ βf)
end

function dirichlet_predictive(α::AbstractVector{<:Real})
    length(α) ≥ 1 || throw(ArgumentError("α must have length ≥ 1"))
    (all(isfinite, α) && all(>(0), α)) || throw(ArgumentError("α must be finite and > 0"))
    αf = collect(Float64, α)
    return αf ./ sum(αf)
end

function mixture(models::AbstractVector{ScenarioModel}, weights::AbstractVector{<:Real})
    M = length(models)
    M ≥ 1 && length(weights) == M || throw(ArgumentError("models and weights must have equal length ≥ 1"))
    m1 = models[1]
    for i in 2:M
        mi = models[i]
        (mi.X == m1.X && mi.assets == m1.assets && mi.has_cash == m1.has_cash) ||
            throw(ArgumentError("model at index $i differs in X, assets, or has_cash from models[1]"))
    end
    wf = collect(Float64, weights)
    (all(isfinite, wf) && all(≥(0.0), wf) && abs(sum(wf) - 1.0) ≤ PROB_TOL) ||
        throw(ArgumentError("weights must be finite, ≥ 0, and sum to 1 within $PROB_TOL"))
    π_mix = sum(wf[i] .* models[i].π for i in eachindex(models))
    return ScenarioModel(π_mix, m1.X; assets = m1.assets, has_cash = m1.has_cash, structure = :mixture)
end

function bma_weights(logml::AbstractVector{<:Real})
    length(logml) ≥ 1 || throw(ArgumentError("logml must have length ≥ 1"))
    all(isfinite, logml) || throw(ArgumentError("logml must be finite"))
    lf = collect(Float64, logml)
    m = maximum(lf)
    w = exp.(lf .- m)
    return w ./ sum(w)
end

