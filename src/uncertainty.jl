"""
    beta_predictive(α, β)

Posterior predictive win probabilities `α ./ (α .+ β)` for independent binary events
with Beta(α_k, β_k) posteriors. For independent binary events the Bayes-optimal bet
uses exactly these posterior means — prior width alone does not change the bet (DESIGN §5.7).
"""
function beta_predictive(α::AbstractVector{<:Real}, β::AbstractVector{<:Real})
    n = length(α)
    n ≥ 1 && length(β) == n || throw(ArgumentError("α and β must have equal length ≥ 1"))
    require_pos_finite(α, "beta_predictive: α")
    require_pos_finite(β, "beta_predictive: β")
    αf = collect(Float64, α)
    return αf ./ (αf .+ collect(Float64, β))
end

function dirichlet_predictive(α::AbstractVector{<:Real})
    length(α) ≥ 1 || throw(ArgumentError("α must have length ≥ 1"))
    require_pos_finite(α, "dirichlet_predictive: α")
    αf = collect(Float64, α)
    return αf ./ sum(αf)
end

"""
    predictive_from_draws(Pdraws, w; cash = true)

Joint posterior predictive model from an `M × n` matrix `Pdraws` of win-probability draws
and gross payoffs `w`. It differs from `simultaneous_bets(beta_predictive(…), w)` only through
dependence between the bets' probabilities — prior width alone does not move the bet.
"""
function predictive_from_draws(Pdraws::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; cash::Bool = true)
    M, n = size(Pdraws)
    M ≥ 1 && n == length(w) || throw(ArgumentError("Pdraws must have M ≥ 1 rows and length(w) columns"))
    1 ≤ n ≤ MAX_SIMULTANEOUS_BETS || throw(ArgumentError("n > $MAX_SIMULTANEOUS_BETS gives more than 65 536 scenarios"))
    require_prob_01(Pdraws, "predictive_from_draws: Pdraws")
    Pf = Matrix{Float64}(Pdraws)
    base = simultaneous_bets(fill(0.5, n), w; cash)
    π = [sum(prod(wins_bit(s, k) ? Pf[m, k] : 1.0 - Pf[m, k] for k in 1:n) for m in 1:M) / M for s in 1:(1 << n)]
    return ScenarioModel(π, base.X; assets = base.assets, has_cash = cash, structure = :mixture)
end

function sampled_payoffs(p::AbstractVector{<:Real}, samplers::AbstractVector, M::Integer; rng, cash::Bool = true)
    n = length(p)
    length(samplers) == n || throw(ArgumentError("p and samplers must have equal length"))
    1 ≤ n ≤ MAX_SIMULTANEOUS_BETS || throw(ArgumentError("1 ≤ n ≤ $MAX_SIMULTANEOUS_BETS required, got n = $n"))
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1, got $M"))
    B = 1 << n; S = M * B
    S ≤ MAX_SAMPLED_SCENARIOS || throw(ArgumentError("M * 2^n = $S exceeds $MAX_SAMPLED_SCENARIOS"))
    require_prob_01(p, "sampled_payoffs: p")
    pf = collect(Float64, p)
    W = Matrix{Float64}(undef, M, n)
    for m in 1:M, k in 1:n
        sk = samplers[k]
        val = sk isa Distributions.Sampleable ? rand(rng, sk) : sk(rng)
        (isfinite(val) && val > 0) || throw(ArgumentError("sampled payoff at m = $m, k = $k must be finite and > 0, got $val"))
        W[m, k] = Float64(val)
    end
    π_block = [prod(wins_bit(s, k) ? pf[k] : 1.0 - pf[k] for k in 1:n) / M for s in 1:B]
    X_bets = [wins_bit(mod1(r, B), k) ? W[cld(r, B), k] : 0.0 for r in 1:S, k in 1:n]
    X = cash ? hcat(ones(Float64, S), X_bets) : X_bets
    return ScenarioModel(repeat(π_block, M), X; assets = bet_assets(n, cash), has_cash = cash, structure = :sampled)
end

function growth_se(m::ScenarioModel, a::AbstractVector{<:Real})
    m.structure === :sampled || throw(ArgumentError("growth_se requires m.structure == :sampled, got :$(m.structure)"))
    n = size(m.X, 2) - m.has_cash
    B = 1 << n
    M = size(m.X, 1) ÷ B
    M == 1 && return 0.0
    V = m.X * a
    g = [M * sum(m.π[r] * log(V[r]) for r in (i - 1) * B + 1 : i * B) for i in 1:M]
    return std(g) / sqrt(M)
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

