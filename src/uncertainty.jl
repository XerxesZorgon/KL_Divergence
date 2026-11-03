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

"""
    predictive_from_draws(Pdraws, w; cash = true)

Joint posterior predictive model from an `M × n` matrix `Pdraws` of win-probability draws
and gross payoffs `w`. It differs from `simultaneous_bets(beta_predictive(…), w)` only through
dependence between the bets' probabilities — prior width alone does not move the bet.
"""
function predictive_from_draws(Pdraws::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; cash::Bool = true)
    M, n = size(Pdraws)
    M ≥ 1 && n == length(w) || throw(ArgumentError("Pdraws must have M ≥ 1 rows and length(w) columns"))
    1 ≤ n ≤ 16 || throw(ArgumentError("n > 16 gives more than 65 536 scenarios"))
    all(x -> isfinite(x) && 0.0 ≤ x ≤ 1.0, Pdraws) ||
        throw(ArgumentError("Pdraws entries must be finite and in [0, 1]"))
    Pf = Matrix{Float64}(Pdraws)
    base = simultaneous_bets(fill(0.5, n), w; cash)
    S = 1 << n
    wins(s, k) = ((s - 1) >> (k - 1)) & 1 == 1
    π = [sum(prod(wins(s, k) ? Pf[m, k] : 1.0 - Pf[m, k] for k in 1:n) for m in 1:M) / M for s in 1:S]
    return ScenarioModel(π, base.X; assets = base.assets, has_cash = cash, structure = :mixture)
end

function sampled_payoffs(p::AbstractVector{<:Real}, samplers::AbstractVector, M::Integer; rng, cash::Bool = true)
    n = length(p)
    length(samplers) == n || throw(ArgumentError("p and samplers must have equal length"))
    1 ≤ n ≤ 16 || throw(ArgumentError("1 ≤ n ≤ 16 required, got n = $n"))
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1, got $M"))
    B = 1 << n
    S = M * B
    S ≤ 1_048_576 || throw(ArgumentError("M * 2^n = $S exceeds 1_048_576"))
    all(x -> isfinite(x) && 0.0 ≤ x ≤ 1.0, p) ||
        throw(ArgumentError("win probabilities p must be finite and in [0, 1]"))
    pf = collect(Float64, p)
    W = Matrix{Float64}(undef, M, n)
    for m in 1:M, k in 1:n
        sk = samplers[k]
        val = sk isa Distributions.Sampleable ? rand(rng, sk) : sk(rng)
        (isfinite(val) && val > 0) ||
            throw(ArgumentError("sampled payoff at m = $m, k = $k must be finite and > 0, got $val"))
        W[m, k] = Float64(val)
    end
    wins(s, k) = ((s - 1) >> (k - 1)) & 1 == 1
    π_block = [prod(wins(s, k) ? pf[k] : 1.0 - pf[k] for k in 1:n) / M for s in 1:B]
    π = repeat(π_block, M)
    X_bets = Matrix{Float64}(undef, S, n)
    for m in 1:M, s in 1:B
        r = (m - 1) * B + s
        for k in 1:n
            X_bets[r, k] = wins(s, k) ? W[m, k] : 0.0
        end
    end
    X = cash ? hcat(ones(Float64, S), X_bets) : X_bets
    bet_names = ["bet_$k" for k in 1:n]
    assets = cash ? vcat(["cash"], bet_names) : bet_names
    return ScenarioModel(π, X; assets = assets, has_cash = cash, structure = :sampled)
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

