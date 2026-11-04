# Shared-case runner (T-PY-1). Computes a case's outputs from its inputs; used by scripts/fill_cases.jl (to write the
# expected values on the reference machine) and by test/test_cases.jl (to check them). python/tests/case_runner.py is the
# line-for-line Python mirror. Keep the two in step: a case kind added here must be added there.

_rows(rows) = isempty(rows) ? zeros(0, 0) : Matrix{Float64}(permutedims(reduce(hcat, [Float64.(r) for r in rows])))
_v(x) = Float64.(x)

function build_model(d::AbstractDict)
    t = d["type"]
    t == "single_event" && return single_event(_v(d["q"]), _v(d["ohat"]); cash = d["cash"])
    t == "simultaneous" && return simultaneous_bets(_v(d["p"]), _v(d["w"]); cash = d["cash"])
    t == "correlated" && return correlated_bets(_v(d["P_joint"]), _rows(d["W"]))
    t == "scenario" && return ScenarioModel(_v(d["pi"]), _rows(d["X"]); assets = String.(d["assets"]),
                                            has_cash = d["has_cash"], structure = Symbol(d["structure"]))
    throw(ArgumentError("unknown model type $t"))
end

function run_case(c::AbstractDict)
    kind = c["kind"]; inp = c["inputs"]
    if kind == "allocation"
        r = optimal_allocation(build_model(inp["model"]))
        return Dict{String,Any}("a" => r.a, "growth" => r.growth)
    elseif kind == "kl"
        r = kl_robust_allocation(build_model(inp["model"]), Float64(inp["eps"]))
        return Dict{String,Any}("a" => r.a, "worst_case_growth" => r.worst_case_growth, "lambda" => r.λ)
    elseif kind == "wasserstein"
        m = build_model(inp["model"]); J = size(m.X, 2)
        U = haskey(inp, "unseen") ? _rows(inp["unseen"]) : zeros(0, J)
        r = wasserstein_robust_allocation(m, Float64(inp["eps"]); unseen = U, cost = Symbol(inp["cost"]))
        return Dict{String,Any}("a" => r.a, "worst_case_growth" => r.worst_case_growth, "lambda" => r.λ)
    elseif kind == "diagnostic"
        f = inp["function"]; x = get(inp, "args", Any[])
        f == "kl" && return Dict{String,Any}("value" => kl(_v(x[1]), _v(x[2])))
        f == "n_detect" && return Dict{String,Any}("value" => n_detect(x[1], x[2]))
        f == "n_relative" && return Dict{String,Any}("value" => n_relative(x[1], x[2]))
        f == "ruin_probability_classical" && return Dict{String,Any}("value" => ruin_probability_classical(x[1], Int(x[2]), Int(x[3])))
        f == "drawdown_probability" && return Dict{String,Any}("value" => drawdown_probability(x[1], x[2]))
        f == "kl_radius_chi2" && return Dict{String,Any}("value" => kl_radius_chi2(Int(x[1]), Int(x[2])))
        f == "bma_weights" && return Dict{String,Any}("value" => bma_weights(_v(x[1])))
        f == "beta_predictive" && return Dict{String,Any}("value" => beta_predictive(_v(x[1]), _v(x[2])))
        m = build_model(inp["model"])
        f == "kl_cash_radius" && return Dict{String,Any}("value" => kl_cash_radius(m)[1])
        a = optimal_allocation(m).a
        f == "zero_growth_point" && return Dict{String,Any}("value" => zero_growth_point(m, a).c0)
        f == "growth_curve" && return Dict{String,Any}("value" => growth_curve(m, a, _v(x[1])))
        if f == "kl_worst_case"
            v, λ, Q = kl_worst_case(m, a, Float64(x[1]))
            return Dict{String,Any}("value" => v, "lambda" => λ, "Q" => Q)
        end
        if f == "wasserstein_worst_case"
            v, λ, Q = wasserstein_worst_case(m, a, Float64(x[1]); unseen = _rows(inp["unseen"]), cost = Symbol(inp["cost"]))
            return Dict{String,Any}("value" => v, "lambda" => λ, "Q" => Q)
        end
        throw(ArgumentError("unknown diagnostic $f"))
    end
    throw(ArgumentError("unknown case kind $kind"))
end

"Largest absolute difference between computed outputs `got` and stored `expected` over every expected key."
function case_error(got::AbstractDict, expected::AbstractDict)
    err = 0.0
    for (k, e) in expected
        haskey(got, k) || return Inf
        g = got[k]
        length(g) == length(e) || return Inf
        err = max(err, maximum(abs.(Float64.(collect(g)) .- Float64.(collect(e)))))
    end
    return err
end
