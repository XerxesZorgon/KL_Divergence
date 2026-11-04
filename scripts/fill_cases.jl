# Writes test/cases/cases.toml: the shared Julia/Python parity cases (T-PY-1) with expected values computed by Kelly_KL.
# Run from code/ on the reference machine only when the cases change:  julia --project=. scripts/fill_cases.jl
# The case inputs live here (single source); the TOML file is generated — do not edit it by hand.
# TOML rather than JSON so neither language needs an extra dependency (Julia: TOML stdlib; Python ≥ 3.11: tomllib).

using Kelly_KL, TOML
include(joinpath(@__DIR__, "..", "test", "cases", "case_runner.jl"))

const TWO_BETS = Dict("type" => "simultaneous", "p" => [0.3, 0.25], "w" => [4.0, 6.0], "cash" => true)
const E7_MODEL = Dict("type" => "scenario", "pi" => [0.25, 0.30, 0.25, 0.20],
                      "X" => [[1.0, 1.40, 1.25], [1.0, 1.10, 1.05], [1.0, 0.90, 0.95], [1.0, 0.60, 0.80]],
                      "assets" => ["cash", "A", "B"], "has_cash" => true, "structure" => "correlated")
const CRASH = [[1.0, 0.20, 0.60]]

const CASES = [
    Dict("id" => "two_bets_parent_article", "kind" => "allocation", "inputs" => Dict("model" => TWO_BETS)),
    Dict("id" => "single_event_three_outcomes", "kind" => "allocation",
         "inputs" => Dict("model" => Dict("type" => "single_event", "q" => [0.5, 0.3, 0.2], "ohat" => [2.2, 3.1, 5.5], "cash" => true))),
    Dict("id" => "single_event_margin_cash_held", "kind" => "allocation",
         "inputs" => Dict("model" => Dict("type" => "single_event", "q" => [0.45, 0.35, 0.2], "ohat" => [2.4, 2.6, 4.4], "cash" => true))),
    Dict("id" => "single_event_no_cash", "kind" => "allocation",
         "inputs" => Dict("model" => Dict("type" => "single_event", "q" => [0.5, 0.3, 0.2], "ohat" => [1.9, 3.0, 4.8], "cash" => false))),
    Dict("id" => "three_simultaneous_bets", "kind" => "allocation",
         "inputs" => Dict("model" => Dict("type" => "simultaneous", "p" => [0.55, 0.3, 0.12], "w" => [2.0, 3.6, 9.0], "cash" => true))),
    Dict("id" => "correlated_two_bets", "kind" => "allocation",
         "inputs" => Dict("model" => Dict("type" => "correlated", "P_joint" => [0.40, 0.15, 0.10, 0.35],
                                          "W" => [[0.0, 0.0], [2.1, 0.0], [0.0, 2.3], [2.1, 2.3]]))),
    Dict("id" => "two_assets_e7_model", "kind" => "allocation", "inputs" => Dict("model" => E7_MODEL)),
    Dict("id" => "kl_robust_two_bets", "kind" => "kl", "inputs" => Dict("model" => TWO_BETS, "eps" => 0.01)),
    Dict("id" => "kl_robust_e7_n5000", "kind" => "kl", "inputs" => Dict("model" => E7_MODEL, "eps" => 0.000781472)),
    Dict("id" => "kl_robust_all_cash", "kind" => "kl", "inputs" => Dict("model" => TWO_BETS, "eps" => 0.05)),
    Dict("id" => "wasserstein_l1_unseen_crash", "kind" => "wasserstein",
         "inputs" => Dict("model" => E7_MODEL, "eps" => 0.0125, "unseen" => CRASH, "cost" => "l1")),
    Dict("id" => "wasserstein_linf_unseen_crash", "kind" => "wasserstein",
         "inputs" => Dict("model" => E7_MODEL, "eps" => 0.00825, "unseen" => CRASH, "cost" => "linf")),
    Dict("id" => "wasserstein_two_bets_no_unseen", "kind" => "wasserstein",
         "inputs" => Dict("model" => TWO_BETS, "eps" => 0.05, "cost" => "l1")),
    Dict("id" => "diag_kl", "kind" => "diagnostic", "inputs" => Dict("function" => "kl", "args" => [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])),
    Dict("id" => "diag_n_detect", "kind" => "diagnostic", "inputs" => Dict("function" => "n_detect", "args" => [0.5, 2.2])),
    Dict("id" => "diag_n_relative", "kind" => "diagnostic", "inputs" => Dict("function" => "n_relative", "args" => [0.5, 2.2])),
    Dict("id" => "diag_ruin_classical", "kind" => "diagnostic", "inputs" => Dict("function" => "ruin_probability_classical", "args" => [0.48, 10, 20])),
    Dict("id" => "diag_drawdown", "kind" => "diagnostic", "inputs" => Dict("function" => "drawdown_probability", "args" => [0.3, 1.5])),
    Dict("id" => "diag_kl_radius_chi2", "kind" => "diagnostic", "inputs" => Dict("function" => "kl_radius_chi2", "args" => [4, 200])),
    Dict("id" => "diag_bma_weights", "kind" => "diagnostic", "inputs" => Dict("function" => "bma_weights", "args" => [[-120.3, -118.9, -125.0]])),
    Dict("id" => "diag_beta_predictive", "kind" => "diagnostic", "inputs" => Dict("function" => "beta_predictive", "args" => [[31.0, 13.0], [69.0, 87.0]])),
    Dict("id" => "diag_kl_cash_radius", "kind" => "diagnostic", "inputs" => Dict("function" => "kl_cash_radius", "model" => TWO_BETS)),
    Dict("id" => "diag_zero_growth_point", "kind" => "diagnostic", "inputs" => Dict("function" => "zero_growth_point", "model" => TWO_BETS)),
    Dict("id" => "diag_growth_curve", "kind" => "diagnostic",
         "inputs" => Dict("function" => "growth_curve", "model" => TWO_BETS, "args" => [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]])),
    Dict("id" => "diag_kl_worst_case", "kind" => "diagnostic", "inputs" => Dict("function" => "kl_worst_case", "model" => TWO_BETS, "args" => [0.01])),
    Dict("id" => "diag_wasserstein_worst_case", "kind" => "diagnostic",
         "inputs" => Dict("function" => "wasserstein_worst_case", "model" => E7_MODEL, "args" => [0.0125], "unseen" => CRASH, "cost" => "l1")),
]

function main(; out = joinpath(@__DIR__, "..", "test", "cases", "cases.toml"))
    rows = Dict{String,Any}[]
    for c in CASES
        got = run_case(c)
        exp = Dict{String,Any}(k => (v isa AbstractArray ? collect(Float64, v) : Float64(v)) for (k, v) in got)
        push!(rows, Dict{String,Any}("id" => c["id"], "kind" => c["kind"], "tol" => 1e-10, "inputs" => c["inputs"], "expected" => exp))
        println("filled ", c["id"])
    end
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Generated by scripts/fill_cases.jl on Julia $(VERSION) — do not edit by hand.")
        TOML.print(io, Dict("case" => rows); sorted = true)
    end
    println("wrote $(length(rows)) cases to $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
