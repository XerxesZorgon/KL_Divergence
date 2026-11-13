module Kelly_KL

using LinearAlgebra
using Statistics
using JuMP
import Clarabel, SCS, Distributions

const MOI = JuMP.MOI

export ScenarioModel, single_event, simultaneous_bets, correlated_bets, payoffs_from_odds, growth
export BetResult, CertificationError, SolverError, kkt_residual, kkt_components, certify
export optimal_allocation
export beta_predictive, dirichlet_predictive, predictive_from_draws, sampled_payoffs, growth_se, mixture, bma_weights
export kl, growth_loss, holdout_logscore, logscore_difference, growth_curve, zero_growth_point
export drawdown_probability, effective_multiple, ruin_probability_classical, simulate_ruin, n_detect, n_relative
export RobustResult, kl_worst_case, kl_inner_direct, kl_cash_radius, kl_robust_allocation, wasserstein_worst_case, wasserstein_robust_allocation, kl_radius_chi2

const PROB_TOL = 1e-12
const ACTIVE_TOL = 1e-12
const CERT_TOL = 1e-10
const GAP_TOL = 1e-7
const DIRECT_TOL = 1e-12
const DIRECT_MAX_ITER = 500
const MAX_SIMULTANEOUS_BETS = 16
const MAX_SAMPLED_SCENARIOS = 1_048_576
const VALID_STRUCTURES = (:single_event, :simultaneous, :correlated, :sampled, :mixture)

include("model.jl")
include("box.jl")
include("certify.jl")
include("exact.jl")
include("conic.jl")
include("polish.jl")
include("uncertainty.jl")
include("optimize.jl")
include("diagnostics.jl")
include("robust.jl")
include("crosscheck.jl")

end
