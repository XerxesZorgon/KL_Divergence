module Kelly_KL

using LinearAlgebra

export ScenarioModel, single_event, payoffs_from_odds, growth
export BetResult, CertificationError, SolverError, kkt_residual, certify
export optimal_allocation
export kl

const PROB_TOL = 1e-12
const ACTIVE_TOL = 1e-12
const CERT_TOL = 1e-10
const RESERVE_TOL = 1e-15
const VALID_STRUCTURES = (:single_event, :simultaneous, :correlated, :sampled, :mixture)

include("model.jl")
include("certify.jl")
include("exact.jl")
include("optimize.jl")
include("diagnostics.jl")
include("crosscheck.jl")

end
