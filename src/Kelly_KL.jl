module Kelly_KL

using LinearAlgebra

export ScenarioModel, single_event, payoffs_from_odds, growth
export BetResult, CertificationError, SolverError, kkt_residual, certify
export optimal_allocation
export kl

include("model.jl")
include("certify.jl")
include("exact.jl")
include("optimize.jl")
include("diagnostics.jl")

end
