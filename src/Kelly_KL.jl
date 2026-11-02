module Kelly_KL

using LinearAlgebra

export ScenarioModel, single_event, payoffs_from_odds, growth
export BetResult, CertificationError, SolverError, kkt_residual, certify

include("model.jl")
include("certify.jl")

end
