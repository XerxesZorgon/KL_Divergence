struct BetResult
    a::Vector{Float64}
    growth::Float64
    kkt::Float64
    method::Symbol
    active::Vector{Int}
    assets::Vector{String}
end

function BetResult(m::ScenarioModel, a::AbstractVector{<:Real}, method::Symbol; τ_a::Real = ACTIVE_TOL)
    a_vec = Vector{Float64}(a)
    return BetResult(a_vec, growth(m, a_vec), kkt_residual(m.X, a_vec, m.π; τ_a = τ_a), method, findall(>(τ_a), a_vec), m.assets)
end

struct CertificationError <: Exception
    kkt::Float64
    tol::Float64
    method::Symbol
    msg::String
end

Base.showerror(io::IO, e::CertificationError) =
    print(io, "CertificationError(method=:", e.method, ", kkt=", e.kkt, " > tol=", e.tol, "): ", e.msg)

struct SolverError <: Exception
    status::String
    msg::String
end

Base.showerror(io::IO, e::SolverError) = print(io, "SolverError(status=", e.status, "): ", e.msg)

function kkt_residual(X::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, π::AbstractVector{<:Real}; τ_a::Real = ACTIVE_TOL)
    V = X * a
    any(s -> π[s] > 0 && !(V[s] > 0), eachindex(π)) && return Inf
    r = X' * [π[s] > 0 ? π[s] / V[s] : 0.0 for s in eachindex(π)]
    return max(maximum(r .- 1), maximum(abs, r[a .> τ_a] .- 1; init = 0.0), max(0.0, -minimum(a)), abs(sum(a) - 1))
end

function certify(result::BetResult; tol::Real = CERT_TOL)
    result.kkt <= tol && return result
    throw(CertificationError(result.kkt, Float64(tol), result.method, "uncertified bet"))
end

