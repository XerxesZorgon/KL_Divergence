struct BetResult
    a::Vector{Float64}
    growth::Float64
    kkt::Float64
    method::Symbol
    active::Vector{Int}
    assets::Vector{String}
end

function BetResult(m::ScenarioModel, a::AbstractVector{<:Real}, method::Symbol; τ_a::Real = 1e-12)
    a_vec = Vector{Float64}(a)
    return BetResult(
        a_vec,
        growth(m, a_vec),
        kkt_residual(m.X, a_vec, m.π; τ_a = τ_a),
        method,
        findall(>(τ_a), a_vec),
        m.assets
    )
end

struct CertificationError <: Exception
    kkt::Float64
    tol::Float64
    method::Symbol
    msg::String
end

function Base.showerror(io::IO, e::CertificationError)
    print(io, "CertificationError(method=:", e.method, ", kkt=", e.kkt, " > tol=", e.tol, "): ", e.msg)
end

struct SolverError <: Exception
    status::String
    msg::String
end

function Base.showerror(io::IO, e::SolverError)
    print(io, "SolverError(status=", e.status, "): ", e.msg)
end

function kkt_residual(X::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, π::AbstractVector{<:Real}; τ_a::Real = 1e-12)
    V = X * a
    S, J = size(X)
    for s in 1:S
        if π[s] > 0 && !(V[s] > 0)
            return Inf
        end
    end
    r = zeros(Float64, J)
    for s in 1:S
        if π[s] > 0
            w = π[s] / V[s]
            for j in 1:J
                r[j] += w * X[s, j]
            end
        end
    end
    return max(
        maximum(r .- 1),
        maximum(abs, r[a .> τ_a] .- 1; init = 0.0),
        max(0.0, -minimum(a)),
        abs(sum(a) - 1)
    )
end

function certify(result::BetResult; tol::Real = 1e-10)
    if result.kkt <= tol
        return result
    end
    throw(CertificationError(result.kkt, Float64(tol), result.method, "uncertified bet"))
end
