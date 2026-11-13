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

function _kkt_residual_v01(X::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, π::AbstractVector{<:Real}, τ_a::Real)
    V = X * a
    any(s -> π[s] > 0 && !(V[s] > 0), eachindex(π)) && return Inf
    r = X' * [π[s] > 0 ? π[s] / V[s] : 0.0 for s in eachindex(π)]
    return max(maximum(r .- 1), maximum(abs, r[a .> τ_a] .- 1; init = 0.0), max(0.0, -minimum(a)), abs(sum(a) - 1))
end

"""
    kkt_components(X, a, π; τ_a = 1e-12, upper = nothing) -> (ρ, ν_star, U, Z, F)

Residual ρ, budget multiplier ν*, and the capped (U), zero (Z) and free (F) index sets (DESIGN §10.2, ADR-013).
Without caps (`upper === nothing` or every entry ≥ 1) ρ is the v0.1 residual and ν* = 1. With a finite cap the minimax
residual ρ_u is used. If some π_s > 0 has V_s ≤ 0, ρ = Inf and ν* = NaN.
"""
function kkt_components(X::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, π::AbstractVector{<:Real};
                        τ_a::Real = ACTIVE_TOL, upper = nothing)
    if upper === nothing || all(≥(1), upper)
        Z = findall(≤(τ_a), a)
        return (_kkt_residual_v01(X, a, π, τ_a), 1.0, Int[], Z, setdiff(eachindex(a), Z))
    end
    u = [x ≥ 1 ? Inf : Float64(x) for x in upper]
    U = findall(j -> isfinite(u[j]) && a[j] ≥ u[j] - τ_a, eachindex(a))
    Z = findall(j -> !(j in U) && a[j] ≤ τ_a, eachindex(a))
    F = setdiff(eachindex(a), U, Z)
    V = X * a
    any(s -> π[s] > 0 && !(V[s] > 0), eachindex(π)) && return (Inf, NaN, U, Z, F)
    r = X' * [π[s] > 0 ? π[s] / V[s] : 0.0 for s in eachindex(π)]
    A = maximum(r[vcat(F, Z)]; init = -Inf)
    B = minimum(r[vcat(F, U)]; init = Inf)
    ν_part = (isfinite(A) && isfinite(B)) ? max(0.0, (A - B) / 2) : 0.0
    ν_star = isfinite(A) && isfinite(B) ? (A + B) / 2 : (isfinite(B) ? B : A)
    ρ = max(ν_part, max(0.0, -minimum(a)), maximum(max.(0.0, a .- u)), abs(sum(a) - 1))
    return (ρ, ν_star, U, Z, F)
end

"""
    kkt_residual(X, a, π; τ_a = 1e-12, upper = nothing) -> Float64

KKT residual ρ of the allocation `a`. With per-asset caps `upper` (finite entries < 1) it is the generalized residual ρ_u (ADR-013).
"""
kkt_residual(X::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, π::AbstractVector{<:Real}; τ_a::Real = ACTIVE_TOL, upper = nothing) =
    kkt_components(X, a, π; τ_a, upper)[1]

function certify(result::BetResult; tol::Real = CERT_TOL)
    result.kkt <= tol && return result
    throw(CertificationError(result.kkt, Float64(tol), result.method, "uncertified bet"))
end

