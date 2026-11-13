# Reader for the committed box-oracle fixtures (ADR-014). Tests only; the generator is test/oracle/generate_box_oracle.py.
using TOML

_vec(x) = Vector{Float64}(x)
_idx(x) = isempty(x) ? Int[] : Vector{Int}(x)      # empty TOML arrays can parse as Vector{Any}

function _oracle_case(d::AbstractDict)
    opt(k) = haskey(d, k) ? d[k] : nothing
    return (id = d["id"], kind = d["kind"],
            pi = _vec(d["pi"]), X = Matrix{Float64}(reduce(vcat, (permutedims(_vec(row)) for row in d["X"]))),
            upper = _vec(d["upper"]), has_cash = d["has_cash"],
            eps = opt("eps") === nothing ? nothing : Float64(d["eps"]),
            a = _vec(d["a"]),
            growth = opt("growth") === nothing ? nothing : Float64(d["growth"]),
            worst_case_growth = opt("worst_case_growth") === nothing ? nothing : Float64(d["worst_case_growth"]),
            lambda = opt("lambda") === nothing ? nothing : Float64(d["lambda"]),
            nu_star = Float64(d["nu_star"]), U = _idx(d["U"]), Z = _idx(d["Z"]), F = _idx(d["F"]),
            rho_u = Float64(d["rho_u"]), degenerate = d["degenerate"])
end

function load_box_oracle(path = joinpath(@__DIR__, "oracle", "box_oracle.toml"))
    t = TOML.parsefile(path)
    cases = [_oracle_case(d) for d in t["cases"]]
    return (t["header"], filter(c -> c.kind == "allocation", cases), filter(c -> c.kind == "kl", cases))
end

# Convention of the existing tests: with cash, column 1 is the asset named "cash" (constructor rule 4), other assets "a<j>"; structure = :correlated.
model_of(c) = ScenarioModel(c.pi, c.X; assets = [c.has_cash && j == 1 ? "cash" : "a$j" for j in 1:size(c.X, 2)], has_cash = c.has_cash, structure = :correlated)
