# Reference comparator for reproduce_all.jl (ADR-009; T-REP-1, T-REP-2).
#
# Two modes:
#   exact  (reference machine, KELLY_KL_REFERENCE_HOST=1): files must be byte-identical;
#   tolerant (any other machine): the text is split into numbers and the text between them; the non-numeric text must match
#           exactly and every pair of numbers must agree within relative 1e-12 (absolute 1e-15 near zero).
# Files named `*_meta.toml` (Julia version, git commit, wall time) are never compared.

using Printf

const REL_TOL = 1e-12
const ABS_TOL = 1e-15
const NUMBER = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"

is_meta(path) = endswith(path, "_meta.toml")
numbers_close(a, b) = a == b || abs(a - b) ≤ max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))

"Split `s` into alternating (text, number) tokens; numbers are returned as `Float64`."
function tokens(s::AbstractString)
    out = Any[]; i = firstindex(s)
    for m in eachmatch(NUMBER, s)
        m.offset > i && push!(out, s[i:prevind(s, m.offset)])
        push!(out, parse(Float64, m.match))
        i = m.offset + ncodeunits(m.match)
    end
    i ≤ lastindex(s) && push!(out, s[i:end])
    return out
end

function first_difference(a::AbstractString, b::AbstractString)
    la, lb = split(a, '\n'), split(b, '\n')
    for k in 1:max(length(la), length(lb))
        x = k ≤ length(la) ? la[k] : "<end of file>"; y = k ≤ length(lb) ? lb[k] : "<end of file>"
        x == y || return "line $k: reference `$(first(x, 80))` vs output `$(first(y, 80))`"
    end
    return "files differ"
end

"""
    compare_file(ref, out; exact) -> (ok::Bool, detail::String)

Compare one output file with its reference (see the header for the two modes).
"""
function compare_file(ref::AbstractString, out::AbstractString; exact::Bool)
    isfile(out) || return (false, "missing output")
    rb, ob = read(ref), read(out)
    rb == ob && return (true, "identical")
    rs, os = String(rb), String(ob)
    exact && return (false, "not byte-identical; " * first_difference(rs, os))
    rt, ot = tokens(rs), tokens(os)
    length(rt) == length(ot) || return (false, "different structure; " * first_difference(rs, os))
    worst = 0.0
    for (x, y) in zip(rt, ot)
        if x isa Float64 && y isa Float64
            numbers_close(x, y) || return (false, "number $x vs $y; " * first_difference(rs, os))
            worst = max(worst, abs(x - y) / max(abs(x), abs(y), ABS_TOL))
        elseif x != y
            return (false, "text differs; " * first_difference(rs, os))
        end
    end
    return (true, @sprintf("within tolerance (max relative difference %.1e)", worst))
end

relfiles(dir) = sort([relpath(joinpath(root, f), dir) for (root, _, fs) in walkdir(dir) for f in fs if !is_meta(f)])

"""
    compare_dirs(ref_dir, out_dir; exact) -> Vector{(file, ok, detail)}

Every non-meta file under `ref_dir` must have a matching file under `out_dir`, and every non-meta output must have a reference.
"""
function compare_dirs(ref_dir::AbstractString, out_dir::AbstractString; exact::Bool)
    rows = Tuple{String,Bool,String}[]
    refs = relfiles(ref_dir); outs = isdir(out_dir) ? relfiles(out_dir) : String[]
    for f in refs
        push!(rows, (f, compare_file(joinpath(ref_dir, f), joinpath(out_dir, f); exact)...))
    end
    for f in setdiff(outs, refs)
        push!(rows, (f, false, "output has no reference"))
    end
    return rows
end
