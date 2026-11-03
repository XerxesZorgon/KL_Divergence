using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const PINNED = [
    ("JuMP", "1.32.1"), ("MathOptInterface", "1.54.0"), ("Clarabel", "0.11.1"),
    ("SCS", "2.7.1"), ("Distributions", "0.25.132"), ("StableRNGs", "1.1.0"),
]

Pkg.add([PackageSpec(name = n, version = v) for (n, v) in PINNED])
Pkg.add(["Random", "LinearAlgebra", "Printf", "Statistics", "TOML"])
for (n, v) in PINNED
    Pkg.compat(n, v)
end
Pkg.compat("julia", "1.12")

