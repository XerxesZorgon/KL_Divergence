using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

Pkg.add([
    PackageSpec(name="JuMP", version="1.32.1"),
    PackageSpec(name="MathOptInterface", version="1.54.0"),
    PackageSpec(name="Clarabel", version="0.11.1"),
    PackageSpec(name="SCS", version="2.7.1"),
    PackageSpec(name="ForwardDiff", version="1.4.6"),
    PackageSpec(name="Distributions", version="0.25.132"),
    PackageSpec(name="StableRNGs", version="1.1.0"),
])

Pkg.add(["Random", "LinearAlgebra", "Printf", "Statistics", "TOML"])

Pkg.compat("JuMP", "1.32.1")
Pkg.compat("MathOptInterface", "1.54.0")
Pkg.compat("Clarabel", "0.11.1")
Pkg.compat("SCS", "2.7.1")
Pkg.compat("ForwardDiff", "1.4.6")
Pkg.compat("Distributions", "0.25.132")
Pkg.compat("StableRNGs", "1.1.0")
Pkg.compat("julia", "1.12")

