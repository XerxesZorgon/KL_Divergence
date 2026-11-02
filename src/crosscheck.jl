"""
    cover_update(X, π; tol=1e-11, maxiter=2_000_000) -> (a, kkt, iters)

Log-optimal allocation via Cover's multiplicative update (Cover 1984).
Cross-check only, never used for returned results; it stalls when a weight underflows to 0 (see test).
"""
function cover_update(X::AbstractMatrix, π::AbstractVector; tol=1e-11, maxiter=2_000_000)
    nA = size(X, 2)
    a = fill(1.0 / nA, nA)
    r = similar(a)
    kkt = Inf
    it = 0
    for k in 1:maxiter
        it = k
        V = X * a
        r .= X' * (π ./ V)
        kkt = maximum(r) - 1.0
        kkt < tol && break
        a .*= r
        a ./= sum(a)              # guards against round-off drift
    end
    return a, kkt, it
end
