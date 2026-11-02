function polish_log_optimal(X, π, a0; act_tol = 1e-6, newton_max = 50, outer_max = 10, step_tol = 1e-15, add_tol = 1e-12, tol = CERT_TOL)
    pos = findall(>(0.0), π); Xp = X[pos, :]; πp = π[pos]
    a = [x < act_tol ? 0.0 : x for x in a0]; a ./= sum(a)
    for _ in 1:outer_max
        A = findall(>(0.0), a)
        for _ in 1:newton_max
            V = Xp * a; XA = Xp[:, A]
            g = XA' * (πp ./ V)
            H = -(XA' * (XA .* (πp ./ V .^ 2)))
            k = length(A)
            K = [H ones(k); ones(1, k) 0.0]
            d = (pinv(K) * vcat(-g, 0.0))[1:k]
            t = 1.0
            accepted = false
            while t ≥ 1e-12
                an = copy(a); an[A] .+= t .* d
                if all(>(0.0), an[A]) && all(>(0.0), Xp * an)
                    accepted = true; break
                end
                t /= 2
            end
            if !accepted
                ratios = [d[i] < 0 ? a[A[i]] / -d[i] : Inf for i in 1:k]
                a[A[argmin(ratios)]] = 0.0; a ./= sum(a)
                break
            end
            a[A] .+= t .* d
            maximum(abs, t .* d) < step_tol && break
        end
        V = Xp * a; r = Xp' * (πp ./ V)
        add = [j for j in eachindex(a) if a[j] == 0.0 && r[j] > 1 + add_tol]
        isempty(add) && kkt_residual(X, a, π) ≤ tol && break
        for j in add; a[j] = 1e-8; end
        a ./= sum(a)
    end
    return a, kkt_residual(X, a, π)
end

