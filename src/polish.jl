function polish_log_optimal(X, π, a0; act_tol=1e-6, newton_max=50, outer_max=10, step_tol=1e-15, add_tol=1e-12, tol=CERT_TOL)
    pos = findall(>(0.0), π)
    Xp = X[pos, :]
    πp = π[pos]
    a = [x < act_tol ? 0.0 : x for x in a0]
    a ./= sum(a)
    if any(<=(0.0), Xp * a)                    # thresholding removed a weight that keeps some V_s > 0
        a = max.(collect(Float64, a0), 0.0)
        a ./= sum(a)
        any(<=(0.0), Xp * a) && (a = 0.999 .* a .+ 0.001 / length(a))   # uniform mix is feasible (model rule 6)
    end
    for _ in 1:outer_max
        for _ in 1:newton_max
            A = findall(>(0.0), a)
            k = length(A)
            V = Xp * a
            XA = Xp[:, A]
            g = XA' * (πp ./ V)
            H = -(XA' * (XA .* (πp ./ V .^ 2)))
            K = [H ones(k); ones(1, k) 0.0]
            d = (pinv(K)*vcat(-g, 0.0))[1:k]
            tmax, iblk = 1.0, 0                       # ratio test: first weight the full step would push to ≤ 0
            for i in 1:k
                if d[i] < 0 && a[A[i]] + d[i] ≤ 0.0
                    ti = a[A[i]] / -d[i]
                    ti < tmax && ((tmax, iblk) = (ti, i))
                end
            end
            t = tmax
            drop = iblk > 0
            while t ≥ 1e-12
                an = copy(a)
                an[A] .+= t .* d
                drop && (an[A[iblk]] = 0.0)
                all(≥(0.0), an) && all(>(0.0), Xp * an) && break
                t /= 2
                drop = false                  # shorter step: keep the blocking weight
            end
            t < 1e-12 && break                        # no feasible progress; the outer loop decides
            a[A] .+= t .* d
            drop && (a[A[iblk]] = 0.0)
            a .= max.(a, 0.0)
            a ./= sum(a)
            !drop && maximum(abs, t .* d) < step_tol && break
        end
        V = Xp * a
        r = Xp' * (πp ./ V)
        add = [j for j in eachindex(a) if a[j] == 0.0 && r[j] > 1 + add_tol]
        isempty(add) && kkt_residual(X, a, π) ≤ tol && break
        for j in add
            a[j] = 1e-8
        end
        a ./= sum(a)
    end
    return a, kkt_residual(X, a, π)
end
