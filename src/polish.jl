function crra_weights(Xp, πp, a, k)
    V = Xp * a
    lw = log.(πp) .- k .* log.(V); lw .-= maximum(lw); Q = exp.(lw); Q ./= sum(Q)
    return Q, V
end

function polish_crra(X, π, a0, k; act_tol = 1e-6, newton_max = 50, outer_max = 10, step_tol = 1e-15, add_tol = 1e-12, tol = CERT_TOL)
    k ≥ 0 || throw(ArgumentError("k = 1/λ must be ≥ 0"))
    pos = findall(>(0.0), π); Xp = X[pos, :]; πp = π[pos]
    a = [x < act_tol ? 0.0 : x for x in a0]; a ./= sum(a)
    if any(<=(0.0), Xp * a)
        a = max.(collect(Float64, a0), 0.0); a ./= sum(a)
        any(<=(0.0), Xp * a) && (a = 0.999 .* a .+ 0.001 / length(a))
    end
    for _ in 1:outer_max
        for _ in 1:newton_max
            A = findall(>(0.0), a); kk = length(A)
            Q, V = crra_weights(Xp, πp, a, k); Y = Xp ./ V
            r = Y' * Q; YA = Y[:, A]
            g = r[A]
            H = -(1 + k) .* (YA' * (YA .* Q)) .+ k .* (r[A] * r[A]')
            K = [H ones(kk); ones(1, kk) 0.0]
            d = (pinv(K) * vcat(-g, 0.0))[1:kk]
            tmax, iblk = 1.0, 0
            for i in 1:kk
                if d[i] < 0 && a[A[i]] + d[i] ≤ 0.0
                    ti = a[A[i]] / -d[i]; ti < tmax && ((tmax, iblk) = (ti, i))
                end
            end
            t = tmax; drop = iblk > 0
            while t ≥ 1e-12
                an = copy(a); an[A] .+= t .* d; drop && (an[A[iblk]] = 0.0)
                all(≥(0.0), an) && all(>(0.0), Xp * an) && break
                t /= 2; drop = false
            end
            t < 1e-12 && break
            a[A] .+= t .* d; drop && (a[A[iblk]] = 0.0)
            a .= max.(a, 0.0); a ./= sum(a)
            !drop && maximum(abs, t .* d) < step_tol && break
        end
        Q, V = crra_weights(Xp, πp, a, k); r = (Xp ./ V)' * Q
        add = [j for j in eachindex(a) if a[j] == 0.0 && r[j] > 1 + add_tol]
        isempty(add) && kkt_residual(Xp, a, Q) ≤ tol && break
        for j in add; a[j] = 1e-8; end
        a ./= sum(a)
    end
    Q, _ = crra_weights(Xp, πp, a, k)
    return a, kkt_residual(Xp, a, Q)
end

polish_log_optimal(X, π, a0; kw...) = polish_crra(X, π, a0, 0.0; kw...)
