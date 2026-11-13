function crra_weights(Xp, πp, a, k)
    V = Xp * a
    lw = log.(πp) .- k .* log.(V); lw .-= maximum(lw); Q = exp.(lw); Q ./= sum(Q)
    return Q, V
end

function polish_crra(X, π, a0, k; act_tol = 1e-6, newton_max = 50, outer_max = 10, step_tol = 1e-15, add_tol = 1e-12, tol = CERT_TOL, upper = nothing)
    k ≥ 0 || throw(ArgumentError("k = 1/λ must be ≥ 0"))
    (upper === nothing || all(isinf, upper)) ||
        return _polish_crra_capped(X, π, a0, k, upper; act_tol, newton_max, outer_max, step_tol, add_tol, tol)
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

# Cap-aware snap onto A_u: project, set near-cap coordinates exactly to u_j, then spread the rounding residual over the free ones.
function _project_snap(a, u)
    # Fast path: clamp, snap to caps, spread the residual over the interior coordinates only, so exact zeros stay zero
    # (the Euclidean projection would lift them when Σ < 1).
    c = clamp.(a, 0.0, u)
    for j in eachindex(c)
        isfinite(u[j]) && c[j] ≥ u[j] - 1e-14 && (c[j] = u[j])
    end
    Fi = findall(j -> 0 < c[j] < u[j], eachindex(c))
    if !isempty(Fi)
        δ = (1 - sum(c)) / length(Fi)
        if all(j -> 0 < c[j] + δ < u[j], Fi)
            c[Fi] .+= δ
            return c
        end
    end
    b = _project_capped(a, u)
    for j in eachindex(b)
        isfinite(u[j]) && b[j] ≥ u[j] - 1e-14 && (b[j] = u[j])
    end
    F = findall(j -> 0 < b[j] < u[j], eachindex(b))
    isempty(F) || (b[F] .+= (1 - sum(b)) / length(F))
    return b
end

# polish_crra with per-asset caps u (Vector{Float64}, Inf = uncapped; at least one finite). DESIGN §10.4, ADR-013.
function _polish_crra_capped(X, π, a0, k, u; act_tol, newton_max, outer_max, step_tol, add_tol, tol)
    pos = findall(>(0.0), π); Xp = X[pos, :]; πp = π[pos]
    a = _project_snap([x < act_tol ? 0.0 : x for x in a0], u)
    if any(<=(0.0), Xp * a)
        a = _project_snap(max.(collect(Float64, a0), 0.0), u)
        any(<=(0.0), Xp * a) && (a = _project_snap(0.999 .* a .+ 0.001 / length(a), u))
    end
    for _ in 1:outer_max
        for _ in 1:newton_max
            A = findall(j -> 0 < a[j] < u[j], eachindex(a)); kk = length(A)
            kk == 0 && break
            Q, V = crra_weights(Xp, πp, a, k); Y = Xp ./ V
            r = Y' * Q; YA = Y[:, A]
            g = r[A]
            H = -(1 + k) .* (YA' * (YA .* Q)) .+ k .* (r[A] * r[A]')
            K = [H ones(kk); ones(1, kk) 0.0]
            d = (pinv(K) * vcat(-g, 0.0))[1:kk]
            tmax, iblk, kind = 1.0, 0, :none
            for i in 1:kk
                ai = a[A[i]]
                if d[i] < 0 && ai + d[i] ≤ 0.0
                    ti = ai / -d[i]; ti < tmax && ((tmax, iblk, kind) = (ti, i, :zero))
                elseif d[i] > 0 && ai + d[i] ≥ u[A[i]]
                    ti = (u[A[i]] - ai) / d[i]; ti < tmax && ((tmax, iblk, kind) = (ti, i, :cap))
                end
            end
            t = tmax; blocked = iblk > 0
            while t ≥ 1e-12
                an = copy(a); an[A] .+= t .* d
                blocked && (an[A[iblk]] = kind === :zero ? 0.0 : u[A[iblk]])
                all(≥(0.0), an) && all(an .≤ u) && all(>(0.0), Xp * an) && break
                t /= 2; blocked = false
            end
            t < 1e-12 && break
            a[A] .+= t .* d
            blocked && (a[A[iblk]] = kind === :zero ? 0.0 : u[A[iblk]])
            a = _project_snap(max.(a, 0.0), u)
            !blocked && maximum(abs, t .* d) < step_tol && break
        end
        Q, V = crra_weights(Xp, πp, a, k); r = (Xp ./ V)' * Q
        ρ, ν, U, Z, _ = kkt_components(Xp, a, Q; upper = u)
        add = [j for j in Z if r[j] > ν + add_tol]
        release = [j for j in U if r[j] < ν - add_tol]
        isempty(add) && isempty(release) && ρ ≤ tol && break
        for j in add; a[j] = 1e-8; end
        for j in release; a[j] = u[j] - 1e-8; end
        a = _project_snap(a, u)
    end
    Q, _ = crra_weights(Xp, πp, a, k)
    return a, kkt_residual(Xp, a, Q; upper = u)
end
