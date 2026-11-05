# E8 — Exact comparison of plug-in, shrunken, half-Kelly and Bayesian bets on a binary bet
#
# Extended Kelly II, experiment E8.
# Setting: one binary bet at even odds (fractional odds b = 1). The bettor sees x wins in n training trials,
# x ~ Binomial(n, p), stakes a fraction f of wealth, and earns expected log growth
# g(f, p) = p·log(1+f) + (1−p)·log(1−f) per bet. Kelly stake s(q) = max((q(b+1) − 1)/b, 0) (uncapped).
# Every rule's final stake is capped at 0.99.
#
# Part A: fixed truth p ∈ {0.52, 0.55, 0.57, 1 − (5/6)^5, 0.65}, n ∈ {8, 20, 40, 100, 400}.
# Part B: truth drawn from a prior p ~ Beta(A, A), A ∈ {1, 10, 50}, same n (labelled Beta(A=1), etc.).
# Checks: P1 (Bayes optimality in Part B), P2 (Oracle optimality in Parts A & B),
#         P3 (MC cross-check), P4 (fractional-Kelly identity), P5 (half-Kelly = Bayes identity).

using Random, Statistics, Printf, StableRNGs, Kelly_KL
include(joinpath(@__DIR__, "plotting.jl"))

growth_binary(f::Float64, p::Float64) = f == 0.0 ? 0.0 : p * log1p(f) + (1.0 - p) * log1p(-f)
kelly_binary(q::Float64) = max(2.0 * q - 1.0, 0.0) # uncapped s(q)

function golden_section_max(f, a::Float64, b::Float64; tol::Float64 = 1e-10)
    invphi = (sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    while (b - a) > tol
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2.0
end

function precompute_bm_binom(ns)
    khat = Dict{Tuple{Int,Int},Float64}()
    for n in ns
        s_vals = [kelly_binary(xp / n) for xp in 0:n]
        for x in 0:n
            phat = x / n
            s0 = kelly_binary(phat)
            if s0 == 0.0
                khat[(n, x)] = 0.0
            elseif phat >= 1.0
                khat[(n, x)] = 1.0
            else
                log_pmf = zeros(n + 1)
                log_pmf[1] = n * log1p(-phat)
                for xp in 0:(n - 1)
                    log_pmf[xp + 2] = log_pmf[xp + 1] + log(n - xp) - log(xp + 1) + log(phat) - log1p(-phat)
                end
                pmf = exp.(log_pmf)
                pmf ./= sum(pmf)

                obj(k) = sum(pmf[xp + 1] * growth_binary(min(k * s_vals[xp + 1], 0.99), phat) for xp in 0:n)
                k_opt = golden_section_max(obj, 0.0, 1.0; tol = 1e-10)
                khat[(n, x)] = k_opt
            end
        end
    end
    return khat
end

function stake_rule(rule::Symbol, x::Int, n::Int, khat::Dict{Tuple{Int,Int},Float64}, A_prior::Int = 1)
    phat = x / n
    s0 = kelly_binary(phat)
    if rule === :oracle
        error("oracle stake requires true p, not x")
    elseif rule === :kelly
        return min(s0, 0.99)
    elseif rule === :bm
        if s0 == 0.0
            return 0.0
        else
            # b = 1 => ((b+1)/b)^2 = 4.0 (Baker & McHale 2013, Eq. 5)
            sigma2 = phat * (1.0 - phat) / n
            k = s0^2 / (s0^2 + 4.0 * sigma2)
            return min(k * s0, 0.99)
        end
    elseif rule === :bm_binom
        k = khat[(n, x)]
        return min(k * s0, 0.99)
    elseif rule === :half
        return min(0.5 * s0, 0.99)
    elseif rule === :bayes_1
        return min(kelly_binary((x + 1.0) / (n + 2.0)), 0.99)
    elseif rule === :bayes_10
        return min(kelly_binary((x + 10.0) / (n + 20.0)), 0.99)
    elseif rule === :bayes_50
        return min(kelly_binary((x + 50.0) / (n + 100.0)), 0.99)
    elseif rule === :bayes_matched
        return min(kelly_binary((x + Float64(A_prior)) / (n + 2.0 * A_prior)), 0.99)
    else
        error("unknown rule: $rule")
    end
end

function main(; seed = 2008, out_dir = joinpath(@__DIR__, "output"), quick = false)
    t0 = time()
    mkpath(out_dir)

    ns = [8, 20, 40, 100, 400]
    p_demere = 1.0 - (5.0 / 6.0)^5
    ps_A = [0.52, 0.55, 0.57, p_demere, 0.65]
    rules_A = [:oracle, :kelly, :bm, :bm_binom, :half, :bayes_1, :bayes_10, :bayes_50]

    As_B = [1, 10, 50]
    rules_B = [:oracle, :kelly, :bm, :bm_binom, :half, :bayes_1, :bayes_matched]

    khat = precompute_bm_binom(ns)

    rows = Tuple{String,String,Int,String,Float64}[]
    partA_cells = Dict{Tuple{Float64,Int},Dict{Symbol,Float64}}()

    # Part A (fixed truth)
    for p in ps_A
        truth_str = p == p_demere ? "p=0.598122" : @sprintf("p=%.2f", p)
        growth_oracle = growth_binary(min(kelly_binary(p), 0.99), p)
        for n in ns
            log_pmf = zeros(n + 1)
            log_pmf[1] = n * log1p(-p)
            for x in 0:(n - 1)
                log_pmf[x + 2] = log_pmf[x + 1] + log(n - x) - log(x + 1) + log(p) - log1p(-p)
            end
            pmf = exp.(log_pmf)
            pmf ./= sum(pmf)

            cell_dict = Dict{Symbol,Float64}()
            for r in rules_A
                gr = r === :oracle ? growth_oracle : sum(pmf[x + 1] * growth_binary(stake_rule(r, x, n, khat), p) for x in 0:n)
                cell_dict[r] = gr
                push!(rows, ("A", truth_str, n, string(r), gr))
            end
            partA_cells[(p, n)] = cell_dict
        end
    end

    # Part B (truth drawn from a prior)
    partB_cells = Dict{Tuple{Int,Int},Dict{Symbol,Float64}}()

    # 100 000-point midpoint grid on (0, 1) for oracle
    M_grid = 100_000
    p_grid = [(k - 0.5) / M_grid for k in 1:M_grid]
    s_grid = [min(kelly_binary(pk), 0.99) for pk in p_grid]
    g_grid = [growth_binary(s_grid[k], p_grid[k]) for k in 1:M_grid]

    for A in As_B
        truth_str = "Beta(A=$A)"
        log_w = [(A - 1) * (log(pk) + log1p(-pk)) for pk in p_grid]
        max_lw = maximum(log_w)
        w = exp.(log_w .- max_lw)
        w ./= sum(w)
        growth_oracle_B = sum(w .* g_grid)

        for n in ns
            log_BB = zeros(n + 1)
            log_BB[1] = sum(log((A + j) / (2A + j)) for j in 0:(n - 1))
            for x in 0:(n - 1)
                log_BB[x + 2] = log_BB[x + 1] + log(n - x) - log(x + 1) + log(x + A) - log(n - x - 1 + A)
            end
            BB = exp.(log_BB)
            BB ./= sum(BB)

            cell_dict = Dict{Symbol,Float64}()
            for r in rules_B
                if r === :oracle
                    gr = growth_oracle_B
                else
                    gr = 0.0
                    for x in 0:n
                        mx = (x + A) / (n + 2A)
                        fx = stake_rule(r, x, n, khat, A)
                        gr += BB[x + 1] * growth_binary(fx, mx)
                    end
                end
                cell_dict[r] = gr
                push!(rows, ("B", truth_str, n, string(r), gr))
            end
            partB_cells[(A, n)] = cell_dict
        end
    end

    # Check P1: in every Part B cell, bayes_matched ≥ every other non-oracle rule − 1e-12
    min_margin_P1 = Inf
    for ((A, n), cell) in partB_cells
        bm_val = cell[:bayes_matched]
        for (r, val) in cell
            (r === :oracle || r === :bayes_matched) && continue
            margin = bm_val - val
            min_margin_P1 = min(min_margin_P1, margin)
        end
    end
    pass_P1 = min_margin_P1 ≥ -1e-12

    # Check P2: in every cell of both parts, oracle ≥ every other rule − 1e-9
    min_margin_P2 = Inf
    for ((p, n), cell) in partA_cells
        orc_val = cell[:oracle]
        for (r, val) in cell
            r === :oracle && continue
            margin = orc_val - val
            min_margin_P2 = min(min_margin_P2, margin)
        end
    end
    for ((A, n), cell) in partB_cells
        orc_val = cell[:oracle]
        for (r, val) in cell
            r === :oracle && continue
            margin = orc_val - val
            min_margin_P2 = min(min_margin_P2, margin)
        end
    end
    pass_P2 = min_margin_P2 ≥ -1e-9

    # Check P3: Monte Carlo cross-check of Part A, p = 0.57, n = 40, rule `half`
    # 10^6 draws of x (StableRNG(2008), x as a sum of n Bernoulli draws); |MC mean − exact| ≤ 3·SE
    rng = StableRNG(seed)
    M_p3 = quick ? 100_000 : 1_000_000
    sum_y = 0.0
    sum_sq_y = 0.0
    p_p3 = 0.57
    n_p3 = 40
    for _ in 1:M_p3
        x = 0
        for _ in 1:n_p3
            x += rand(rng) < p_p3
        end
        y = growth_binary(stake_rule(:half, x, n_p3, khat), p_p3)
        sum_y += y
        sum_sq_y += y * y
    end
    mc_mean = sum_y / M_p3
    mc_var = (sum_sq_y - sum_y^2 / M_p3) / (M_p3 - 1)
    mc_se = sqrt(mc_var / M_p3)
    exact_p3 = partA_cells[(0.57, 40)][:half]
    diff_p3 = abs(mc_mean - exact_p3)
    pass_P3 = diff_p3 ≤ 3.0 * mc_se

    # Check P4: for every n and A ∈ {1, 10, 50} and every x = 0..n,
    # bayes_A stake equals min(0.99, n/(n + 2A)·max(2x/n − 1, 0)) within 1e-15
    max_diff_P4 = 0.0
    for n in ns, A in As_B, x in 0:n
        f_bayes = min(kelly_binary((x + Float64(A)) / (n + 2.0 * A)), 0.99)
        f_frac = min(0.99, (n / (n + 2.0 * A)) * max(2.0 * x / n - 1.0, 0.0))
        diff = abs(f_bayes - f_frac)
        max_diff_P4 = max(max_diff_P4, diff)
    end
    pass_P4 = max_diff_P4 ≤ 1e-15

    # Check P5: half growth equals bayes_10 at n = 20 and bayes_50 at n = 100 for every Part A truth,
    # and equals bayes_matched for Beta(A=10) at n = 20 and Beta(A=50) at n = 100, each within 1e-15
    max_diff_P5 = 0.0
    for p in ps_A
        diff_b10 = abs(partA_cells[(p, 20)][:half] - partA_cells[(p, 20)][:bayes_10])
        diff_b50 = abs(partA_cells[(p, 100)][:half] - partA_cells[(p, 100)][:bayes_50])
        max_diff_P5 = max(max_diff_P5, diff_b10, diff_b50)
    end
    diff_B10 = abs(partB_cells[(10, 20)][:half] - partB_cells[(10, 20)][:bayes_matched])
    diff_B50 = abs(partB_cells[(50, 100)][:half] - partB_cells[(50, 100)][:bayes_matched])
    max_diff_P5 = max(max_diff_P5, diff_B10, diff_B50)
    pass_P5 = max_diff_P5 ≤ 1e-15

    pass = pass_P1 && pass_P2 && pass_P3 && pass_P4 && pass_P5

    # Output CSV and metadata
    write_csv(joinpath(out_dir, "e8_demere.csv"), ["part", "truth", "n", "rule", "growth"], rows)
    write_meta(joinpath(out_dir, "e8_meta.toml"); seed, wall_seconds = time() - t0,
               extra = Dict{String,Any}(
                   "p1_min_margin" => min_margin_P1,
                   "p1_pass" => pass_P1,
                   "p2_min_margin" => min_margin_P2,
                   "p2_pass" => pass_P2,
                   "p3_mc_mean" => mc_mean,
                   "p3_exact" => exact_p3,
                   "p3_se" => mc_se,
                   "p3_diff" => diff_p3,
                   "p3_pass" => pass_P3,
                   "p4_max_diff" => max_diff_P4,
                   "p4_pass" => pass_P4,
                   "p5_max_diff" => max_diff_P5,
                   "p5_pass" => pass_P5,
                   "pass" => pass
               ))

    @printf("E8: check P1 (bayes_matched >= non-oracle - 1e-12 in Part B): min margin = %+.3e → %s\n",
            min_margin_P1, pass_P1 ? "PASS" : "FAIL")
    @printf("E8: check P2 (oracle >= all rules - 1e-9 in Parts A and B): min margin = %+.3e → %s\n",
            min_margin_P2, pass_P2 ? "PASS" : "FAIL")
    @printf("E8: check P3 (MC cross-check p=0.57, n=40, half, %d draws): MC %.12e vs exact %.12e (diff %.2e, 3*SE %.2e) → %s\n",
            M_p3, mc_mean, exact_p3, diff_p3, 3.0 * mc_se, pass_P3 ? "PASS" : "FAIL")
    @printf("E8: check P4 (fractional-Kelly identity within 1e-15): max abs diff = %.3e → %s\n",
            max_diff_P4, pass_P4 ? "PASS" : "FAIL")
    @printf("E8: check P5 (half-Kelly = Bayes identity within 1e-15): max abs diff = %.3e → %s\n",
            max_diff_P5, pass_P5 ? "PASS" : "FAIL")
    println("E8: ", pass ? "PASS" : "FAIL")

    return (; rows, pass)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
