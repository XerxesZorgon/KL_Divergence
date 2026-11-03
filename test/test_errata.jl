using StableRNGs
import Distributions

@testset "errata (T-ERR-3)" begin
    # parent article (2025-11-24) Scenario A s = 1000, Scenario B s = 2, means (0.30, 0.25),
    # gross payoffs (4.0, 6.0); repo objective_gen_bayes draws p ~ Beta(s·m, s(1−m)), then
    # win ~ Bernoulli(p), so each bet wins with probability m for every s.
    ARTICLE_A = [0.06565483237670092, 0.0990407051985005]
    ARTICLE_B = [0.044006266789879726, 0.09526125397014709]
    ARTICLE_REDUCTION = 0.1544

    # Fixed payoffs
    p̄_1000 = beta_predictive(1000.0 .* [0.30, 0.25], 1000.0 .* [0.70, 0.75])
    m_1000_fixed = simultaneous_bets(p̄_1000, [4.0, 6.0])
    r_1000_fixed = optimal_allocation(m_1000_fixed)
    assert_certified(r_1000_fixed, m_1000_fixed)

    p̄_2 = beta_predictive(2.0 .* [0.30, 0.25], 2.0 .* [0.70, 0.75])
    m_2_fixed = simultaneous_bets(p̄_2, [4.0, 6.0])
    r_2_fixed = optimal_allocation(m_2_fixed)
    assert_certified(r_2_fixed, m_2_fixed)

    @test maximum(abs.(r_2_fixed.a .- r_1000_fixed.a)) ≤ 1e-12
    @test abs(sum(r_1000_fixed.a[2:3]) - 0.16365467286269636) ≤ 1e-10
    reduction = 1 - sum(r_2_fixed.a[2:3]) / sum(r_1000_fixed.a[2:3])
    @test abs(reduction) ≤ 1e-10

    # LogNormal payoffs as in the repo (θ = 0.05), common random numbers
    samplers = [Distributions.LogNormal(log(4.0), 0.05), Distributions.LogNormal(log(6.0), 0.05)]
    m_1000_ln = sampled_payoffs(p̄_1000, samplers, 200; rng = StableRNG(1301))
    r_1000_ln = optimal_allocation(m_1000_ln)
    assert_certified(r_1000_ln, m_1000_ln)

    m_2_ln = sampled_payoffs(p̄_2, samplers, 200; rng = StableRNG(1301))
    r_2_ln = optimal_allocation(m_2_ln)
    assert_certified(r_2_ln, m_2_ln)

    @test maximum(abs.(r_2_ln.a .- r_1000_ln.a)) ≤ 1e-12

    # Documented discrepancy
    @test abs(1 - sum(ARTICLE_B) / sum(ARTICLE_A) - ARTICLE_REDUCTION) ≤ 1e-4
    @test maximum(abs.(ARTICLE_A .- r_1000_fixed.a[2:3])) > 1e-3
end

