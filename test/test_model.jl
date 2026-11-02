using LinearAlgebra

@testset "model" begin
    # Valid construction and unrenormalized storage
    π_unnorm = [0.3, 0.7 + 5e-13]
    X_valid = [1.0 2.0 0.0; 1.0 0.0 1.5]
    m = ScenarioModel(π_unnorm, X_valid;
                      assets = ["cash", "outcome_1", "outcome_2"],
                      has_cash = true,
                      structure = :single_event)
    @test m.π == π_unnorm
    @test m.π[2] === π_unnorm[2]
    @test m.X == X_valid
    @test m.has_cash == true
    @test m.structure == :single_event

    # Rule 1: dimensions and unique assets
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], ones(3, 2);
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], ones(2, 2);
                                             assets = ["cash"],
                                             has_cash = true,
                                             structure = :single_event)
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], ones(2, 2);
                                             assets = ["cash", "cash"],
                                             has_cash = true,
                                             structure = :single_event)

    # Rule 2: π_s ≥ 0 and |sum(π) - 1| ≤ 1e-12
    @test_throws ArgumentError ScenarioModel([-0.1, 1.1], ones(2, 2);
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)
    @test_throws ArgumentError ScenarioModel([0.5, 0.6], ones(2, 2);
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)

    # Rule 3: X finite and ≥ 0
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], [1.0 -1.0; 1.0 2.0];
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], [1.0 Inf; 1.0 2.0];
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)

    # Rule 4: has_cash ⇒ X[:,1] == ones(S) and assets[1] == "cash"
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], [1.0 2.0; 0.5 2.0];
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], ones(2, 2);
                                             assets = ["not_cash", "a"],
                                             has_cash = true,
                                             structure = :single_event)

    # Rule 5: structure ∈ (:single_event, :simultaneous, :correlated, :sampled, :mixture)
    @test_throws ArgumentError ScenarioModel([0.5, 0.5], ones(2, 2);
                                             assets = ["cash", "a"],
                                             has_cash = true,
                                             structure = :invalid_structure)

    # Rule 6: feasibility (positive-probability row must have a positive entry)
    @test_throws DomainError ScenarioModel([0.5, 0.5], [2.0 0.0; 0.0 0.0];
                                           assets = ["a", "b"],
                                           has_cash = false,
                                           structure = :single_event)

    # single_event exact equality with E1 returns_matrix expression
    q = [0.2, 0.5, 0.3]
    ohat = [4.0, 1.8, 3.2]
    n = length(ohat)
    m_cash = single_event(q, ohat; cash = true)
    @test m_cash.X == hcat(ones(n), Matrix(Diagonal(ohat)))
    @test m_cash.assets == ["cash", "outcome_1", "outcome_2", "outcome_3"]
    @test m_cash.has_cash == true
    @test m_cash.structure == :single_event

    m_nocash = single_event(q, ohat; cash = false)
    @test m_nocash.X == Matrix(Diagonal(ohat))
    @test m_nocash.assets == ["outcome_1", "outcome_2", "outcome_3"]
    @test m_nocash.has_cash == false
    @test m_nocash.structure == :single_event

    # payoffs_from_odds
    @test payoffs_from_odds([2.0, 3.5]; commission = 0.05) ≈ [1.95, 3.375] atol = 1e-15

    # growth
    X_g = [1.0 2.0; 1.0 0.0]
    a_g = [0.5, 0.5]
    π_g = [0.5, 0.5]
    m_g = ScenarioModel(π_g, X_g; assets = ["cash", "outcome_1"], has_cash = true, structure = :single_event)
    @test growth(X_g, a_g, π_g) ≈ -0.14384103622589045 atol = 1e-15
    @test growth(m_g, a_g) ≈ -0.14384103622589045 atol = 1e-15
end
