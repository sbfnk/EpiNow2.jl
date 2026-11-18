@testitem "Model helpers" setup = [TestSetup] begin
    # convolve
    signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    kernel = [0.5, 0.3, 0.2]
    result = EpiNow2.convolve(signal, kernel)
    @test length(result) == 5
    @test result[1] ≈ 1.0 * 0.5

    # HSGP basis
    basis = EpiNow2.hsgp_basis(5, 1.5, 30)
    @test size(basis) == (30, 5)

    # HSGP coefficients
    spd = EpiNow2.hsgp_coefficients(5, 1.5, 0.1, 0.5, matern, 1.5)
    @test length(spd) == 5
    @test all(isfinite.(spd))

    # NegativeBinomial2
    nb = EpiNow2.NegativeBinomial2(10.0, 5.0)
    @test nb isa NegativeBinomial
    @test mean(nb) ≈ 10.0
end

@testitem "R_to_growth / growth_to_R" setup = [TestSetup] begin
    gt_pmf = [0.3, 0.4, 0.2, 0.1]
    # Normal case
    r = R_to_growth(1.5, gt_pmf)
    @test isfinite(r)
    @test r > 0

    # R=1 should give r≈0
    r1 = R_to_growth(1.0, gt_pmf)
    @test abs(r1) < 0.01

    # Round-trip: R → r → R
    R_back = growth_to_R(r, gt_pmf)
    @test R_back ≈ 1.5 atol = 0.05
end
