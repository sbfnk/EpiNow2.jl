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

@testitem "GP basis settings follow the lengthscale prior" setup = [TestSetup] begin
    # Reference values from R's gp_basis_settings() with the default
    # Matern 3/2 kernel and lengthscale prior, LogNormal(mean = 21, sd = 7,
    # max = 60).
    gp = gp_opts()
    q = EpiNow2.gp_ls_quantiles(gp.ls)
    @test q ≈ [11.68003, 33.94469] rtol = 1.0e-5
    for (n, L, M) in ((29, 10.91079, 45), (59, 5.267279, 45), (100, 3.085881, 45))
        settings = EpiNow2.gp_basis_settings(gp, n)
        @test settings[1] ≈ L rtol = 1.0e-5
        @test settings[2] == M
    end
    m52 = EpiNow2.gp_basis_settings(gp_opts(matern_order = 2.5), 59)
    @test m52[1] ≈ 4.799077 rtol = 1.0e-5
    @test m52[2] == 32
    se_settings = EpiNow2.gp_basis_settings(gp_opts(kernel = se), 59)
    @test se_settings[1] ≈ 3.745621 rtol = 1.0e-5
    @test se_settings[2] == 17
    @test EpiNow2.gp_basis_settings(gp_opts(kernel = periodic), 59) == (1.5, 10)
    # Explicit settings are used as given
    @test EpiNow2.gp_basis_settings(
        gp_opts(basis_prop = 0.2, boundary_scale = 1.5), 59
    ) == (1.5, 12)
end
