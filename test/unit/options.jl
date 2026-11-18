@testitem "Options" setup = [TestSetup] begin
    # Default options should construct without error
    @test gt_opts() isa EpiNow2.GTOpts
    @test delay_opts() isa EpiNow2.DelayOpts
    @test rt_opts() isa EpiNow2.RtOpts
    @test gp_opts() isa EpiNow2.GPOpts
    @test obs_opts() isa EpiNow2.ObsOpts
    @test inference_opts() isa EpiNow2.InferenceOpts

    # With distribution argument
    gt = gt_opts(LogNormal(1.6, 0.5))
    @test gt.dist isa LogNormal
end

@testitem "opts_list" setup = [TestSetup] begin
    ol = opts_list(["A", "B", "C"], rt_opts(rw = 7))
    @test length(ol) == 3
    @test ol["A"].rw == 7
end

@testitem "error paths" setup = [TestSetup] begin
    # Missing columns
    @test_throws ArgumentError EpiNow2.EpiData(DataFrame(x = [1, 2]))
    @test_throws ArgumentError EpiNow2.SecondaryData(DataFrame(x = [1, 2]))

    # Invalid PMF
    @test_throws ArgumentError NonParametricDist([0.5, 0.3])  # doesn't sum to 1

    # Only nuts sampler is supported — invalid symbols throw MethodError
    @test_throws MethodError inference_opts(sampler = :advi)
end
