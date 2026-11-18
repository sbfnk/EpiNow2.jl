@testitem "Quality: piracy apart from delay addition" tags = [:quality] begin
    using Aqua
    using EpiNow2
    # The managed Aqua check runs with `piracies = false` because `+` on
    # delay distributions extends `Base.:+` to `Distributions` types. Treat
    # that one function as owned here so any other piracy still fails.
    Aqua.test_piracies(EpiNow2; treat_as_own = [Base.:+])
end
