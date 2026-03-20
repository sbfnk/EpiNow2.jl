# API Reference

## Main functions

```@docs
estimate_infections
epinow
regional_epinow
example_confirmed
```

## Options

```@docs
EpiNow2.GTOpts
EpiNow2.DelayOpts
EpiNow2.TruncOpts
EpiNow2.RtOpts
EpiNow2.GPOpts
EpiNow2.ObsOpts
EpiNow2.ForecastOpts
EpiNow2.InferenceOpts
```

## Distribution types

```@docs
NonParametricDist
UncertainDistribution
CompositeDelay
discretise
convolve_pmfs
```

## Result accessors

```@docs
get_samples
get_predictions
get_parameters
```
