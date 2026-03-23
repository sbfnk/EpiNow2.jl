using Documenter
using EpiNow2
using CairoMakie

makedocs(;
    modules=[EpiNow2],
    sitename="EpiNow2.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        size_threshold=500_000
    ),
    pages=[
        "Home" => "index.md",
        "Workflow" => "estimate_infections_workflow.md",
        "API Reference" => "api.md",
    ],
    warnonly=[:missing_docs, :cross_references, :docs_block]
)

deploydocs(;
    repo="github.com/epiforecasts/EpiNow2.jl.git",
    push_preview=true
)
