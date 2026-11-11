# [CairoMakie extension](@id extension-cairo-makie)

`EpiNow2CairoMakieExt` is loaded automatically when CairoMakie is available alongside EpiNow2.

## Scope

Describe what this extension adds, and what a reader needs to load to get it.

## Public API

Add CairoMakie to `docs/Project.toml` and the extension module to
`EXTRA_MODULES` in `docs/docs_config.jl`, then replace this block with the
one inside it, and the extension's docstrings render here.

````markdown
```@autodocs
Modules = [Base.get_extension(EpiNow2, :EpiNow2CairoMakieExt)]
```
````
