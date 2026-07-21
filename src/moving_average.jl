"""
    moving_average(x, window)

Compute the centred moving average of vector `x` over `window` points.
"""
function moving_average(x, window)
    n = length(x)
    out = zeros(n)
    half = div(window, 2)
    for i in 1:n
        lo = i - half
        hi = i + half
        out[i] = sum(x[lo:hi]) / window
    end
    return out
end
