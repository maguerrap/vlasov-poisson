@jit
def bilinear_interpolation(grid_x, grid_y, values, x, y):
    """
    Perform bilinear interpolation for the given x and y points.

    Parameters:
    - grid_x: 1D array of x-coordinates (must be sorted in ascending order).
    - grid_y: 1D array of y-coordinates (must be sorted in ascending order).
    - values: 2D array of shape (len(grid_y), len(grid_x)) representing the function values at grid points.
    - x: 1D array of x-coordinates where interpolation is desired.
    - y: 1D array of y-coordinates where interpolation is desired.

    Returns:
    - Interpolated values at the specified (x, y) points.
    """
    # Find indices of the grid points surrounding the (x, y) points
    ix = jnp.clip(jnp.searchsorted(grid_x, x) - 1, 0, len(grid_x) - 2)
    iy = jnp.clip(jnp.searchsorted(grid_y, y) - 1, 0, len(grid_y) - 2)

    # Get the grid coordinates of the surrounding points
    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = grid_y[iy]
    y1 = grid_y[iy + 1]

    # Compute the weights for interpolation
    denom = (x1 - x0) * (y1 - y0)
    denom = jnp.where(denom == 0, 1e-10, denom)  # Prevent division by zero

    wa = ((x1 - x) * (y1 - y)) / denom
    wb = ((x - x0) * (y1 - y)) / denom
    wc = ((x1 - x) * (y - y0)) / denom
    wd = ((x - x0) * (y - y0)) / denom

    # Gather the values at the surrounding grid points
    f00 = values[iy, ix]
    f10 = values[iy, ix + 1]
    f01 = values[iy + 1, ix]
    f11 = values[iy + 1, ix + 1]

    # Compute the interpolated values
    interpolated = wa * f00 + wb * f10 + wc * f01 + wd * f11

    return interpolated
