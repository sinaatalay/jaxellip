"""
jaxellip -- Fast, differentiable complete elliptic integrals using JAX.

This module provides functions `ellipk(m)`, `ellipe(m)`, and `ellipkm1(x)` with a
drop-in API compatible with SciPy. Implementation is based on Carlson symmetric forms
`R_F` and `R_D`, computed via an iterative duplication algorithm with quartic
convergence (9 steps by default), fully fused into a single XLA computation via JAX
primitives.
"""

from functools import partial

import jax
import jax.numpy as jnp

# Number of duplication steps to achieve quartic convergence
_NUM_DUPLICATION_STEPS = 9


@partial(jax.jit, inline=True)
def _carlson_duplication_step(
    state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], _: None
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
    """
    Perform one iteration of Carlson's duplication algorithm for R_F and R_D.

    Args:
        state: A tuple of five arrays:
            - X, Y, Z: Current values of the integral arguments.
            - rd_accumulator: Accumulator for R_D.
            - scale_factor: Current scale factor in the iteration.
        _: Unused placeholder for compatibility with `jax.lax.scan`.

    Returns:
        new_state: Updated state after one duplication step.
        None: Placeholder for scan output.
    """
    X, Y, Z, rd_acc, scale = state

    sqrt_X, sqrt_Y, sqrt_Z = jnp.sqrt(X), jnp.sqrt(Y), jnp.sqrt(Z)
    lambda_val = sqrt_X * sqrt_Y + sqrt_Y * sqrt_Z + sqrt_Z * sqrt_X

    rd_acc += scale / (sqrt_Z * (Z + lambda_val))
    scale *= 0.25

    X = 0.25 * (X + lambda_val)
    Y = 0.25 * (Y + lambda_val)
    Z = 0.25 * (Z + lambda_val)

    return (X, Y, Z, rd_acc, scale), None


@jax.jit
def _compute_rf_rd(
    x0: jax.Array, y0: jax.Array, z0: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Compute Carlson symmetric integrals R_F(x0, y0, z0) and R_D(x0, y0, z0)
    using iterative duplication and series correction.

    Args:
        x0, y0, z0: Non-negative input arrays representing the arguments of the integrals.

    Returns:
        rf: Value of R_F(x0, y0, z0).
        rd: Value of R_D(x0, y0, z0).
    """
    init_state = (x0, y0, z0, jnp.zeros_like(x0), jnp.ones_like(x0))

    (X, Y, Z, rd_acc, scale), _ = jax.lax.scan(
        _carlson_duplication_step, init_state, None, length=_NUM_DUPLICATION_STEPS
    )

    mu = (X + Y + Z) / 3.0
    dx = (mu - X) / mu
    dy = (mu - Y) / mu
    dz = (mu - Z) / mu
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz

    rf = (
        1.0 - e2 / 10.0 + e3 / 14.0 + (e2 * e2) / 24.0 - 3.0 * e2 * e3 / 44.0
    ) / jnp.sqrt(mu)

    ave = (X + Y + 3.0 * Z) / 5.0
    dx2 = (ave - X) / ave
    dy2 = (ave - Y) / ave
    dz2 = (ave - Z) / ave
    ea = dx2 * dy2 - dz2 * dz2
    eb = dx2 * dy2 * dz2

    rd = 3.0 * rd_acc + scale * (
        1.0
        + (3.0 / 14.0) * ea
        + (1.0 / 6.0) * eb
        + (9.0 / 88.0) * (ea * ea)
        - (3.0 / 22.0) * eb * dz2
    )

    return rf, rd


@jax.jit
def ellipk(m: jax.Array) -> jax.Array:
    """
    Compute the complete elliptic integral of the first kind, K(m).

    Uses the Carlson R_F form: K(m) = R_F(0, 1 - m, 1), valid for 0 <= m <= 1.

    Args:
        m: Input array of modulus values.

    Returns:
        K(m) array with:
            - K(1) = +inf
            - NaN for m > 1
    """
    rf, _ = _compute_rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    inf_mask = m == 1.0
    nan_mask = m > 1.0
    result = jnp.where(inf_mask, jnp.inf, rf)
    return jnp.where(nan_mask, jnp.nan, result)


@jax.jit
def ellipe(m: jax.Array) -> jax.Array:
    """
    Compute the complete elliptic integral of the second kind, E(m).

    Uses the Carlson forms: E(m) = R_F(0, 1 - m, 1) - (m / 3) * R_D(0, 1 - m, 1).

    Args:
        m: Input array of modulus values.

    Returns:
        E(m) array with:
            - E(1) = 1
            - NaN for m > 1
    """
    rf, rd = _compute_rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    val = rf - (m / 3.0) * rd
    val = jnp.where(m == 1.0, jnp.array(1.0, dtype=val.dtype), val)
    return jnp.where(m > 1.0, jnp.nan, val)


@jax.jit
def ellipkm1(x: jax.Array) -> jax.Array:
    """
    Compute K(1 - x) with improved accuracy for small x.

    Uses a logarithmic series approximation for x < 1e-3:
        K(1 - x) ≈ (1/2) * ln(16/x) + (x/16) * (ln(16/x) - 1)
    Falls back to ellipk(1 - x) otherwise.

    Args:
        x: Input array representing 1 - m.

    Returns:
        K(1 - x) array with high accuracy near x = 0.
    """
    use_series = x < 1e-3
    log_term = jnp.log(16.0 / x)
    series_approx = 0.5 * log_term + (x / 16.0) * (log_term - 1.0)

    return jnp.where(use_series, series_approx, ellipk(1.0 - x))
