"""
jaxellip - Super-fast, differentiable complete elliptic integrals.

Optimized version using JAX primitives only (lax.scan / lax.select), resulting in a
single fused XLA computation. The public API is drop-in compatible with SciPy's ellipk,
ellipe, and ellipkm1.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax

__all__ = ["ellipe", "ellipk", "ellipkm1"]

_ITERS = 9  # 9 duplication steps (quartic convergence); keeps E-error < 1e-4


@partial(jit, inline=True)
def _dup_step(vals, _):
    """Performs a single Carlson duplication step.

    Args:
        vals: Tuple containing intermediate values in the duplication process.

    Returns:
        Updated tuple after one duplication step.
    """
    x, y, z, sum_rd, fac = vals
    sx, sy, sz = jnp.sqrt(x), jnp.sqrt(y), jnp.sqrt(z)
    lam = sx * sy + sy * sz + sz * sx
    sum_rd += fac / (sz * (z + lam))
    fac *= 0.25
    x = 0.25 * (x + lam)
    y = 0.25 * (y + lam)
    z = 0.25 * (z + lam)
    return (x, y, z, sum_rd, fac), None


@jit
def _rf_rd(x: Array, y: Array, z: Array) -> tuple[Array, Array]:
    """Computes the Carlson symmetric forms R_F and R_D.

    Uses an iterative duplication method with tail approximations.

    Args:
        x: First input array.
        y: Second input array.
        z: Third input array.

    Returns:
        A tuple (rf, rd) where:
        - rf is the Carlson R_F value.
        - rd is the Carlson R_D value.
    """
    init = (x, y, z, jnp.zeros_like(x), jnp.ones_like(x))
    (x, y, z, sum_rd, fac), _ = lax.scan(_dup_step, init, None, length=_ITERS)

    # Tail for R_F
    mu = (x + y + z) / 3.0
    dx, dy, dz = (mu - x) / mu, (mu - y) / mu, (mu - z) / mu
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    rf = (
        1.0 - e2 / 10.0 + e3 / 14.0 + (e2 * e2) / 24.0 - 3.0 * e2 * e3 / 44.0
    ) / jnp.sqrt(mu)

    # Tail for R_D
    ave = (x + y + 3.0 * z) / 5.0
    dx2, dy2, dz2 = (ave - x) / ave, (ave - y) / ave, (ave - z) / ave
    ea = dx2 * dy2 - dz2 * dz2
    eb = dx2 * dy2 * dz2
    rd = 3.0 * sum_rd + fac * (
        1.0
        + (3.0 / 14.0) * ea
        + (1.0 / 6.0) * eb
        + (9.0 / 88.0) * ea * ea
        - (3.0 / 22.0) * eb * dz2
    )
    return rf, rd


@jit
def ellipk(m: Array) -> Array:
    """Computes the complete elliptic integral of the first kind K(m).

    Args:
        m: Input modulus (0 <= m <= 1).

    Returns:
        The elliptic integral K(m), or inf/nan for invalid inputs.
    """
    rf, _ = _rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    rf = lax.select(m == 1.0, jnp.full_like(rf, jnp.inf), rf)
    return lax.select(m > 1.0, jnp.full_like(rf, jnp.nan), rf)


@jit
def ellipe(m: Array) -> Array:
    """Computes the complete elliptic integral of the second kind E(m).

    Args:
        m: Input modulus (0 <= m <= 1).

    Returns:
        The elliptic integral E(m), or 1.0/NaN for edge cases.
    """
    rf, rd = _rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    val = rf - (m / 3.0) * rd
    val = lax.select(m == 1.0, jnp.full_like(val, 1.0), val)
    return lax.select(m > 1.0, jnp.full_like(val, jnp.nan), val)


@jit
def ellipkm1(x: Array) -> Array:
    """Computes K(1 - x), accurate for x near 0 (i.e., m near 1).

    Uses a logarithmic approximation for x < 1e-3.

    Args:
        x: Input value (x = 1 - m), close to 0 for high accuracy.

    Returns:
        The elliptic integral K(1 - x), with improved accuracy for small x.
    """
    small = x < 1e-3
    approx = 0.5 * jnp.log(16.0 / x) + (x / 16.0) * (jnp.log(16.0 / x) - 1.0)
    return lax.select(small, approx, ellipk(1.0 - x))
