"""
jaxellip -- Fast, differentiable complete elliptic integrals using JAX.

Drop-in compatible with ``scipy.special`` for ``ellipk(m)``, ``ellipe(m)`` and
``ellipkm1(x)``. Built on Carlson symmetric forms R_F and R_D, evaluated with a
fixed number of duplication steps chosen so the iteration converges across the
entire float64 domain. Differentiable in every JAX mode.
"""

import jax
import jax.numpy as jnp

# 14 duplication steps drive R_F and R_D to the float64 round-off floor across
# the representable domain. N=12 is already enough in the audit experiments; 14
# keeps a small safety margin with negligible extra cost.
_NUM_DUPLICATION_STEPS = 14

# Below this threshold, K(1 - x) uses the small-x logarithmic series. At and
# above it, direct R_F(0, x, 1) is accurate; the two forms agree to round-off in
# the overlap around this switch.
_ELLIPKM1_SERIES_THRESHOLD = 1e-8

_FLOAT64_SIGN_MASK = 1 << 63
_FLOAT64_ABS_MASK = (1 << 63) - 1
_FLOAT64_EXPONENT_MASK = 0x7FF
_FLOAT64_FRACTION_MASK = (1 << 52) - 1
_FLOAT64_FRACTION_BITS = 52
_FLOAT64_FRACTION_SCALE = float(1 << 52)
_FLOAT64_EXPONENT_BIAS = 1023.0
_FLOAT64_SUBNORMAL_EXPONENT = 1074.0
_LOG_2 = 0.6931471805599453


@jax.custom_jvp
def _where_invalid_nan(invalid: jax.Array, x: jax.Array, value: jax.Array) -> jax.Array:
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@_where_invalid_nan.defjvp
def _where_invalid_nan_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    invalid, x, value = primals
    _, x_dot, value_dot = tangents
    invalid_scale = jax.lax.stop_gradient(
        jnp.where(invalid, jnp.full_like(value_dot, jnp.nan), jnp.ones_like(value_dot))
    )
    tangent = jnp.where(invalid, x_dot, value_dot) * invalid_scale
    return _where_invalid_nan(invalid, x, value), tangent


def _float64_bits(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.uint64)


def _uint64(value: int) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.uint64)


def _float64_abs_bits(x: jax.Array) -> jax.Array:
    return _float64_bits(x) & _uint64(_FLOAT64_ABS_MASK)


def _is_positive(x: jax.Array) -> jax.Array:
    if x.dtype != jnp.float64:
        return x > 0.0
    bits = _float64_bits(x)
    return ((bits & _uint64(_FLOAT64_ABS_MASK)) != 0) & (
        (bits & _uint64(_FLOAT64_SIGN_MASK)) == 0
    )


def _is_zero(x: jax.Array) -> jax.Array:
    if x.dtype != jnp.float64:
        return x == 0.0
    return _float64_abs_bits(x) == 0


def _is_negative(x: jax.Array) -> jax.Array:
    if x.dtype != jnp.float64:
        return x < 0.0
    bits = _float64_bits(x)
    return ((bits & _uint64(_FLOAT64_ABS_MASK)) != 0) & (
        (bits & _uint64(_FLOAT64_SIGN_MASK)) != 0
    )


def _positive_log_value(x: jax.Array) -> jax.Array:
    if x.dtype != jnp.float64:
        return jnp.log(x)

    bits = _float64_bits(x)
    exponent_bits = (bits >> _uint64(_FLOAT64_FRACTION_BITS)) & _uint64(
        _FLOAT64_EXPONENT_MASK
    )
    fraction_bits = bits & _uint64(_FLOAT64_FRACTION_MASK)
    fraction = fraction_bits.astype(x.dtype)

    normal_log = (
        jnp.log1p(fraction / _FLOAT64_FRACTION_SCALE)
        + (exponent_bits.astype(x.dtype) - _FLOAT64_EXPONENT_BIAS) * _LOG_2
    )
    subnormal_log = jnp.log(fraction) - _FLOAT64_SUBNORMAL_EXPONENT * _LOG_2
    return jnp.where(exponent_bits == 0, subnormal_log, normal_log)


@jax.custom_jvp
def _positive_log(x: jax.Array) -> jax.Array:
    return _positive_log_value(x)


@_positive_log.defjvp
def _positive_log_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (x,) = primals
    (x_dot,) = tangents
    log_x = _positive_log_value(x)
    reciprocal = jnp.exp(-log_x) if x.dtype == jnp.float64 else 1.0 / x
    return log_x, reciprocal * x_dot


def _duplication_step(
    state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], _: None
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
    """One Carlson duplication step for R_F and R_D.

    Every argument here is strictly positive (the zero first argument is consumed
    by the closed-form first step in ``_compute_rf_rd``), so a plain ``jnp.sqrt``
    is safe and differentiable to all orders.
    """
    X, Y, Z, rd_acc, scale = state

    sqrt_X, sqrt_Y, sqrt_Z = jnp.sqrt(X), jnp.sqrt(Y), jnp.sqrt(Z)
    lambda_val = sqrt_X * sqrt_Y + sqrt_Y * sqrt_Z + sqrt_Z * sqrt_X

    rd_acc = rd_acc + scale / (sqrt_Z * (Z + lambda_val))
    scale = scale * 0.25

    X = 0.25 * (X + lambda_val)
    Y = 0.25 * (Y + lambda_val)
    Z = 0.25 * (Z + lambda_val)

    return (X, Y, Z, rd_acc, scale), None


@jax.jit
def _compute_rf_rd(y0: jax.Array, z0: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Carlson symmetric integrals R_F(0, y0, z0) and R_D(0, y0, z0).

    The first Carlson argument is always zero in this module (K, E and ellipkm1
    all reduce to R_F/R_D with a zero first argument), so the first duplication
    step is evaluated in closed form: ``sqrt(0) = 0`` collapses two of the three
    lambda terms. This keeps every argument inside ``jax.lax.scan`` strictly
    positive, which is what lets the loop use a plain ``jnp.sqrt``. A custom-JVP
    "safe" sqrt inside the scan instead makes forward-mode second-order AD (e.g.
    ``jax.hessian``) return NaN; doing the zero step in closed form removes the
    differentiated ``sqrt(0)`` entirely and makes every AD mode finite.
    """
    # Closed-form first duplication step with x0 = 0.
    sqrt_y0, sqrt_z0 = jnp.sqrt(y0), jnp.sqrt(z0)
    lambda0 = sqrt_y0 * sqrt_z0
    rd_acc0 = 1.0 / (sqrt_z0 * (z0 + lambda0))
    init_state = (
        0.25 * lambda0,
        0.25 * (y0 + lambda0),
        0.25 * (z0 + lambda0),
        rd_acc0,
        jnp.full_like(y0, 0.25),
    )
    (X, Y, Z, rd_acc, scale), _ = jax.lax.scan(
        _duplication_step, init_state, None, length=_NUM_DUPLICATION_STEPS - 1
    )

    mu = (X + Y + Z) / 3.0
    dx = (mu - X) / mu
    dy = (mu - Y) / mu
    dz = (mu - Z) / mu
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    e2_sq, e2_cu, e3_sq = e2 * e2, e2 * e2 * e2, e3 * e3
    rf = (
        1.0
        - e2 / 10.0
        + e3 / 14.0
        + e2_sq / 24.0
        - 3.0 * e2 * e3 / 44.0
        - 5.0 * e2_cu / 208.0
        + 3.0 * e3_sq / 104.0
    ) / jnp.sqrt(mu)

    ave = (X + Y + 3.0 * Z) / 5.0
    dx2 = (ave - X) / ave
    dy2 = (ave - Y) / ave
    dz2 = (ave - Z) / ave
    c1, c2, c3, c4 = 3.0 / 14.0, 1.0 / 6.0, 9.0 / 22.0, 3.0 / 26.0
    ea2 = dx2 * dy2
    eb2 = dz2 * dz2
    ec2 = ea2 - eb2
    ed2 = ea2 - 6.0 * eb2
    ef2 = ed2 + 2.0 * ec2
    s1 = ed2 * (-c1 + 0.25 * c3 * ed2 - 1.5 * c4 * dz2 * ef2)
    s2 = dz2 * (c2 * ef2 + dz2 * (-c3 * ec2 + dz2 * c4 * ea2))
    rd = 3.0 * rd_acc + (scale / (ave * jnp.sqrt(ave))) * (1.0 + s1 + s2)

    return rf, rd


@jax.jit
def ellipk(m: jax.Array) -> jax.Array:
    """Complete elliptic integral of the first kind, K(m) = R_F(0, 1-m, 1)."""
    rf, _ = _compute_rf_rd(1.0 - m, jnp.ones_like(m))
    result = jnp.where(m == 1.0, jnp.inf, rf)
    return _where_invalid_nan(m > 1.0, m, result)


@jax.jit
def ellipkm1(x: jax.Array) -> jax.Array:
    """K(1 - x), accurate as x approaches the m=1 singularity."""
    use_series = x < _ELLIPKM1_SERIES_THRESHOLD
    positive_x = _is_positive(x)
    zero_x = _is_zero(x)
    negative_x = _is_negative(x)

    # Guard dead branches so they cannot poison finite in-domain derivatives.
    x_for_rf = jnp.where(use_series, jnp.ones_like(x), x)
    x_for_log = jnp.where(positive_x, x, jnp.ones_like(x))

    # log(16) - log(x), not log(16 / x): the quotient form overflows its
    # derivative for tiny positive x.
    log_term = jnp.log(16.0) - _positive_log(x_for_log)
    series = 0.5 * log_term + (x / 8.0) * (log_term - 2.0)

    rf, _ = _compute_rf_rd(x_for_rf, jnp.ones_like(x))

    result = jnp.where(use_series, series, rf)
    result = jnp.where(zero_x, jnp.inf, result)
    return _where_invalid_nan(negative_x, x, result)


@jax.custom_jvp
def _ellipe_impl(m: jax.Array) -> jax.Array:
    """Complete elliptic integral of the second kind, E(m)."""
    rf, rd = _compute_rf_rd(1.0 - m, jnp.ones_like(m))
    val = rf - (m / 3.0) * rd
    val = jnp.where(m == 1.0, jnp.ones_like(m), val)
    return _where_invalid_nan(m > 1.0, m, val)


@_ellipe_impl.defjvp
def _ellipe_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (m,) = primals
    (m_dot,) = tangents
    _, rd = _compute_rf_rd(1.0 - m, jnp.ones_like(m))
    tangent = (-rd / 6.0) * m_dot
    invalid_scale = jax.lax.stop_gradient(
        jnp.where(m > 1.0, jnp.full_like(tangent, jnp.nan), jnp.ones_like(tangent))
    )
    tangent = jnp.where(m > 1.0, m_dot, tangent) * invalid_scale
    return _ellipe_impl(m), tangent


ellipe = jax.jit(_ellipe_impl)
