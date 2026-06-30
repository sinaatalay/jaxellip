from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

import jaxellip

jax.config.update("jax_enable_x64", True)

type NumericArray = jax.Array | np.ndarray
type NumericFunction = Callable[[NumericArray], NumericArray]

# Worst-case relative agreement with SciPy actually achieved across the tested
# range is ~2e-14 (ellipk, at m = -1e9), ~8e-16 (ellipkm1) and ~7e-15 (ellipe).
# No tested value crosses zero, so the comparison below is purely relative
# (abs=0): these tolerances are what is enforced, each with >=40x of margin.
RELATIVE_TOLERANCES_AGAINST_SCIPY = {
    "ellipk": 1e-12,
    "ellipkm1": 1e-13,
    "ellipe": 1e-12,
}

# Autodiff matches the SciPy-based analytic derivatives to ~1e-13 (median
# ~1e-16) on the well-conditioned ranges below, so 1e-11 is a tight check with a
# robust margin. The small absolute floor guards the smallest-magnitude
# derivatives without loosening the relative check where it matters.
RELATIVE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES = 1e-11
ABSOLUTE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES = 1e-12


functions = ["ellipk", "ellipkm1", "ellipe"]
a_lot_of_numbers = jnp.concatenate(
    [jnp.geomspace(-1e9, -1e-30, 100000), jnp.linspace(0, 1.1, 100000)]
)
assert a_lot_of_numbers.max() > 1.0
inputs: dict[str, jax.Array] = {
    "ellipk": a_lot_of_numbers,
    "ellipkm1": jnp.linspace(0.5, 1.0, 100000),
    "ellipe": a_lot_of_numbers,
}

# Inputs for the derivative test. The analytic derivative formulas suffer
# catastrophic cancellation as m -> 0 (E - (1 - m) K -> 0/0) and hit a real
# singularity as m -> 1, so the comparison stays away from |m| ~ 0 and m ~ 1,
# where the *reference* (not the autodiff) becomes unreliable. ellipkm1 takes
# x = 1 - m, so its range is expressed in x.
well_conditioned_numbers = jnp.concatenate(
    [jnp.linspace(-100.0, -0.01, 50000), jnp.linspace(0.01, 0.99, 50000)]
)
derivative_inputs: dict[str, jax.Array] = {
    "ellipk": well_conditioned_numbers,
    "ellipe": well_conditioned_numbers,
    "ellipkm1": jnp.linspace(0.01, 0.99, 100000),
}

# Inputs for the out-of-domain gradient test. The forward result is NaN/inf by
# design here, but the *gradient* must stay finite: the masked-out Carlson
# branch would otherwise poison reverse mode (0 * inf = NaN). These ranges
# include the exact boundaries (m = 1 for ellipk/ellipe, x = 0 for ellipkm1).
out_of_domain_inputs: dict[str, jax.Array] = {
    "ellipk": jnp.linspace(1.0, 1e6, 1000),
    "ellipe": jnp.linspace(1.0, 1e6, 1000),
    "ellipkm1": jnp.linspace(-1e6, 0.0, 1000),
}


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals(function_name: str) -> None:
    scipy_function = cast(NumericFunction, getattr(scipy.special, function_name))
    jax_function = cast(NumericFunction, getattr(jaxellip, function_name))

    input_values = inputs[function_name]
    # SciPy's native input type, for a fair comparison.
    scipy_input = np.asarray(input_values)

    jax_result = np.asarray(jax_function(input_values))
    scipy_result = np.asarray(scipy_function(scipy_input))

    # Compare where both are finite, using a single shared mask so the two arrays
    # stay aligned (both are NaN for m > 1).
    finite = np.isfinite(jax_result) & np.isfinite(scipy_result)
    assert jax_result[finite] == pytest.approx(
        scipy_result[finite],
        rel=RELATIVE_TOLERANCES_AGAINST_SCIPY[function_name],
        abs=0.0,
    )


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals_derivatives(function_name: str) -> None:
    jax_function = cast(
        Callable[[jax.Array], jax.Array], getattr(jaxellip, function_name)
    )

    input_values = derivative_inputs[function_name]

    # ——— analytic derivatives, computed from SciPy's K(m) and E(m) ———
    # ellipk(m) and ellipe(m) are differentiated with respect to m directly.
    # ellipkm1(x) = K(1 - x) is differentiated with respect to x, so by the
    # chain rule d/dx ellipkm1(x) = -dK/dm evaluated at m = 1 - x.
    m = np.asarray(input_values)
    if function_name == "ellipkm1":
        m = 1.0 - m
    K = scipy.special.ellipk(m)
    E = scipy.special.ellipe(m)

    # https://functions.wolfram.com/EllipticIntegrals/EllipticK/introductions/CompleteEllipticIntegrals/ShowAll.html
    # See "Representations of derivatives":
    #   dK/dm = (E - (1 - m) K) / (2 m (1 - m))
    #   dE/dm = (E - K) / (2 m)
    if function_name in ("ellipk", "ellipkm1"):
        dK_dm = (E - (1.0 - m) * K) / (2.0 * m * (1.0 - m))
        analytic_derivatives = -dK_dm if function_name == "ellipkm1" else dK_dm
    else:
        analytic_derivatives = (E - K) / (2.0 * m)

    jax_derivatives = np.asarray(jax.vmap(jax.grad(jax_function))(input_values))

    # Compare where both are finite, using a single shared mask so the two
    # arrays stay aligned.
    finite = np.isfinite(jax_derivatives) & np.isfinite(analytic_derivatives)

    assert jax_derivatives[finite] == pytest.approx(
        analytic_derivatives[finite],
        rel=RELATIVE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES,
        abs=ABSOLUTE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES,
    )


@pytest.mark.parametrize("function_name", functions)
def test_gradient_finite_out_of_domain(function_name: str) -> None:
    jax_function = cast(
        Callable[[jax.Array], jax.Array], getattr(jaxellip, function_name)
    )
    gradients = jax.vmap(jax.grad(jax_function))(out_of_domain_inputs[function_name])
    assert jnp.all(jnp.isfinite(gradients))
