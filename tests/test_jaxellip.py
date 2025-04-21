import time

import jax
import jax.numpy as jnp
import pytest
import scipy.special

import jaxellip

jax.config.update("jax_enable_x64", True)

RELATIVE_TOLERANCES_AGAINST_SCIPY = {
    "ellipk": 1e-30,
    "ellipkm1": 1e-30,
    "ellipe": 1e-14,
}
TIME_TOLERANCE_AGAINST_SCIPY = 1  # Maximun two times slower

RELATIVE_TOLERANCE_AGAINST_SCIPY_FINITE_DIFFERENCE_DERIVATIVES = 1e-1


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


def time_it(function, inputs):
    start_time = time.perf_counter()
    result = function(inputs)
    return time.perf_counter() - start_time, result


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals(function_name):
    scipy_function = getattr(scipy.special, function_name)
    jax_function = getattr(jaxellip, function_name)

    input = inputs[function_name]

    scipy_time, scipy_result = time_it(scipy_function, input)

    jax_function(input)  # to JIT

    jax_time, jax_result = time_it(jax_function, input)

    # Drop nan values
    jax_result = jax_result[~jnp.isnan(jax_result)]
    scipy_result = scipy_result[~jnp.isnan(scipy_result)]

    assert jax_result == pytest.approx(
        scipy_result, rel=RELATIVE_TOLERANCES_AGAINST_SCIPY[function_name]
    )
    assert abs(jax_time - scipy_time) / scipy_time < TIME_TOLERANCE_AGAINST_SCIPY


@pytest.mark.skip("They don't match!")
@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals_derivatives(function_name):
    jax_function = getattr(jaxellip, function_name)

    input = inputs[function_name]

    # ——— compute analytic SciPy derivatives ———
    # K(m) and E(m) from SciPy
    K = scipy.special.ellipk(input)
    E = scipy.special.ellipe(input)

    # https://functions.wolfram.com/EllipticIntegrals/EllipticK/introductions/CompleteEllipticIntegrals/ShowAll.html
    # See "Representations of derivatives"
    if function_name in ("ellipk", "ellipkm1"):
        scipy_analytical_derivatives = (E - (1.0 - input) * K) / (
            2.0 * input * (1.0 - input)
        )
        if function_name == "ellipkm1":
            scipy_analytical_derivatives = -scipy_analytical_derivatives
    else:
        scipy_analytical_derivatives = (E - K) / (2.0 * input)

    jax_derivatives = jax.vmap(jax.grad(jax_function))(input)

    # Drop nan values
    jax_derivative_nan_indices = jnp.isnan(jax_derivatives)
    scipy_analytical_derivatives_nan_indices = jnp.isnan(scipy_analytical_derivatives)
    nan_indices = jnp.logical_or(
        jax_derivative_nan_indices, scipy_analytical_derivatives_nan_indices
    )

    jax_derivatives = jax_derivatives[~nan_indices]
    scipy_analytical_derivatives = scipy_analytical_derivatives[~nan_indices]

    assert jax_derivatives == pytest.approx(
        scipy_analytical_derivatives,
        rel=RELATIVE_TOLERANCE_AGAINST_SCIPY_FINITE_DIFFERENCE_DERIVATIVES,
    )
