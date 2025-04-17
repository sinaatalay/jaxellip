import time

import jax
import jax.numpy as jnp
import pytest
import scipy.special

import jaxellip

functions = ["ellipk", "ellipkm1", "ellipe"]
inputs = jnp.linspace(-10, 10, 1000)


def time_it(function, inputs):
    start_time = time.perf_counter()
    result = function(inputs)
    return time.perf_counter() - start_time, result


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals(function_name):
    relative_tolerance = 1e-4
    time_tolerance = 1.5  # It's okay to be 50% slower

    scipy_function = getattr(scipy.special, function_name)
    jax_function = getattr(jaxellip, function_name)

    scipy_time, scipy_result = time_it(scipy_function, inputs)

    jax_function(inputs)  # to JIT

    jax_time, jax_result = time_it(jax_function, inputs)

    # Drop nan values
    jax_result = jax_result[~jnp.isnan(jax_result)]
    scipy_result = scipy_result[~jnp.isnan(scipy_result)]

    assert jax_result == pytest.approx(scipy_result, rel=relative_tolerance)
    assert jax_time < scipy_time * time_tolerance


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals_derivatives(function_name):
    relative_tolerance = 3e-3

    scipy_function = getattr(scipy.special, function_name)
    jax_function = getattr(jaxellip, function_name)

    epsilon = 7e-4
    scipy_finite_difference_derivatives = (
        scipy_function(inputs + epsilon) - scipy_function(inputs - epsilon)
    ) / (2 * epsilon)

    jax_derivatives = jax.vmap(jax.grad(jax_function))(inputs)

    # Drop nan values
    jax_derivatives = jax_derivatives[~jnp.isnan(jax_derivatives)]
    scipy_derivatives = scipy_finite_difference_derivatives[
        ~jnp.isnan(scipy_finite_difference_derivatives)
    ]

    assert jax_derivatives == pytest.approx(scipy_derivatives, rel=relative_tolerance)
