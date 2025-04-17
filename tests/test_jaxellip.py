import time

import jax.numpy as jnp
import pytest
import scipy.special

import jaxellip

functions = ["ellipk", "ellipkm1", "ellipe"]
inputs = jnp.linspace(-10, 10, 1000)
relative_tolerance = 1e-4
time_tolerance = 1.5  # It's okay to be 50% slower


def time_it(function, inputs):
    start_time = time.perf_counter()
    result = function(inputs)
    return time.perf_counter() - start_time, result


@pytest.mark.parametrize("function_name", functions)
def test_jaxellip_single_input(function_name):
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
