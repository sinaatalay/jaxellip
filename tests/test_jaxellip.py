import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

import jaxellip

jax.config.update("jax_enable_x64", True)

# Worst-case relative agreement with SciPy actually achieved across the tested
# range is ~2e-14 (ellipk, at m = -1e9), ~8e-16 (ellipkm1) and ~7e-15 (ellipe).
# No tested value crosses zero, so the comparison below is purely relative
# (abs=0): these tolerances are what is enforced, each with >=40x of margin.
RELATIVE_TOLERANCES_AGAINST_SCIPY = {
    "ellipk": 1e-12,
    "ellipkm1": 1e-13,
    "ellipe": 1e-12,
}
# jaxellip may be at most (1 + this) times as slow as SciPy.
TIME_TOLERANCE_AGAINST_SCIPY = 1
NUMBER_OF_TIMING_REPEATS = 9

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


def time_it(function, inputs):
    # jax.block_until_ready forces JAX's asynchronous dispatch to finish, so we
    # time the real computation rather than just the dispatch; it is a no-op for
    # SciPy's NumPy results. The minimum over several repeats is the runtime
    # estimate least sensitive to scheduling noise (noise only adds time).
    best_time = float("inf")
    result = None
    for _ in range(NUMBER_OF_TIMING_REPEATS):
        start_time = time.perf_counter()
        result = jax.block_until_ready(function(inputs))
        best_time = min(best_time, time.perf_counter() - start_time)
    return best_time, result


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals(function_name):
    scipy_function = getattr(scipy.special, function_name)
    jax_function = getattr(jaxellip, function_name)

    input = inputs[function_name]
    scipy_input = np.asarray(input)  # SciPy's native input, for a fair comparison

    scipy_time, scipy_result = time_it(scipy_function, scipy_input)

    jax_function(input)  # trigger the one-time JIT compilation before timing

    jax_time, jax_result = time_it(jax_function, input)
    jax_result = np.asarray(jax_result)

    # Compare where both are finite, using a single shared mask so the two arrays
    # stay aligned (both are NaN for m > 1).
    finite = np.isfinite(jax_result) & np.isfinite(scipy_result)
    assert jax_result[finite] == pytest.approx(
        scipy_result[finite],
        rel=RELATIVE_TOLERANCES_AGAINST_SCIPY[function_name],
        abs=0.0,
    )

    # Only fail if jaxellip is too slow; being faster than SciPy is fine.
    assert jax_time < (1 + TIME_TOLERANCE_AGAINST_SCIPY) * scipy_time


@pytest.mark.parametrize("function_name", functions)
def test_elliptic_integrals_derivatives(function_name):
    jax_function = getattr(jaxellip, function_name)

    input = derivative_inputs[function_name]

    # ——— analytic derivatives, computed from SciPy's K(m) and E(m) ———
    # ellipk(m) and ellipe(m) are differentiated with respect to m directly.
    # ellipkm1(x) = K(1 - x) is differentiated with respect to x, so by the
    # chain rule d/dx ellipkm1(x) = -dK/dm evaluated at m = 1 - x.
    m = np.asarray(input)
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

    jax_derivatives = np.asarray(jax.vmap(jax.grad(jax_function))(input))

    # Compare where both are finite, using a single shared mask so the two
    # arrays stay aligned.
    finite = np.isfinite(jax_derivatives) & np.isfinite(analytic_derivatives)

    assert jax_derivatives[finite] == pytest.approx(
        analytic_derivatives[finite],
        rel=RELATIVE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES,
        abs=ABSOLUTE_TOLERANCE_AGAINST_SCIPY_ANALYTIC_DERIVATIVES,
    )
