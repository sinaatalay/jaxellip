"""Benchmark jaxellip against SciPy. Run with ``just benchmark``.

This is a measurement, not a gate. Wall-clock comparisons are too hardware- and
load-dependent to assert on in CI without producing false failures, so the
correctness suite no longer times anything. The target is that jaxellip stays
within about 2x of SciPy once the JIT cache is warm; the table below lets a
human judge that.
"""

import time
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import jaxellip

jax.config.update("jax_enable_x64", True)

type NumericArray = jax.Array | np.ndarray
type NumericFunction = Callable[[NumericArray], NumericArray]

NUMBER_OF_TIMING_REPEATS = 9

functions = ["ellipk", "ellipkm1", "ellipe"]
a_lot_of_numbers = jnp.concatenate(
    [jnp.geomspace(-1e9, -1e-30, 100000), jnp.linspace(0, 1.1, 100000)]
)
inputs: dict[str, jax.Array] = {
    "ellipk": a_lot_of_numbers,
    "ellipkm1": jnp.linspace(0.5, 1.0, 100000),
    "ellipe": a_lot_of_numbers,
}


def time_it(function: NumericFunction, input_values: NumericArray) -> float:
    # jax.block_until_ready forces JAX's asynchronous dispatch to finish, so we
    # time the real computation rather than just the dispatch; it is a no-op for
    # SciPy's NumPy results. The minimum over several repeats is the runtime
    # estimate least sensitive to scheduling noise (noise only adds time).
    start_time = time.perf_counter()
    jax.block_until_ready(function(input_values))
    best_time = time.perf_counter() - start_time

    for _ in range(NUMBER_OF_TIMING_REPEATS - 1):
        start_time = time.perf_counter()
        jax.block_until_ready(function(input_values))
        best_time = min(best_time, time.perf_counter() - start_time)

    return best_time


def main() -> None:
    header = f"{'function':<10}{'SciPy (s)':>14}{'jaxellip (s)':>14}{'ratio':>8}"
    print(header)
    print("-" * len(header))

    for function_name in functions:
        scipy_function = cast(NumericFunction, getattr(scipy.special, function_name))
        jax_function = cast(NumericFunction, getattr(jaxellip, function_name))

        input_values = inputs[function_name]
        scipy_input = np.asarray(input_values)

        jax_function(input_values)  # trigger the one-time JIT compilation

        scipy_time = time_it(scipy_function, scipy_input)
        jax_time = time_it(jax_function, input_values)

        print(
            f"{function_name:<10}{scipy_time:>14.6f}{jax_time:>14.6f}"
            f"{jax_time / scipy_time:>8.2f}"
        )


if __name__ == "__main__":
    main()
