"""Compare warm CPU runtimes against scipy.special."""

import math
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

TIMING_REPEATS = 9

_BROAD = jnp.concatenate(
    [-jnp.geomspace(1e-30, 1e308, 100000), jnp.linspace(0, 1.1, 100000)]
)
_LARGE_NEGATIVE = -jnp.geomspace(1e12, 1e308, 100000)
_SMALL_X = jnp.geomspace(1e-308, 1e-7, 100000)
_LARGE_X = jnp.geomspace(1.0, 1e308, 100000)

BENCHMARK_CASES: list[tuple[str, str, jax.Array]] = [
    ("ellipk broad", "ellipk", _BROAD),
    ("ellipe broad", "ellipe", _BROAD),
    ("ellipk large-|m|", "ellipk", _LARGE_NEGATIVE),
    ("ellipe large-|m|", "ellipe", _LARGE_NEGATIVE),
    ("ellipkm1 small-x", "ellipkm1", _SMALL_X),
    ("ellipkm1 large-x", "ellipkm1", _LARGE_X),
]


def _min_runtime(call: Callable[[], object]) -> float:
    best = math.inf
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _benchmark_case(label: str, function_name: str, values: jax.Array) -> str:
    jax_function = cast(NumericFunction, getattr(jaxellip, function_name))
    scipy_function = cast(NumericFunction, getattr(scipy.special, function_name))
    scipy_input = np.asarray(values)

    jax.block_until_ready(jax_function(values))

    jax_time = _min_runtime(lambda: jax.block_until_ready(jax_function(values)))
    scipy_time = _min_runtime(lambda: scipy_function(scipy_input))
    ratio = jax_time / scipy_time

    return (
        f"{label}: jaxellip {jax_time:.6f}s, "
        f"scipy {scipy_time:.6f}s, ratio {ratio:.2f}x"
    )


def main() -> None:
    for label, function_name, values in BENCHMARK_CASES:
        print(_benchmark_case(label, function_name, values))  # noqa: T201


if __name__ == "__main__":
    main()
