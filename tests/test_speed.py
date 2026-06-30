"""Speed checks: jaxellip must stay within a tunable multiple of SciPy.

These replace the old ``benchmarks/`` script -- speed is part of the test suite.
Unlike correctness, wall-clock ratios vary with machine and load, so the
thresholds below are deliberately loose ceilings (knobs to tighten as the
implementation is optimized), not targets. Each timing is the minimum of several
warm runs (noise only adds time). The whole module carries the ``speed`` marker,
so the checks can be skipped with ``pytest -m "not speed"``.
"""

import math
import time
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

# ---- Knobs: a case fails if jaxellip is slower than this multiple of SciPy. ----
# These are failure ceilings, not goals. Current warm-CPU ratios sit well below
# each value; tighten them as the implementation gets faster.
SPEED_RATIO_THRESHOLDS: dict[str, float] = {
    "ellipk broad": 2.0,
    "ellipe broad": 2.0,
    "ellipk large-|m|": 2.0,
    "ellipe large-|m|": 2.0,
    "ellipkm1 small-x": 2.0,
    "ellipkm1 large-x": 2.0,
}
TIMING_REPEATS = 9

_BROAD = jnp.concatenate(
    [-jnp.geomspace(1e-30, 1e308, 100000), jnp.linspace(0, 1.1, 100000)]
)
_LARGE_NEGATIVE = -jnp.geomspace(1e12, 1e308, 100000)
_SMALL_X = jnp.geomspace(1e-308, 1e-7, 100000)
_LARGE_X = jnp.geomspace(1.0, 1e308, 100000)

SPEED_CASES: list[tuple[str, str, jax.Array]] = [
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


@pytest.mark.speed
@pytest.mark.parametrize(
    ("label", "function_name", "values"),
    SPEED_CASES,
    ids=[case[0] for case in SPEED_CASES],
)
def test_speed_within_threshold(
    label: str, function_name: str, values: jax.Array
) -> None:
    jax_function = cast(NumericFunction, getattr(jaxellip, function_name))
    scipy_function = cast(NumericFunction, getattr(scipy.special, function_name))
    scipy_input = np.asarray(values)

    # block_until_ready forces JAX's async dispatch to finish so we time the real
    # computation; the first call also triggers the one-time JIT compilation.
    jax.block_until_ready(jax_function(values))

    jax_time = _min_runtime(lambda: jax.block_until_ready(jax_function(values)))
    scipy_time = _min_runtime(lambda: scipy_function(scipy_input))
    ratio = jax_time / scipy_time

    threshold = SPEED_RATIO_THRESHOLDS[label]
    assert ratio < threshold, (
        f"{label}: jaxellip is {ratio:.2f}x SciPy, over the {threshold:.1f}x limit"
    )
