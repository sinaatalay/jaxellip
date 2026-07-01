from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import scipy.special

import jaxellip

jax.config.update("jax_enable_x64", True)

type ScalarFunction = Callable[[jax.Array], jax.Array]
type ReferenceFunction = Callable[[float], float]

VALUE_RTOL = 1.09e-14
SCIPY_RTOL = 1.09e-14
DERIVATIVE_RTOL = 7.21e-14
DERIVATIVE_ATOL = 1e-300
HESSIAN_RTOL = 2.3e-15
HESSIAN_ATOL = 1e-300


M_VALUE_GRID = np.concatenate(
    [
        -np.geomspace(1e-30, 1e308, 120),
        np.linspace(0.0, 1.0, 120, endpoint=False),
        1.0 - np.geomspace(1e-16, 1e-1, 120),
    ]
).astype(np.float64)

KM1_VALUE_GRID = np.geomspace(1e-308, 1e308, 240).astype(np.float64)

M_DERIVATIVE_GRID = np.concatenate(
    [
        -np.geomspace(1e-12, 1e100, 40),
        np.array([0.0], dtype=np.float64),
        np.linspace(1e-12, 0.9, 40),
        1.0 - np.geomspace(1e-12, 1e-2, 40),
    ]
).astype(np.float64)

KM1_DERIVATIVE_GRID = np.concatenate(
    [
        np.geomspace(1e-300, 1e-8, 40),
        np.geomspace(1e-8, 1e100, 80),
    ]
).astype(np.float64)

M_HESSIAN_GRID = np.array(
    [
        -1e20,
        -1e6,
        -1.0,
        -1e-6,
        0.0,
        1e-6,
        0.25,
        0.75,
        1.0 - 1e-6,
        1.0 - 1e-10,
    ],
    dtype=np.float64,
)

# ellipkm1 second derivative blows up as x -> 0 (m -> 1) and overflows for tiny
# x, so the second-order grid stays in the well-conditioned interior.
KM1_HESSIAN_GRID = np.geomspace(1e-3, 0.99, 25).astype(np.float64)


def _mp_dps_for(value: float) -> int:
    if value == 0.0:
        return 80
    return max(80, int(abs(np.log10(abs(value)))) + 40)


def _ref_k(m: float) -> float:
    with mp.workdps(80):
        return float(mp.ellipk(mp.mpf(float(m))))


def _ref_e(m: float) -> float:
    with mp.workdps(80):
        return float(mp.ellipe(mp.mpf(float(m))))


def _ref_km1(x: float) -> float:
    with mp.workdps(_mp_dps_for(x)):
        return float(mp.elliprf(0, mp.mpf(float(x)), 1))


def _ref_dk(m: float) -> float:
    with mp.workdps(_mp_dps_for(m)):
        return float(mp.diff(mp.ellipk, mp.mpf(float(m))))


def _ref_de(m: float) -> float:
    with mp.workdps(_mp_dps_for(m)):
        return float(mp.diff(mp.ellipe, mp.mpf(float(m))))


def _ref_dkm1(x: float) -> float:
    with mp.workdps(_mp_dps_for(x)):
        x_mp = mp.mpf(float(x))
        return float(mp.diff(lambda t: mp.elliprf(0, t, 1), x_mp))


def _ref_d2k(m: float) -> float:
    with mp.workdps(max(300, _mp_dps_for(m) + 180)):
        return float(mp.diff(mp.ellipk, mp.mpf(float(m)), 2))


def _ref_d2e(m: float) -> float:
    with mp.workdps(max(300, _mp_dps_for(m) + 180)):
        return float(mp.diff(mp.ellipe, mp.mpf(float(m)), 2))


def _ref_d2km1(x: float) -> float:
    # d^2/dx^2 K(1 - x) = K''(1 - x).
    with mp.workdps(max(120, _mp_dps_for(x) + 60)):
        return float(mp.diff(mp.ellipk, 1.0 - mp.mpf(float(x)), 2))


def _reference_array(values: np.ndarray, reference: ReferenceFunction) -> np.ndarray:
    return np.asarray([reference(float(value)) for value in values], dtype=np.float64)


def _jax_function(function_name: str) -> ScalarFunction:
    return cast(ScalarFunction, getattr(jaxellip, function_name))


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float = 0.0,
) -> None:
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)


@pytest.mark.parametrize(
    ("function_name", "values", "reference"),
    [
        ("ellipk", M_VALUE_GRID, _ref_k),
        ("ellipe", M_VALUE_GRID, _ref_e),
        ("ellipkm1", KM1_VALUE_GRID, _ref_km1),
    ],
)
def test_value_vs_mpmath(
    function_name: str, values: np.ndarray, reference: ReferenceFunction
) -> None:
    jax_function = _jax_function(function_name)

    actual = np.asarray(jax_function(jnp.asarray(values)))
    expected = _reference_array(values, reference)

    _assert_close(actual, expected, rtol=VALUE_RTOL)


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("ellipk", np.concatenate([M_VALUE_GRID, np.array([1.0, 2.0])])),
        ("ellipe", np.concatenate([M_VALUE_GRID, np.array([1.0, 2.0])])),
        ("ellipkm1", np.concatenate([KM1_VALUE_GRID, np.array([0.0, -1.0])])),
    ],
)
def test_value_vs_scipy(function_name: str, values: np.ndarray) -> None:
    jax_function = _jax_function(function_name)
    scipy_function = cast(
        Callable[[np.ndarray], np.ndarray], getattr(scipy.special, function_name)
    )

    actual = np.asarray(jax_function(jnp.asarray(values)))
    expected = np.asarray(scipy_function(values))
    finite = np.isfinite(actual) & np.isfinite(expected)

    _assert_close(actual[finite], expected[finite], rtol=SCIPY_RTOL)


@pytest.mark.parametrize(
    ("function_name", "values", "reference"),
    [
        ("ellipk", M_DERIVATIVE_GRID, _ref_dk),
        ("ellipe", M_DERIVATIVE_GRID, _ref_de),
        ("ellipkm1", KM1_DERIVATIVE_GRID, _ref_dkm1),
    ],
)
def test_grad_reverse_vs_mpmath(
    function_name: str, values: np.ndarray, reference: ReferenceFunction
) -> None:
    jax_function = _jax_function(function_name)
    grad_function = jax.vmap(jax.grad(jax_function))

    actual = np.asarray(grad_function(jnp.asarray(values)))
    expected = _reference_array(values, reference)

    _assert_close(actual, expected, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


@pytest.mark.parametrize(
    ("function_name", "values", "reference"),
    [
        ("ellipk", M_DERIVATIVE_GRID, _ref_dk),
        ("ellipe", M_DERIVATIVE_GRID, _ref_de),
        ("ellipkm1", KM1_DERIVATIVE_GRID, _ref_dkm1),
    ],
)
def test_grad_forward_vs_mpmath(
    function_name: str, values: np.ndarray, reference: ReferenceFunction
) -> None:
    jax_function = _jax_function(function_name)

    def jvp_one(value: jax.Array) -> jax.Array:
        _, tangent = jax.jvp(jax_function, (value,), (jnp.ones_like(value),))
        return tangent

    actual = np.asarray(jax.vmap(jvp_one)(jnp.asarray(values)))
    expected = _reference_array(values, reference)

    _assert_close(actual, expected, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("ellipk", M_DERIVATIVE_GRID),
        ("ellipe", M_DERIVATIVE_GRID),
        ("ellipkm1", KM1_DERIVATIVE_GRID),
    ],
)
def test_forward_equals_reverse(function_name: str, values: np.ndarray) -> None:
    jax_function = _jax_function(function_name)

    def jvp_one(value: jax.Array) -> jax.Array:
        _, tangent = jax.jvp(jax_function, (value,), (jnp.ones_like(value),))
        return tangent

    forward = np.asarray(jax.vmap(jvp_one)(jnp.asarray(values)))
    reverse = np.asarray(jax.vmap(jax.grad(jax_function))(jnp.asarray(values)))

    _assert_close(forward, reverse, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


_SECOND_ORDER_CASES = [
    ("ellipk", M_HESSIAN_GRID, _ref_d2k),
    ("ellipe", M_HESSIAN_GRID, _ref_d2e),
    ("ellipkm1", KM1_HESSIAN_GRID, _ref_d2km1),
]


@pytest.mark.parametrize(("function_name", "values", "reference"), _SECOND_ORDER_CASES)
def test_hessian_vs_mpmath(
    function_name: str, values: np.ndarray, reference: ReferenceFunction
) -> None:
    # jax.hessian is jacfwd(jacrev(...)) -- the idiomatic second-order API and the
    # forward-over-reverse path. It must be finite and correct, not just the
    # reverse-over-reverse grad(grad).
    jax_function = _jax_function(function_name)

    actual = np.asarray(jax.vmap(jax.hessian(jax_function))(jnp.asarray(values)))
    expected = _reference_array(values, reference)

    _assert_close(actual, expected, rtol=HESSIAN_RTOL, atol=HESSIAN_ATOL)


@pytest.mark.parametrize(("function_name", "values", "reference"), _SECOND_ORDER_CASES)
def test_forward_second_order_vs_mpmath(
    function_name: str, values: np.ndarray, reference: ReferenceFunction
) -> None:
    # Pure forward-over-forward second order (jacfwd(jacfwd)). This is the mode the
    # zero-first-argument Carlson iteration used to turn into NaN; it must agree
    # with the analytic second derivative.
    jax_function = _jax_function(function_name)

    actual = np.asarray(
        jax.vmap(jax.jacfwd(jax.jacfwd(jax_function)))(jnp.asarray(values))
    )
    expected = _reference_array(values, reference)

    _assert_close(actual, expected, rtol=HESSIAN_RTOL, atol=HESSIAN_ATOL)


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("ellipk", np.array([-1e100, -1.0, 0.0, 0.5, 1.0 - 1e-10])),
        ("ellipe", np.array([-1e100, -1.0, 0.0, 0.5, 1.0 - 1e-10])),
        ("ellipkm1", np.array([1e-300, 1e-8, 0.5, 1.0, 1e100])),
    ],
)
def test_jacfwd_matches_jacrev(function_name: str, values: np.ndarray) -> None:
    jax_function = _jax_function(function_name)

    forward = np.asarray(jax.vmap(jax.jacfwd(jax_function))(jnp.asarray(values)))
    reverse = np.asarray(jax.vmap(jax.jacrev(jax_function))(jnp.asarray(values)))

    _assert_close(forward, reverse, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


def test_boundary_values() -> None:
    assert np.isposinf(np.asarray(jaxellip.ellipk(jnp.asarray(1.0))))
    assert np.isnan(np.asarray(jaxellip.ellipk(jnp.asarray(2.0))))
    assert np.asarray(jaxellip.ellipe(jnp.asarray(1.0))) == pytest.approx(1.0)
    assert np.isnan(np.asarray(jaxellip.ellipe(jnp.asarray(2.0))))
    assert np.isposinf(np.asarray(jaxellip.ellipkm1(jnp.asarray(0.0))))
    assert np.isnan(np.asarray(jaxellip.ellipkm1(jnp.asarray(-1.0))))


@pytest.mark.parametrize(
    ("function_name", "values", "valid_mask"),
    [
        (
            "ellipk",
            np.array([-10.0, 0.0, 0.5, 1.25, 2.0], dtype=np.float64),
            np.array([True, True, True, False, False]),
        ),
        (
            "ellipe",
            np.array([-10.0, 0.0, 0.5, 1.25, 2.0], dtype=np.float64),
            np.array([True, True, True, False, False]),
        ),
        (
            "ellipkm1",
            np.array([1e-12, 0.5, 10.0, -1e-12, -1.0], dtype=np.float64),
            np.array([True, True, True, False, False]),
        ),
    ],
)
def test_out_of_domain_gradient_isolated(
    function_name: str, values: np.ndarray, valid_mask: np.ndarray
) -> None:
    jax_function = _jax_function(function_name)

    gradients = np.asarray(jax.vmap(jax.grad(jax_function))(jnp.asarray(values)))

    assert np.all(np.isfinite(gradients[valid_mask]))
    assert np.all(np.isnan(gradients[~valid_mask]))


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("ellipk", np.array([-1e100, -1.0, 0.0, 0.5, 1.0 - 1e-10])),
        ("ellipe", np.array([-1e100, -1.0, 0.0, 0.5, 1.0 - 1e-10])),
        ("ellipkm1", np.array([1e-300, 1e-8, 0.5, 1.0, 1e100])),
    ],
)
def test_no_nan_in_domain(function_name: str, values: np.ndarray) -> None:
    jax_function = _jax_function(function_name)

    def jvp_one(value: jax.Array) -> jax.Array:
        _, tangent = jax.jvp(jax_function, (value,), (jnp.ones_like(value),))
        return tangent

    value_result = np.asarray(jax_function(jnp.asarray(values)))
    reverse = np.asarray(jax.vmap(jax.grad(jax_function))(jnp.asarray(values)))
    forward = np.asarray(jax.vmap(jvp_one)(jnp.asarray(values)))

    assert np.all(np.isfinite(value_result))
    assert np.all(np.isfinite(reverse))
    assert np.all(np.isfinite(forward))

    if function_name != "ellipkm1":
        hessian = np.asarray(
            jax.vmap(jax.grad(jax.grad(jax_function)))(jnp.asarray(values))
        )
        assert np.all(np.isfinite(hessian))


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("ellipk", np.array([-1e12, -1.0, 0.0, 0.5, 1.0 - 1e-8])),
        ("ellipe", np.array([-1e12, -1.0, 0.0, 0.5, 1.0 - 1e-8])),
        ("ellipkm1", np.array([1e-100, 1e-8, 0.5, 1.0, 1e12])),
    ],
)
def test_jit_and_vmap_compose(function_name: str, values: np.ndarray) -> None:
    jax_function = _jax_function(function_name)
    input_values = jnp.asarray(values)

    vmapped = jax.vmap(jax_function)
    jitted_vmapped = jax.jit(vmapped)
    vmapped_grad = jax.vmap(jax.grad(jax_function))
    jitted_vmapped_grad = jax.jit(vmapped_grad)

    _assert_close(
        np.asarray(jitted_vmapped(input_values)),
        np.asarray(vmapped(input_values)),
        rtol=VALUE_RTOL,
    )
    _assert_close(
        np.asarray(jitted_vmapped_grad(input_values)),
        np.asarray(vmapped_grad(input_values)),
        rtol=DERIVATIVE_RTOL,
        atol=DERIVATIVE_ATOL,
    )
