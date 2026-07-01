# `jaxellip`: JAX implementations of some elliptic integrals


[![test](https://github.com/sinaatalay/jaxellip/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sinaatalay/jaxellip/actions/workflows/test.yaml)
[![pypi-version](<https://img.shields.io/pypi/v/jaxellip?label=PyPI%20version&color=rgb(0%2C79%2C144)>)](https://pypi.python.org/pypi/jaxellip)
[![pypi-downloads](<https://img.shields.io/pepy/dt/jaxellip?label=PyPI%20downloads&color=rgb(0%2C%2079%2C%20144)>)](https://pypistats.org/packages/jaxellip)

Elliptic integrals are available in `scipy.special`, but are not implemented in `jax.scipy.special`, making them neither differentiable nor compatible with JAX’s JIT compilation. This package provides JAX-compatible implementations of several elliptic integrals from `scipy.special`: `ellipk`, `ellipkm1`, and `ellipe`. The results are tested against arbitrary-precision `mpmath` references and cross-checked against `scipy.special` (see [tests](https://github.com/sinaatalay/jaxellip/tree/main/tests/test_jaxellip.py)).

## Usage

1. Install `jaxellip`.

```bash
pip install jaxellip
```

2. Import `jaxellip` in your Python code and use it.

```python
import jaxellip
import jax.numpy as jnp

input = jnp.linspace(-10, 10, 1000)

jaxellip.ellipk(input) # Complete elliptic integral of the first kind
jaxellip.ellipkm1(input) # Complete elliptic integral of the first kind around m = 1
jaxellip.ellipe(input) # Complete elliptic integral of the second kind
```

The elliptic integrals in `jaxellip` follow the same parameter conventions as `scipy.special`. In particular, the input parameter $m$ is the elliptic parameter, defined as $m = k^2$, where $k$ is the modulus. This differs from some literature or libraries (e.g., Boost or Abramowitz & Stegun), where the integrals are sometimes expressed in terms of $k$ directly.

The function `ellipkm1(m)` computes the complete elliptic integral of the first kind with argument $1 - m$, i.e., $K(1 - m)$. This is particularly useful for improved numerical stability and precision when $m$ is close to 1, since $K(m)$ diverges logarithmically as $m \to 1$.

### Complete elliptic integral of the first kind (`ellipk` and `ellipkm1`)

$$
K(m) = \int_0^{\pi/2} \frac{1}{\sqrt{1 - m \sin^2 \theta}}  d\theta
$$

### Complete elliptic integral of the second kind (`ellipe`)

$$
E(m) = \int_0^{\pi/2} \sqrt{1 - m \sin^2 \theta}  d\theta
$$

## Numerical Method

`ellipk` and `ellipe` use the same Cephes minimax polynomials as SciPy: a degree-10 polynomial in the complementary parameter $1 - m$ plus a logarithmic term, with negative $m$ reduced to $[0, 1)$ by the Landen transformation. `ellipkm1(x)` evaluates $K(1 - x)$ from $x$ directly (never forming $1 - x$) and uses a logarithmic series for $x < 10^{-8}$. The forward pass does not iterate.

Derivatives use exact custom JVP rules instead of differentiating the polynomial. Each reduces to the Carlson symmetric integral $R_D$ (for example, $dK/dm = R_D(0, 1, 1-m)/6$), evaluated by a short fixed iteration whose only zero argument is handled in closed form. This makes `jaxellip` differentiable in every JAX mode (forward, reverse, and higher order such as `jax.hessian`), so it stays correct inside a larger function that is differentiated as a whole.

Edge cases match SciPy: `ellipk(1)` is `inf`, `ellipe(1)` is `1`, `ellipkm1(0)` is `inf`, and out-of-domain inputs (`m > 1`, or `x < 0`) return `NaN` for both value and gradient.

## Comparison against SciPy

`jaxellip` is tested against `mpmath` (a high-precision reference) and `scipy.special`:

- **Accuracy.** Values agree with both to about `1e-14` relative. Derivatives, in forward, reverse, and second-order modes, agree with `mpmath`'s analytic derivatives to about `1e-13` (first order) and `1e-15` (second order). This holds over the whole domain: `m` down to `-1e308`, up to the `m -> 1` singularity, and `ellipkm1` for `x` from `1e-308` to `1e308`.
- **Speed.** Warm CPU runtimes are comparable to `scipy.special` in the tested cases. Exact ratios depend on the device, runner load, JAX/XLA version, and SciPy build, so the test suite checks only that there are no large performance regressions.

## Developer Guide

Install:

1. [Visual Studio Code](https://code.visualstudio.com/)
2. [uv](https://docs.astral.sh/uv/)
3. [just](https://just.systems/)

Clone and set up:

```bash
git clone https://github.com/sinaatalay/jaxellip.git
cd jaxellip
just sync
code .
```

Use `.venv` as the Python interpreter in VS Code.

Repository layout:

- `src/jaxellip/`: package code
- `tests/`: tests against SciPy and JAX autodiff
- `pyproject.toml`: package metadata and tool settings
- `uv.lock`: locked dependency versions
- `justfile`: development commands

Common commands:

```bash
just test    # run tests
just check   # run all checks
just format  # format code
```
