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

`jaxellip` evaluates the complete elliptic integrals through Carlson symmetric
forms:

- `ellipk(m) = R_F(0, 1 - m, 1)`
- `ellipe(m) = R_F(0, 1 - m, 1) - (m / 3) R_D(0, 1 - m, 1)`
- `ellipkm1(x) = K(1 - x) = R_F(0, x, 1)` away from the singular point

`R_F` and `R_D` are computed by a fixed-count Carlson duplication iteration using
14 steps. The fixed count keeps the implementation compatible with `jax.jit`,
`jax.vmap`, forward-mode AD, reverse-mode AD, and higher-order AD while driving
the iteration to the float64 round-off floor over the tested domain.

For very small positive `x`, `ellipkm1(x)` uses the logarithmic expansion

```text
K(1 - x) = 0.5 L + (x / 8) (L - 2),  L = log(16) - log(x)
```

for `x < 1e-8`. The switch point lies in the overlap where the series and direct
Carlson form agree to round-off. The log is written as `log(16) - log(x)` so AD
does not form an overflowing `1 / x**2` term for tiny `x`.

The Carlson iteration contains square roots at zero. Plain `sqrt(0)` has an
infinite derivative, which creates `0 * inf = NaN` in forward and higher-order
AD. Internally, `jaxellip` uses a custom-JVP square root whose derivative at zero
is defined as zero for the constant-zero Carlson argument.

`ellipe` uses a custom JVP with the cancellation-free identity

```text
dE/dm = -R_D(0, 1 - m, 1) / 6
```

This avoids the cancellation that appears when differentiating
`R_F - (m / 3) R_D` near `m = 1`. `ellipk` and `ellipkm1` use autodiff through
their value computations.

Boundary behavior matches the real-valued SciPy contract:

- `ellipk(1) = inf`, and `ellipk(m) = nan` for `m > 1`
- `ellipe(1) = 1`, and `ellipe(m) = nan` for `m > 1`
- `ellipkm1(0) = inf`, and `ellipkm1(x) = nan` for `x < 0`

Values at the poles are exact. The true derivative at the exact pole diverges;
the test suite checks derivatives by approaching the pole from the valid domain.
Out-of-domain values and gradients are `nan`, isolated per element in batched
JAX transforms.

The test suite compares values, reverse-mode gradients, forward-mode JVPs,
`jacfwd`/`jacrev`, and second derivatives against `mpmath` references over
domain-spanning grids, including large negative `m`, `m -> 1`, and tiny-to-huge
`ellipkm1` arguments.

Speed is part of the test suite (`tests/test_speed.py`): each case fails if
jaxellip is slower than a per-case multiple of SciPy on the same input (warm JIT,
minimum of several runs). The thresholds are tunable knobs in
`SPEED_RATIO_THRESHOLDS` -- loose ceilings, not targets, since wall-clock ratios
vary by machine. The checks carry the `speed` marker, so they can be skipped with
`uv run pytest -m "not speed"`.

## Developer Guide

Install:

1. [Visual Studio Code](https://code.visualstudio.com/)
2. [uv](https://docs.astral.sh/uv/)
3. [just](https://just.systems/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install just
```

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
