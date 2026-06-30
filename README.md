# `jaxellip`: JAX implementations of some elliptic integrals


[![test](https://github.com/sinaatalay/jaxellip/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sinaatalay/jaxellip/actions/workflows/test.yaml)
[![coverage](https://coverage-badge.samuelcolvin.workers.dev/sinaatalay/jaxellip.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/sinaatalay/jaxellip)
[![pypi-version](<https://img.shields.io/pypi/v/jaxellip?label=PyPI%20version&color=rgb(0%2C79%2C144)>)](https://pypi.python.org/pypi/jaxellip)
[![pypi-downloads](<https://img.shields.io/pepy/dt/jaxellip?label=PyPI%20downloads&color=rgb(0%2C%2079%2C%20144)>)](https://pypistats.org/packages/jaxellip)

Elliptic integrals are available in `scipy.special`, but are not implemented in `jax.scipy.special`, making them neither differentiable nor compatible with JAX’s JIT compilation. This package provides JAX-compatible implementations of several elliptic integrals from `scipy.special`: `ellipk`, `ellipkm1`, and `ellipe`. The results have been tested against `scipy.special` to ensure accuracy and performance (see [tests](https://github.com/sinaatalay/jaxellip/tree/main/tests/test_jaxellip.py)).

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

## Developer Guide

The repository uses [uv](https://docs.astral.sh/uv/) for Python packaging and dependency management, [just](https://just.systems/) for project commands, [prek](https://prek.j178.dev/) for hooks, Ruff for formatting and linting, and ty for type checking. The package source lives in `src/jaxellip`.

1. Install uv and just.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install just
```

2. Clone the repository and enter it.

```bash
git clone https://github.com/sinaatalay/jaxellip.git
cd jaxellip
```

3. Sync the locked development environment.

```bash
just sync
```

4. Install the local hooks.

```bash
uv run --frozen prek install
```

Use `.venv` as the interpreter in your editor. All developer commands should go through `just`.

### Commands

Format the code with Ruff:
```bash
just format
```

Lint the code with `ruff`:
```bash
just lint
```

Check the types with ty:
```bash
just check-types
```

Run spell checking with typos:
```bash
just spell
```

Run the prek hooks:
```bash
just prek
```

Run the full static check:
```bash
just check
```

Run the tests:
```bash
just test
```

Run the tests and generate a coverage report:
```bash
just test-and-report
```

Build the source distribution and wheel:
```bash
just build
```

### Release

The package version is stored in `pyproject.toml`. To prepare a release, update the version with uv, regenerate the lockfile, and tag the release with a leading `v`.

```bash
uv version 0.1.0
just lock
git tag -a v0.1.0 -m v0.1.0
```
