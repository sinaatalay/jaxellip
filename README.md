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

## Elliptic integrals

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
