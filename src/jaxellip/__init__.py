from importlib.metadata import version

__version__ = version("jaxellip")

from .elliptic_functions_and_integrals import (
    ellipe,
    ellipk,
    ellipkm1,
)

__all__ = [
    "ellipe",
    "ellipk",
    "ellipkm1",
]
