import jax.numpy as jnp

import datasyn.mathutils.dft as dftlib
import datasyn.mathutils.geometry as geolib
from datasyn.jaxutils.typing import *


def make_sensor_dftlayout(Nx: int, Ny: int, extents: FloatArray):
    res = jnp.array([Nx, Ny])
    step = jnp.asarray(extents) / res
    grid = dftlib.DFTSampler.create_centered(spacing=step, shape=(Nx, Ny))
    return grid


def make_sensor_from_imgrad(Nx: int, Ny: int, imgrad: float):
    N = jnp.sqrt(1.0 * (Nx**2 + Ny**2))
    extents = 2 * imgrad / N * jnp.array([Nx, Ny])
    return make_sensor_dftlayout(Nx=Nx, Ny=Ny, extents=extents)


def make_sensor_from_pixsize(Nx: int, Ny: int, pixsize: float):
    extents = pixsize * jnp.array([Nx, Ny])
    return make_sensor_dftlayout(Nx=Nx, Ny=Ny, extents=extents)


def downsample_integer(x: JArray, factor: int | Tuple[int]) -> JArray:
    import jax.image as jimg

    if isinstance(factor, int):
        f1, f2 = factor, factor
    else:
        f1, f2 = factor

    if f1 == 1 and f2 == 1:
        return x

    return jimg.resize(
        x,
        shape=(x.shape[0] // f1, x.shape[1] // f2, *x.shape[2:]),
        method="linear",
        antialias=True,
    )


class SphereWavefront:
    """A wavefront sampler on a sphere surface for Debye diffraction."""

    @property
    def sphere(self) -> geolib.Sphere: ...

    def sample(self, theta: FloatArray, phi: FloatArray) -> ComplexArray: ...
