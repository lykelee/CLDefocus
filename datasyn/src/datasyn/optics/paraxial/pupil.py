from dataclasses import dataclass

import jax.numpy as jnp

from datasyn.optics.typing import *


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Pupil:
    """
    A pupil symmetric around the z-axis.
    """

    z: JArray
    r: JArray

    @staticmethod
    def zero():
        return Pupil(z=jnp.array(0.0), r=jnp.array(0.0))

    @staticmethod
    def one():
        return Pupil(z=jnp.array(0.0), r=jnp.array(1.0))

    @staticmethod
    def inf():
        return Pupil(z=jnp.array(0.0), r=jnp.array(float("inf")))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.z.shape

    @property
    def ndim(self) -> Tuple[int, ...]:
        return self.z.ndim

    def reshape(self, new_shape: Sequence[int]):
        return Pupil(z=self.z.reshape(new_shape), r=self.r.reshape(new_shape))

    def __getitem__(self, idx):
        return Pupil(z=self.z[idx], r=self.r[idx])

    def set_z(self, z: JArray):
        return Pupil(z=z, r=self.r)

    def set_r(self, r: JArray):
        return Pupil(z=self.z, r=r)

    def scale(self, s: JArray):
        """Multiplies the radius by `s`."""
        assert s >= 0
        return Pupil(z=self.z, r=s * self.r)

    def norm_coord_to_world(self, p_xy: JArray):
        assert self.ndim == 0
        w_xy = self.r * p_xy
        w_z = jnp.full_like(p_xy[..., 0:1], self.z)
        w = jnp.concat([w_xy, w_z], axis=-1)
        return w

    @property
    def diam(self) -> FloatArray:
        """The diameter of the pupil."""
        return 2 * self.r

    @property
    def area(self) -> FloatArray:
        """The area of the pupil circle (assuming 3D space)."""
        return jnp.pi * self.r**2
