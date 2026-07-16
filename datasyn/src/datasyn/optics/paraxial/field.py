from dataclasses import dataclass

import jax.numpy as jnp

from datasyn.optics.typing import *


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Field:
    """
    TODO: This is very similar to Pupil!
          They are "disk symmetric around z-axis".
          Particularly, (z, r) representation and "norm coord to world" is identical.
          Think of how we can unify them properly.
    """

    z: JArray
    r: JArray

    def set_z(self, z: JArray):
        return Field(z=z, r=self.r)

    def set_r(self, r: JArray):
        return Field(z=self.z, r=r)

    @property
    def ndim(self) -> Tuple[int, ...]:
        return self.z.ndim

    def norm_coord_to_world(self, f_xy: JArray):
        assert self.ndim == 0
        w_xy = self.r * f_xy
        w_z = jnp.full_like(f_xy[..., 0:1], self.z)
        w = jnp.concat([w_xy, w_z], axis=-1)
        return w
