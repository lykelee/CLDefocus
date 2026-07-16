from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp

import datasyn.mathutils.vecop as vecop
from datasyn.jaxutils.typing import *


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Ball:
    """A hypersphere in n-D space."""

    c: JArray
    r: JArray

    @property
    def dim(self) -> int:
        return self.c.shape[-1]

    def signed_distance(self, pt: JArray):
        return vecop.vecnorm(pt - self.c) - self.r

    def __getitem__(self, idx):
        return Ball(self.c[idx], self.r[idx])


Circle: TypeAlias = Ball
Sphere: TypeAlias = Ball


def circle(c: JArray, r: JArray) -> Circle:
    return Ball(c=jnp.asarray(c), r=jnp.asarray(r))


def sphere(c: JArray, r: JArray) -> Sphere:
    return Ball(c=jnp.asarray(c), r=jnp.asarray(r))
