from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *
from datasyn.optics.paraxial import Pupil
from datasyn.optics.paraxial.rtm2 import VirtualThinLens


class SafeClusterOut(NamedTuple):
    clusters: np.ndarray
    indices: np.ndarray
    mapped: np.ndarray

    @property
    def size(self):
        return self.clusters.size


def safe_cluster(values: np.ndarray, eps: float) -> SafeClusterOut:
    """Clusters 1D values with error <= eps, minimizing number of clusters."""
    values = np.asarray(values)
    shape = values.shape
    values1d = values.flatten()

    unique1d, inv = np.unique(values1d, return_inverse=True)
    values_sort = np.sort(np.asarray(unique1d).reshape(-1))

    clusters = []
    i = 0
    n = len(values_sort)
    while i < n:
        start = values_sort[i]
        j = i
        while j + 1 < n and values_sort[j + 1] - start <= 2 * eps:
            j += 1
        center = 0.5 * (values_sort[i] + values_sort[j])
        clusters.append(center)
        i = j + 1

    clusters = np.asarray(clusters)
    abs_diff = np.abs(unique1d[..., None] - clusters)
    min_idx_unique = np.argmin(abs_diff, axis=-1)
    min_idx = min_idx_unique[inv]
    mapped = clusters[min_idx].reshape(shape)

    return SafeClusterOut(
        clusters=clusters, indices=min_idx.reshape(shape), mapped=mapped
    )


@jx.jit
def calc_signed_coc(thinlens: VirtualThinLens, xp: Pupil, z_obj: float, z_focus: float):
    """
    Calculates circle of confusion radius with sign under paraxial regime.
    The sign is positive if the observation is behind the image, and negative otherwise(front the observation).

    TODO: Integrate to my paraxial module!

    TODO 260714: VirtualThinLens is outdated! PupilModel is more consistent. Refactor this perhaps in the future.
    """
    # Calculate image and observation positions relative to XP plane.
    d_img = thinlens.o2i(z_obj)[0] + thinlens.bpp - xp.z
    d_obs = thinlens.o2i(z_focus)[0] + thinlens.bpp - xp.z

    # This formula can be derived from triangle similarity.
    coc = xp.r * (d_obs - d_img) / d_img
    return coc


@jx.jit
def calc_obj_from_signed_coc(
    thinlens: VirtualThinLens, xp: Pupil, scoc: float, z_focus: float
):
    """
    Signed CoC and object z for given system and focusing are one-to-one corresponding.
    """
    d_obs = thinlens.o2i(z_focus)[0] + thinlens.bpp - xp.z
    d_img = xp.r * d_obs / (scoc + xp.r)
    z_img = d_img - thinlens.bpp + xp.z
    z_obj, _ = thinlens.i2o(z_img)

    return z_obj


@jx.jit
def find_debye_sampling_number(
    thinlens: VirtualThinLens,
    xp: Pupil,
    tanhfov: float,
    focusing: float,
    dpt_min: float,
    dpt_max: float,
    f_xy2: float,
    wvl: float,
    C_debye_scale: float,
):
    # Calculate distances relative to BPP.
    z_obs, _ = thinlens.o2i(focusing)

    def get_sampling_number(z_obj: float):
        z_img, _ = thinlens.o2i(z_obj)  # relative to BPP
        d_img = z_img + thinlens.bpp - xp.z  # relative to XP plane

        xy2_img = z_img * tanhfov * f_xy2
        xpsphr_r2 = d_img**2 + xy2_img
        na2 = xp.r**2 / (xp.r**2 + xpsphr_r2)

        dz = z_obs - z_img  # Defocus z

        sam = C_debye_scale * 4 * na2 * jnp.abs(dz) / (jnp.sqrt(1 - na2) * wvl)
        return sam

    sam1 = get_sampling_number(dpt_min)
    sam2 = get_sampling_number(dpt_max)
    return jnp.maximum(sam1, sam2)
