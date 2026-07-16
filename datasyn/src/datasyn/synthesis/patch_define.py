from dataclasses import dataclass

import jax
import numpy as np

import datasyn.optics.safeop as safeop
from datasyn.jaxutils.typing import *


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PatchAug:
    theta: float = 0.0
    scale: float = 1.0
    flip: int = 1  # -1 or 1
    exposure: float = 1.0


def aug_to_mat(aug: PatchAug):
    from datasyn.jaxutils.utils import rot2d, scale2d

    M = rot2d(aug.theta) @ scale2d(aug.scale, aug.flip * aug.scale)
    return M


@dataclass
class PatchDefine:
    """
    Defines a patch of a scene.
    This can be used to deterministically generate data relevant to patches.
    """

    name: str  # The patch name
    photo: str  # The source photo
    photo_size: Tuple[int, int]
    patch_offset: Tuple[int, int]
    patch_size: Tuple[int, int]
    f_xy: Tuple[float, float]
    aug: PatchAug
    depth_min: float
    depth_max: float

    @staticmethod
    def from_dict(row: dict):
        return PatchDefine(
            name=row["name"],
            photo=row["photo"],
            photo_size=(row["photo_size_h"], row["photo_size_w"]),
            patch_offset=(row["patch_offset_i"], row["patch_offset_j"]),
            patch_size=(row["patch_size_h"], row["patch_size_w"]),
            f_xy=(row["field_x"], row["field_y"]),
            aug=PatchAug(
                theta=row["aug_theta"],
                scale=row["aug_scale"],
                flip=row["aug_flip"],
                exposure=row["aug_exposure"],
            ),
            depth_min=row["depth_min"],
            depth_max=row["depth_max"],
        )


def mldepthpro_depth2rgb(depth):
    """ """
    from matplotlib import pyplot as plt

    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = safeop.div(
        inverse_depth - min_invdepth_vizu, max_invdepth_vizu - min_invdepth_vizu
    ).v
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    return color_depth
