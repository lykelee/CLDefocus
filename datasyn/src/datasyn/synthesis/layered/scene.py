from dataclasses import dataclass

import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *


@dataclass(frozen=True)
class LayeredScene:
    colors: JArray  # order: foreground -> background
    alphas: JArray

    @property
    def n_layers(self):
        return self.colors.shape[0]

    @property
    def imgsize(self):
        return self.colors.shape[1:3]

    @property
    def colors_f2b(self):
        return self.colors

    @property
    def alphas_f2b(self):
        return self.alphas

    @property
    def colors_b2f(self):
        return self.colors[::-1]

    @property
    def alphas_b2f(self):
        return self.alphas[::-1]

    def crop(self, slice_h: slice, slice_w: slice):
        return LayeredScene(
            self.colors[:, slice_h, slice_w], self.alphas[:, slice_h, slice_w]
        )


class LayerRemoval(NamedTuple):
    layers: LayeredScene
    keep: IntArray  # Layer mask, true if kept


def remove_invisible(layers: LayeredScene, eps: float = 0.0):
    """
    Removes all invisible layers (i.e. layers with alpha = 0) from a scene.

    NOTE: This is not JIT-safe since the result has dynamic size!
    """
    keep = (layers.alphas > eps).any(axis=(1, 2))  # (N,)
    idx = jnp.nonzero(keep, size=keep.shape[0], fill_value=0)[0]  # visible indices
    k = int(keep.sum())
    layers_new = LayeredScene(layers.colors[idx][:k], layers.alphas[idx][:k])
    return LayerRemoval(layers_new, keep)


def layers_from_f2b(colors: JArray, alphas: JArray):
    return LayeredScene(colors=colors, alphas=alphas.astype(colors.dtype))


def layers_from_b2f(colors: JArray, alphas: JArray):
    return LayeredScene(colors=colors[::-1], alphas=alphas[::-1].astype(colors.dtype))


@dataclass(frozen=True)
class LayeredDepthScene:
    layers: LayeredScene
    l2d: JArray  # Layer index to depth

    @property
    def colors_f2b(self):
        return self.layers.colors_f2b

    @property
    def alphas_f2b(self):
        return self.layers.alphas_f2b

    @property
    def colors_b2f(self):
        return self.layers.colors_b2f

    @property
    def alphas_b2f(self):
        return self.layers.alphas_b2f

    @property
    def n_layers(self):
        return self.l2d.shape[0]

    @property
    def imgsize(self):
        return self.layers.imgsize

    def set_layers(self, layers: LayeredScene):
        return LayeredDepthScene(layers, self.l2d)

    def crop_and_normalize(self, slice_h: slice, slice_w: slice):
        """
        Crops the scene and remove all invisible layers & depths.
        """
        layers_crop = self.layers.crop(slice_h, slice_w)
        out = remove_invisible(layers_crop)
        layers_new = out.layers
        l2d_new = self.l2d[out.keep]  # Depths of removed layers
        return LayeredDepthScene(layers_new, l2d_new)

    def make_depthmap(self):
        """
        Makes a depth map of the scene.
        This assumes hard matting: alpha is 0 or 1.
        A layer with alpha 1 occludes all the behind layers.

        TODO: Better implementation!
        """
        alphas = jnp.where(self.layers.alphas_f2b > 0.0, 1.0, 0.0)
        colors_d = alphas.at[:].set(self.l2d[:, None, None])

        ld = layers_from_f2b(colors_d, alphas)
        dm = overlap_layers(ld)
        return dm


def layers_from_one_image(color: JArray, alpha: Optional[JArray] = None):
    colors = color[None]
    if alpha is None:
        alphas = jnp.ones_like(colors[..., 0])
    else:
        alphas = alpha[None].astype(colors.dtype)
    return LayeredScene(colors=colors, alphas=alphas)


def overlap_ca(colors_b2f: JArray, alphas_b2f: JArray):
    """
    Overlaps all layers into one image.

    TODO: Physically plausible?
    """
    n_layers = colors_b2f.shape[0]
    ret_0 = jnp.zeros(colors_b2f.shape[1:], dtype=alphas_b2f.dtype)

    def step(i: int, ret: JArray):
        c, a = colors_b2f[i], alphas_b2f[i]
        if c.ndim == 3:
            a = a[:, :, None]
        ret = (1.0 - a) * ret + a * c
        return ret

    ret = jx.fori_loop(0, n_layers, step, ret_0)
    return ret


def overlap_layers(scene: LayeredScene):
    """
    Overlaps all layers into one image.

    TODO: Physically meaningful?
    """
    return overlap_ca(scene.colors_b2f, scene.alphas_b2f)
