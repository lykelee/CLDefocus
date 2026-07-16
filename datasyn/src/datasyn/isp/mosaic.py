"""
NOTE: I'm looking forward to using JAX in the future!
      Consider JAX-friendly style although we're not using them now.
"""

from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def bayer_masks(
    h: int, w: int, pattern: Sequence[int]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Return boolean masks (R,G,B) for the Bayer pattern on an HxW grid.
    0-based indexing; (0,0) is top-left.
    """
    yy, xx = np.indices((h, w))
    eo = (yy & 1, xx & 1)  # (row parity, col parity)
    pattern = tuple(pattern)
    if pattern == (0, 1, 1, 2):  # R G / G B
        R = (eo[0] == 0) & (eo[1] == 0)
        G = (eo[0] == 0) & (eo[1] == 1) | (eo[0] == 1) & (eo[1] == 0)
        B = (eo[0] == 1) & (eo[1] == 1)
    elif pattern == (1, 0, 2, 1):  # G R / B G
        R = (eo[0] == 0) & (eo[1] == 1)
        G = (eo[0] == 0) & (eo[1] == 0) | (eo[0] == 1) & (eo[1] == 1)
        B = (eo[0] == 1) & (eo[1] == 0)
    elif pattern == (1, 2, 0, 1):  # G B / R G
        R = (eo[0] == 1) & (eo[1] == 0)
        G = (eo[0] == 0) & (eo[1] == 0) | (eo[0] == 1) & (eo[1] == 1)
        B = (eo[0] == 0) & (eo[1] == 1)
    elif pattern == (2, 1, 1, 0):  # B G / G R
        R = (eo[0] == 1) & (eo[1] == 1)
        G = (eo[0] == 0) & (eo[1] == 1) | (eo[0] == 1) & (eo[1] == 0)
        B = (eo[0] == 0) & (eo[1] == 0)
    else:
        raise ValueError("Unknown Bayer pattern")
    return R, G, B


def mosaic_bayer(rgb: NDArray, pattern: Sequence[int] = (0, 1, 1, 2)) -> NDArray:
    """
    rgb: (H,W,3) in linear domain [0,1] or arbitrary energy units
    returns mosaiced single-channel (H,W) CFA
    """
    h, w, _ = rgb.shape
    Rm, Gm, Bm = bayer_masks(h, w, pattern)
    cfa = np.where(Rm, rgb[..., 0], 0.0)
    cfa = np.where(Gm, rgb[..., 1], cfa)
    cfa = np.where(Bm, rgb[..., 2], cfa)
    return cfa
