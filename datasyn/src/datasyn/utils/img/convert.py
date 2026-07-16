from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from PIL import Image


def arr1f_to_pil(x: ArrayLike):
    from PIL import Image

    return Image.fromarray(
        np.clip(np.rint(255.0 * np.asarray(x)), 0, 255).astype(np.uint8)
    )


def arr255u8_to_pil(x: ArrayLike):
    from PIL import Image

    return Image.fromarray(np.asarray(x))


def arr65535_to_pil_16bit(x: ArrayLike):
    from PIL import Image

    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        x = np.rint(x)
    return Image.fromarray(np.clip(x, 0, 65535).astype(np.uint16))


def pil_to_arr1f(img: Image.Image, dtype=np.float32):
    x_uint = np.asarray(img)
    maxval = np.iinfo(x_uint.dtype).max
    return x_uint.astype(dtype=dtype) / maxval
