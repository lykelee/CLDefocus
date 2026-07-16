"""
An interface to provide source photos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Photo(ABC):
    """
    Handle for a single loaded photo.

    Implementors should load all heavy data (RAW file, image arrays, ...) in
    ``__init__``.  Lazy loading is acceptable for methods that are not always called.
    """

    @abstractmethod
    def get_linear(self) -> np.ndarray:
        """
        Linear-space RGB image, shape (H, W, 3), dtype float32.

        Returned at the same spatial resolution as ``get_depth()``.
        """
        ...

    @abstractmethod
    def get_depth(self) -> np.ndarray:
        """Depth map, shape (H, W), dtype float32."""
        ...

    def lin_to_rgb(self, lin: np.ndarray) -> np.ndarray:
        """
        Convert a linear-space image to display RGB, shape (H, W, 3).

        Default: identity.  Override with CCM + gamma (RawPhoto) or leave as
        identity for synthetic/linear data.
        Used by Stage 1 (s1_patch_crop) for visualisation and by Stage 6 (s6_blur) for the sharp path.
        """
        return lin

    def raw_to_lin(self, raw: np.ndarray) -> np.ndarray:
        """
        Demosaic a normalized RAW Bayer image to linear RGB, shape (H, W, 3).

        Input: (H, W) float32 normalized Bayer, values in [0, 1].
        Output: (H, W, 3) float32 linear (or near-linear) RGB.

        Default: raises NotImplementedError.  Override with a demosaic
        (NormalPhoto) or EasyRaw demosaic (RawPhoto).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement raw_to_lin."
        )

    def linear_to_raw(self, x: np.ndarray) -> np.ndarray:
        """
        Convert a linear-space patch to normalized RAW (Bayer), shape (H, W), float32.

        Used by Stage 6 (s6_blur) to enter the RAW domain before noise simulation.
        Values should be in [0, 1].

        Default: raises NotImplementedError.  Override with a real ISP
        (RawPhoto) or a virtual Bayer mosaicing (NormalPhoto).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement linear_to_raw."
        )

    def get_paired_blurry_rgb(self) -> np.ndarray:
        """
        Return the paired real blurry RGB image, aligned to get_depth() resolution.

        This is optional and only implemented by providers/datasets that carry a
        real blurry counterpart for each sharp photo. The returned image should
        already be converted to display RGB in the same coordinate frame that
        Stage 1 (s1_patch_crop) patch geometry expects.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_paired_blurry_rgb."
        )


class PhotoProvider(ABC):
    """
    Factory that produces ``Photo`` handles on demand.

    Stage 1 (s1_patch_crop) calls ``get_names()`` once and ``open(photo)`` once per
    (photo, cycle).  Stage 6 (s6_blur) calls ``open(photo)`` once per photo group.
    """

    @abstractmethod
    def get_names(self) -> list[str]:
        """Consistent, ordered list of photo names in this dataset."""
        ...

    @abstractmethod
    def open(self, photo: str) -> Photo:
        """Load and return a ``Photo`` handle for the given photo name."""
        ...


class PhotoProviderSpec(ABC):
    """
    Picklable factory specification for a PhotoProvider.

    Holds only plain picklable data (paths, primitives).  Call ``build()``
    inside the target process to obtain a live ``PhotoProvider``.
    """

    @abstractmethod
    def build(self) -> PhotoProvider:
        """Construct and return a live PhotoProvider from this spec."""
        ...
