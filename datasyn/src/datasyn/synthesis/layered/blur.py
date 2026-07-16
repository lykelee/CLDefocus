from functools import partial

import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *


def pad_img(img: JArray, pad, mode):
    pad_width = img.ndim * [(0, 0)]
    pad_width[0] = pad[0]
    pad_width[1] = pad[1]
    return jnp.pad(img, pad_width=pad_width, mode=mode)


def calc_pad_for_fft(shape_input, shape_target):
    """
    Calculates the padding to make the input shape to the target under FFT grid layout with center alignment.
    """
    total = [t - i for i, t in zip(shape_input, shape_target)]
    left = [(t // 2) - (i // 2) for i, t in zip(shape_input, shape_target)]
    right = [p - L for p, L in zip(total, left)]

    return list(zip(left, right))


@partial(jx.jit, static_argnames=("axes",))
def _conv_img_ker_by_fft(img: JArray, ker: JArray, axes):
    """
    Primitive FFT-convolution for an image and a center-aligned kernel.
    This assumes that the image and kernel already have the same size.
    """
    Fimg = jnp.fft.fft2(img, axes=axes)
    Fker = jnp.fft.fft2(jnp.fft.ifftshift(ker, axes=axes), axes=axes)
    out = jnp.fft.ifft2(Fimg * Fker, axes=axes)
    return out


def conv2d_fft(patch: JArray, kernel: JArray) -> JArray:
    """
    2D convolution supporting single- and multi-channel images & kernels.

    Args:
      patch:  shape (H, W) or (H, W, Ci)
      kernel: shape (kh, kw) or (kh, kw, Ck)

    Returns:
      convolved result of shape
        - (H_out, W_out) if only one output channel,
        - (H_out, W_out, C_out) otherwise.
    """
    # Ensure 3-D representations: (...,Ci)
    if patch.ndim == 2:
        patch = patch[..., None]
    if patch.ndim != 3:
        raise ValueError(f"patch must be (H,W) or (H,W,C); got {patch.shape}")

    if kernel.ndim == 2:
        kernel = kernel[..., None]
    if kernel.ndim != 3:
        raise ValueError(f"kernel must be (kh,kw) or (kh,kw,C); got {kernel.shape}")

    H, W, Ci = patch.shape
    kh, kw, Ck = kernel.shape

    assert kh % 2 == 0 and kw % 2 == 0, (
        "Currently, we only support even-valued kernel size!"
    )

    # Broadcast one of the channels if needed
    if Ci == Ck:
        patch_bc, kernel_bc = patch, kernel
    elif Ci == 1 and Ck > 1:
        patch_bc = jnp.broadcast_to(patch, (H, W, Ck))
        kernel_bc = kernel
    elif Ci > 1 and Ck == 1:
        patch_bc = patch
        kernel_bc = jnp.broadcast_to(kernel, (kh, kw, Ci))
    else:
        raise ValueError(
            f"patch channels {Ci} and kernel channels {Ck} must match or one be 1"
        )

    # NOTE: Now we need to pad both image and kernel.
    #       This might be quite confusing! I want to refactor this later.
    #       We need padding for two reasons:
    #       1. To ensure two signals have the same shape.
    #       2. The image boundary should be padded by (ks//2) per each side because FFT naturally performs circular conv.
    #       By 2, the image is padded and its size is not smaller than the kernel.
    #       Hence, for 1, we always need to pad the kernel.
    #       Parity handling is tedious. We assume kernel size is always even number.
    #       Hence, we consider two cases: total pad is even or odd.
    #       If even, we simply pad by total_pad / 2.
    #       If odd, we pad left by (total_pad - 1) / 2 and right by (total_pad + 1) / 2.
    #       This is for sampling grid setup of standard(numpy-like) FFT.
    #       If kernel size is odd, we'd need swap even and odd pad handlings.

    h_sig = max(H, kh)
    w_sig = max(W, kw)

    pad_patch = calc_pad_for_fft((H, W), (h_sig, w_sig)) + [(0, 0)]
    patch_pad = jnp.pad(patch_bc, pad_patch, mode="empty")

    pad_kernel = calc_pad_for_fft((kh, kw), (h_sig, w_sig)) + [(0, 0)]
    kernel_pad = jnp.pad(kernel_bc, pad_kernel, mode="empty")

    full = _conv_img_ker_by_fft(patch_pad, kernel_pad, axes=(0, 1))
    valid = full[
        pad_patch[0][0] : h_sig - pad_patch[0][1],
        pad_patch[1][0] : w_sig - pad_patch[1][1],
    ]

    # Convert to real if the inputs are all real.
    if not (jnp.iscomplexobj(patch_bc) or jnp.iscomplexobj(kernel_bc)):
        valid = valid.real

    return valid[..., 0] if valid.shape[2] == 1 else valid


class _DoFIndex(NamedTuple):
    """Loop index for dof composite loop"""

    color: JArray
    alpha: JArray
    psf: JArray


@partial(jx.jit, static_argnames=("hkh", "hkw"))
def _layered_dof_composite_step(
    C_prev: JArray,  # accumulated back -> front result so far, (H, W, [C])
    idx: _DoFIndex,
    *,
    hkh: int,
    hkw: int,
):
    """
    Refer to the Equation 4 of the main paper.
    """
    C, a, k = idx

    a_pad = pad_img(a, [(hkh, hkh), (hkw, hkw)], "edge")
    C_pad = pad_img(C, [(hkh, hkh), (hkw, hkw)], "edge")
    A_pad = conv2d_fft(a_pad, k)
    A = A_pad[hkh:-hkh, hkw:-hkw]  # (H, W)        = P_i * A_i
    B_pad = conv2d_fft(a_pad[..., None] * C_pad, k)
    B = B_pad[hkh:-hkh, hkw:-hkw]  # (H, W, 3)     = P_i * (A_i C_i)

    if A.ndim == 2:
        A = A[:, :, None]

    Cout = B + (1.0 - A) * C_prev

    return Cout, None


def layered_dof_composite(
    colors_f2b: JArray,  # (K, H, W, [C]), linear RGB, front -> back
    alphas_f2b: JArray,  # (K, H, W), coverage mattes in [0, 1]
    psfs_f2b: JArray,  # (K, kh, kw)
    dtype: Dtype = None,
) -> JArray:
    # NOTE: We usually use large PSF kernels (ks > 63)
    #       Then FFT is usually better than direct convolution.

    assert colors_f2b.ndim in [3, 4]
    K, H, W = colors_f2b.shape[:3]
    assert alphas_f2b.ndim == 3

    assert alphas_f2b.shape == (K, H, W), "alphas must be (K,H,W)"
    assert psfs_f2b.shape[0] == K

    colors_f2b = colors_f2b.astype(jnp.result_type(colors_f2b.dtype, dtype))
    alphas_f2b = alphas_f2b.astype(jnp.result_type(alphas_f2b.dtype, dtype))
    psfs_f2b = psfs_f2b.astype(jnp.result_type(psfs_f2b.dtype, dtype))

    kh, kw = psfs_f2b.shape[1:3]
    assert kh % 2 == 0 and kw % 2 == 0, "We now support only even-valued kernel sizes!"
    hkh, hkw = kh // 2, kw // 2

    Cout_0 = jnp.zeros(colors_f2b.shape[1:], dtype=colors_f2b.dtype)

    # Composite back -> front.
    body = partial(_layered_dof_composite_step, hkh=hkh, hkw=hkw)
    dof_indices = _DoFIndex(color=colors_f2b, alpha=alphas_f2b, psf=psfs_f2b)
    Cout, _ = jx.scan(body, Cout_0, dof_indices, reverse=True)

    return Cout
