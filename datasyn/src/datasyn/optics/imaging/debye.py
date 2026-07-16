"""
References:

Leutenegger, Marcel, et al. "Fast focus field calculations." Optics express 14.23 (2006): 11277-11291.
Cai, Yanan, et al. "Direct calculation of tightly focused field in an arbitrary plane." Optics Communications 450 (2019): 329-334.
"""

from typing import NamedTuple

import jax.numpy as jnp

import datasyn.optics.safeop as safeop
import datasyn.optics.zernike as zernlib
from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *
from datasyn.jaxutils.wrappings import fake_scan
from datasyn.utils.time import easy_timer

import datasyn.mathutils.grid as gridlib
from datasyn.mathutils.complex import sqabs_complex
from datasyn.mathutils.dft import DFTSampleGrid, DFTSampler
from datasyn.mathutils.dft.bluestein import bluestein_dft_2d
from datasyn.optics.imaging.defocus import downsample_integer
from datasyn.utils.context import maybe_ctx

tau = 2 * jnp.pi


def smallest_greater_int(x):
    return jnp.floor(x).astype(int) + 1


class ChunkSizeResult(NamedTuple):
    chunk_size: int
    num_chunks: int
    padding: int


def find_chunk_size(
    n: int,
    max_chunk_size: int,
    *,
    alpha: float = 1,
) -> ChunkSizeResult:
    """
    Find a large chunk size with small padding.

    The score trades off padding against distance from max_chunk_size.

    Lower alpha prioritizes larger chunk sizes.
    Higher alpha prioritizes low padding.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    if n <= max_chunk_size:
        return ChunkSizeResult(
            chunk_size=n,
            num_chunks=1,
            padding=0,
        )

    best: ChunkSizeResult | None = None
    best_score = (float("inf"), float("inf"))

    for m in range(1, max_chunk_size + 1):
        k = (n + m - 1) // m
        padding = k * m - n

        relative_padding = padding / n
        relative_chunk_loss = (max_chunk_size - m) / max_chunk_size

        score = (alpha * relative_padding + relative_chunk_loss, -m)

        result = ChunkSizeResult(
            chunk_size=m,
            num_chunks=k,
            padding=padding,
        )

        if score < best_score:
            best = result
            best_score = score

    assert best is not None
    return best


def calc_na(pupil_r: float, xp_sphere_r: float, ior: float = 1.0):
    return ior * jnp.sqrt(pupil_r**2 / (pupil_r**2 + xp_sphere_r**2))


_bluestein_dft_2d_jit = jx.jit(bluestein_dft_2d, static_argnames=("mout",))


def tilted_debye(
    wvl: float,
    ior: float,
    sin: float,
    theta_0: float,
    phi_0: float,
    w: float,
    upsample: int,
    vport: gridlib.BoundedGrid,
    MM: int,
    zer_cfg: zernlib.ZernikeConfig,
    zer_coefs: JArray,
    zer_cfg_pss: zernlib.ZernikeConfig,
    zer_coefs_pss: JArray,
    verbose: int = 0,
    jit_czt: bool = True,
    grad_safe: bool = False,
):
    """
    Use the formulation proposed by Cai, Yanan, et al.
    """

    def jax_barrier():
        if verbose >= 2:  # To measure time
            jx.block_until_ready_all()

    from datasyn.optics.imaging.pupil_function.zernike_wavefront import (
        fn_sample_zernike_wavefront,
    )

    M = MM // 2

    if verbose >= 1:
        print(f"Plane spectrum space sampling number: {MM} ({M} along radius)")

    if verbose >= 2:
        print(f"Observation space upsampling: x{upsample}")

    cos_theta_max = jnp.sqrt(1 - sin**2)
    cos_t, sin_t = jnp.cos(theta_0), jnp.sin(theta_0)
    cos_p, sin_p = jnp.cos(phi_0), jnp.sin(phi_0)

    # Sample (m, n) with broadcastable 1D axes to avoid dense meshgrid temporaries.
    coord_dtype = cos_theta_max.dtype
    mm = jnp.arange(-M, M, dtype=coord_dtype) / M
    nn = jnp.arange(-M, M, dtype=coord_dtype) / M

    # Calculate k
    k_0 = tau / wvl
    k = ior * k_0

    k_sin = k * sin
    ku = k * (sin * mm[:, None] - sin_t * cos_theta_max)
    kv = k_sin * nn[None, :]
    ku, kv = jnp.broadcast_arrays(ku, kv)  # (MM, MM)
    dk = k_sin / M
    k_grid = DFTSampler.create_centered([dk, dk], (2 * M, 2 * M))

    # Calculate the minimum kz allowed by the Numerical Aperture
    kz_min = k * cos_theta_max

    # Explicit inverse-rotation coefficients avoid vstack/matmul/reshape temporaries.
    rot_x_ku = cos_t * cos_p
    rot_x_kv = -sin_p
    rot_x_kw = sin_t * cos_p
    rot_y_ku = cos_t * sin_p
    rot_y_kv = cos_p
    rot_y_kw = sin_t * sin_p
    rot_z_ku = -sin_t
    rot_z_kw = cos_t

    # Calculate (theta, phi) and sample wavefront
    jax_barrier()
    with maybe_ctx(
        easy_timer(f"Tilted Debye - Sample wavefront, MM = {MM}"),
        enabled=verbose >= 2,
    ):
        # NOTE 260506:
        # Don't uncomment this jit!
        # I don't know why but I found this slows down this part.
        def process_domain_for_sign(kw_sign: JArray, ku_: JArray, kv_: JArray):
            """Processes either the D+ (kw_sign=1) or D- (kw_sign=-1) domain."""

            kuv_sq = ku_**2 + kv_**2
            valid_circle = kuv_sq <= k**2
            kw_mag = safeop.sqrt(k**2 - kuv_sq).v

            # Assign the sign to kw
            kw = kw_sign * kw_mag

            # Rotate backward to original coordinates
            kx = rot_x_ku * ku_ + rot_x_kv * kv_ + rot_x_kw * kw
            ky = rot_y_ku * ku_ + rot_y_kv * kv_ + rot_y_kw * kw
            kz = rot_z_ku * ku_ + rot_z_kw * kw

            chi = (kz >= kz_min) & valid_circle

            # Determine the Mask (chi)
            # It must be within the propagating circle AND satisfy the NA condition on original kz
            # chi = valid_circle & (kz >= kz_min)

            # Map valid kx, ky, kz back to spherical angle phi.
            if grad_safe:
                phi = safeop.gradsafe_arctan2(ky, kx)
            else:
                phi = jnp.arctan2(ky, kx)
            phi = jnp.where(chi, phi, 0)
            phi = jnp.mod(phi, 2 * jnp.pi)

            # Evaluate the wavefront at these angles
            # We assume eval_wavefront handles the mask internally to save computation,
            # or just computes 0 where mask is False.
            if grad_safe:
                rho = safeop.gradsafe_norm(jnp.stack([kx, ky], axis=-1)) / k_sin
            else:
                rho = safeop.sqrt(kx**2 + ky**2).v / k_sin
            rho = jnp.where(chi, rho, 0)

            # Apply phase propagation and divide by |kw|
            # Phase term: e^(+i * |kw| * w) for D+, e^(-i * |kw| * w) for D-
            defocus_phase = jnp.exp(1j * kw_sign * kw_mag * w)

            jax_barrier()
            with easy_timer("Tilted Debye - Wavefront", disable=verbose < 2):
                U_x = fn_sample_zernike_wavefront(
                    wvl=wvl,
                    rho=rho,
                    phi=phi,
                    coefs=zer_coefs,
                    cfg=zer_cfg,
                    coefs_pss=zer_coefs_pss,
                    cfg_pss=zer_cfg_pss,
                )
                jax_barrier()

            # Zero out invalid regions explicitly just in case
            U_x = jnp.where(chi, U_x, 0)

            # Safe division by kw_mag: divide where chi is True, else set to 0.
            term_x = safeop.div(defocus_phase * U_x, kw_mag).v
            jax_barrier()

            return term_x

        def process_domain(ku_: JArray, kv_: JArray):
            jax_barrier()
            with easy_timer("Tilted Debye - process_domain - 1", disable=verbose < 2):
                # Process D+ (where kw > 0)
                Ax_pos = process_domain_for_sign(1, ku_, kv_)
                jax_barrier()

            jax_barrier()
            with easy_timer("Tilted Debye - process_domain - -1", disable=verbose < 2):
                # Process D- (where kw < 0)
                Ax_neg = process_domain_for_sign(-1, ku_, kv_)
                jax_barrier()

            # Sum the domains to form the final integrand
            FT_in_x = Ax_pos + Ax_neg

            return FT_in_x

        def chunk_process_domain_2d(
            fun,
            ku_: JArray,
            kv_: JArray,
            chunk_size_: int,
        ):
            n0, n1 = ku_.shape
            c = chunk_size_
            num0 = (n0 + c - 1) // c
            num1 = (n1 + c - 1) // c
            pad0 = num0 * c - n0
            pad1 = num1 * c - n1

            def blockify_2d(x: JArray):
                if pad0 > 0 or pad1 > 0:
                    x = jnp.pad(x, ((0, pad0), (0, pad1)), mode="edge")
                return (
                    x.reshape(num0, c, num1, c)
                    .transpose(0, 2, 1, 3)
                    .reshape(num0 * num1, c, c)
                )

            ku_chunks = blockify_2d(ku_)
            kv_chunks = blockify_2d(kv_)

            def step(_, chunk_args):
                return None, fun(*chunk_args)

            # NOTE 260506:
            # I don't know why, but native for loop is better!
            _, out_chunks = fake_scan(step, None, (ku_chunks, kv_chunks))
            out = (
                out_chunks.reshape(num0, num1, c, c)
                .transpose(0, 2, 1, 3)
                .reshape(num0 * c, num1 * c)
            )
            return out[:n0, :n1]

        # Try to find efficient chunking parameter with large chunk and small padding.
        chunk_size = find_chunk_size(MM, max_chunk_size=1024).chunk_size

        jax_barrier()
        with easy_timer("Tilted Debye - chunk_process_domain_2d", disable=verbose < 2):
            FT_in_x = chunk_process_domain_2d(
                process_domain,
                ku,
                kv,
                chunk_size,
            )
            jax_barrier()

        jax_barrier()

    # Create evaluation grid

    vport_fine = vport.upsample(upsample)

    # Perform CZT

    jax_barrier()
    with maybe_ctx(
        easy_timer(f"Tilted Debye - Bluestein, MM = {MM}"), enabled=verbose >= 2
    ):
        # NOTE: I didn't consider this is inverse FT yet!

        U_out_freq_grid = DFTSampleGrid(sampler=k_grid, values=FT_in_x)

        # Evaluate DFT over the full viewport (static shape, JIT-compatible).
        # Frequency bandwidth = 1 / dk (spatial is k space: wavenumber).
        # Multiplying 2pi is due to space-frequency relationship x = 2pi * f.
        f1, f2 = vport_fine.world_range()
        f1 = f1 / tau
        f2 = f2 / tau

        if jit_czt:
            U_out_eval, _ = _bluestein_dft_2d_jit(
                U_out_freq_grid, f1=f1, f2=f2, mout=vport_fine.shape
            )
        else:
            U_out_eval, _ = bluestein_dft_2d(
                U_out_freq_grid, f1=f1, f2=f2, mout=vport_fine.shape
            )

        jax_barrier()

    # Zero out samples outside the frequency bandwidth to suppress aliasing.
    # This replaces the old ROI sub-evaluation: the mask is equivalent but
    # fully traceable (no dynamic shape).
    bandwidths = 1 / (1 * dk)  # Frequency bandwidth = 1 / dx
    shape_arr = jnp.array(vport_fine.shape, dtype=vport_fine.step.dtype)
    half_span = 0.5 * vport_fine.step * shape_arr  # (2,) — constant under JIT?
    xs = vport_fine.step[0] * jnp.arange(
        vport_fine.shape[0], dtype=vport_fine.step.dtype
    )
    ys = vport_fine.step[1] * jnp.arange(
        vport_fine.shape[1], dtype=vport_fine.step.dtype
    )
    bw_x = jnp.abs(xs - half_span[0]) <= 0.5 * tau * bandwidths
    bw_y = jnp.abs(ys - half_span[1]) <= 0.5 * tau * bandwidths
    bw_mask = bw_x[:, None] & bw_y[None, :]
    U_out_eval = U_out_eval * bw_mask

    U_out_eval = sqabs_complex(U_out_eval)
    I_out = downsample_integer(U_out_eval, upsample)

    return I_out


def debye_sampling_bound_for_antialiasing(ior: float, wvl: float, sin: float, z: float):
    """
    TODO: Reorganize! Obsolete functions in older code!

    Lower bound of the sampling number by equation 14.

    NOTE: This returns twice of equation 14 because this is sampling number across *diameter*, not radius.
            The `M` in the equation 14 is sampling number across radius.
    """
    return 2 * smallest_greater_int(
        2 * ior * (sin**2) * jnp.abs(z) / (jnp.sqrt(1 - sin**2) * wvl)
    )
