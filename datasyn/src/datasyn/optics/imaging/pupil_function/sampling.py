from functools import partial

import jax.numpy as jnp

import datasyn.optics.ray as ray
import datasyn.optics.ray as rayopt
import datasyn.optics.safeop as safeop
from datasyn.jaxutils import jx
from datasyn.optics.complens.tabular_lens import TabularLens, trace_tabular_lens
from datasyn.optics.paraxial import Field, Pupil
from datasyn.optics.ray import DEFAULT_WAVELENGTH, ORay
from datasyn.optics.typing import *


def stratified_samples_unit2D(
    rng: RngKey,
    grid_shape=Tuple[int, int],
    per_cell=1,
    keep_structure: bool = False,
    jitter: bool = True,
):
    from datasyn.jaxutils.sampling import stratified_samples

    return stratified_samples(
        rng,
        grid_shape=grid_shape,
        bounds=(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        per_cell=per_cell,
        jitter=jitter,
        shuffle=False,
        keep_structure=keep_structure,
    )


def input_ray(
    ep: Pupil,
    field: Field,
    f_xy: JArray,
    p_xy: JArray,
    wvln: float = DEFAULT_WAVELENGTH,
    use_pss: bool = False,
):
    """
    Samples input rays from a field point toward EP.
    """
    src = field.norm_coord_to_world(f_xy)
    dst = ep.norm_coord_to_world(p_xy)

    ray_g_in = ray.generate_rays_from_origin_target(src, dst)

    opl = jnp.full_like(ray_g_in.o[..., 0], 0.0)

    ray_in = rayopt.make_oray(
        g=ray_g_in,
        wvln=wvln,
        opl=opl,
        valid=True,
        pss=float("-inf") if use_pss else None,
    )

    return ray_in


def input_chief_ray(
    ep: Pupil,
    field: Field,
    f_xy: JArray,
    wl: float = DEFAULT_WAVELENGTH,
):
    """
    Samples a chief input ray from a field point toward the center of EP.
    """
    p_xy = jnp.zeros((2,))
    return input_ray(ep, field, f_xy, p_xy, wvln=wl)


class TraceChiefRayOut(NamedTuple):
    ray_in: ORay
    ray_out: ORay


def trace_chief_ray(
    lens: TabularLens,
    ep: Pupil,
    field: Field,
    f_xy: JArray,
    wl: float = DEFAULT_WAVELENGTH,
):
    chief_in = input_chief_ray(ep, field, f_xy, wl=wl)
    chief_out = trace_tabular_lens(lens, chief_in[None])
    return TraceChiefRayOut(chief_in, chief_out)


def generate_random_input_rays(
    ep: Pupil,
    rng: RngKey,
    n: int,
    field: Field,
    f_xy: JArray,
    wvln: float = DEFAULT_WAVELENGTH,
    jitter: bool = True,
    temp_disk_sampling: bool = True,
    use_pss: bool = False,
):
    p_xy = stratified_samples_unit2D(rng, (n, n), jitter=jitter)  # (N, 2)

    if temp_disk_sampling:
        from datasyn.jaxutils.sampling import map_square_to_disk

        p_xy = map_square_to_disk(p_xy)

    def sample(p: JArray):
        return input_ray(ep, field, f_xy, p, wvln=wvln, use_pss=use_pss)

    rays = jx.vmap(sample)(p_xy)
    return rays, p_xy


def best_focus_z(o: JArray, d: JArray, w: JArray):
    """
    Finds a z-plane that minimizes the variance(RMS) of spot points of lines.

    o: (N, 3)
    d: (N, 3) with d[:,2] != 0 (not necessarily unit vectors)
    w: (N,) nonnegative weights
    returns: scalar z0*
    """
    w = jnp.asarray(w)
    wsum = jnp.sum(w)

    o_xy = o[:, :2]  # (N,2)
    o_z = o[:, 2:3]  # (N,1)
    d_xy = d[:, :2]  # (N,2)
    d_z = d[:, 2:3]  # (N,1)

    a = d_xy / d_z  # (N,2)
    b = o_xy - o_z * a  # (N,2)

    # weighted means (broadcast w as (N,1))
    w1 = w[:, None]
    a_bar = safeop.div(jnp.sum(w1 * a, axis=0, keepdims=True), wsum).v
    b_bar = safeop.div(jnp.sum(w1 * b, axis=0, keepdims=True), wsum).v

    A = a - a_bar  # (N,2)
    B = b - b_bar  # (N,2)

    BA = jnp.sum(B * A, axis=-1)  # (N,) dot products
    AA = jnp.sum(A**2, axis=-1)  # (N,) squared norms

    num = jnp.sum(w * BA)  # Σ w_i (B_i·A_i)
    den = jnp.sum(w * AA)  # Σ w_i ||A_i||^2

    z0 = safeop.div(-num, den).v

    if True:
        BB = jnp.sum(B**2, axis=-1)
        SBB = jnp.sum(w * BB)
        min_var = safeop.div(SBB - safeop.div(num**2, den).v, wsum).v

    return z0, min_var


def lens_best_focus_z(
    lens: TabularLens,
    z_start: float,
    ep: Pupil,
    rng: RngKey,
    field: Field,
    f_xy: JArray,
    n: int = 63,
    wvl: float = DEFAULT_WAVELENGTH,
):
    """
    Finds the best focus z that minimizes the RMS spot size.
    """
    rays_in, p_xy = generate_random_input_rays(
        ep,
        rng,
        n,
        field,
        f_xy,
        wvln=wvl,
        jitter=False,
        temp_disk_sampling=True,
    )  # (N,)

    # To ensure the rays are all front of the lens.
    rays_in = jx.vmap(partial(ray.project_to_z, z_start))(rays_in).ray

    rays_out = trace_tabular_lens(lens, rays_in)
    focus, temp_var = best_focus_z(rays_out.o, rays_out.d, rays_out.valid)

    return focus


def lens_best_focus_by_intersection(
    lens: TabularLens,
    z_start: float,
    ep: Pupil,
    rng: RngKey,
    field: Field,
    f_xy: JArray,
    n: int = 63,
    wvl: float = DEFAULT_WAVELENGTH,
):
    rays_in, p_xy = generate_random_input_rays(
        ep,
        rng,
        n,
        field,
        f_xy,
        wvln=wvl,
        jitter=False,
        temp_disk_sampling=True,
    )  # (N,)

    # To ensure the rays are all front of the lens.
    rays_in = jx.vmap(partial(ray.project_to_z, z_start))(rays_in).ray

    rays_out = trace_tabular_lens(lens, rays_in)

    focus = rayopt.oray_intersection(rays_out)

    return focus


def weighted_mean(x: JArray, w: JArray):
    """
    TODO: Broadcasting! Move to safeop module!
    """
    return safeop.div(jnp.sum(x * w[:, None], axis=0), jnp.sum(w))


def weighted_rmse(x: JArray, w: JArray, target: JArray):
    """
    TODO: Broadcasting! Move to safeop module!
    """
    return safeop.sqrt(
        safeop.div(jnp.sum(w * jnp.sum((x - target[None]) ** 2, axis=-1)), jnp.sum(w)).v
    ).v


def ray_xy_rmse(rays: ORay, pt: JArray):
    """
    TODO: Broadcasting!
    """
    return weighted_rmse(rays.o[:, 0:2], rays.amp, pt)
