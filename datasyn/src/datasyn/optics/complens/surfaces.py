import jax
import jax.numpy as jnp

import datasyn.jaxutils.nputils as myjnputils
import datasyn.mathutils.vecop as vecop
import datasyn.optics.ray as rayopt
import datasyn.optics.safeop as safeop
from datasyn.jaxutils import jx
from datasyn.optics.typing import *


@jx.jit
def refract(eta: JArray, n: JArray, d: JArray):
    """
    Calculates the refracted direction by Snell's law.
    The refracted direction is undefined if total internal reflection occurs.
    To handle this, this also returns definedness mask.

    eta: relevant refraction coefficient (eta = eta_i / eta_t)

    TODO: Batch handling of eta.
          This is originally designed for constant eta.
          However, batched version is used now.
          Ambiguous broadcasting currently.
    """
    assert n.ndim - 1 == 1 and d.ndim - 1 == 1, (
        "TEMP: This should handle arbitrary batch axes!"
    )

    # TODO: quick-and-dirty!
    eta = jnp.atleast_1d(eta)

    cos_i = -jx.vmap(jnp.dot)(n, d)

    # Compute sin^2(theta_t) via Snell's law
    sin2_t = eta**2 * (1.0 - cos_i**2)

    # Cosine of transmitted angle
    # (Undefined when total internal reflection)
    cos_t, defined = safeop.sqrt(1.0 - sin2_t).vd

    # Refracted direction
    d_ = eta[:, None] * d + (eta * cos_i - cos_t)[:, None] * n

    # Mathematically, this is not required since it is already a unit vector.
    # However, this is to guard against numerical drift.
    # TODO: Optionally enable/disable this?
    d_ = vecop.normdir(d_)

    return d_, defined


class IntersectionResult(NamedTuple):
    t: FloatArray
    ray: rayopt.TLensRay
    normal: FloatArray
    defined: BoolArray


@jx.jit
def intersect_plane(ray: rayopt.TLensRay, ior: float):
    """
    Intersect a ray with a plane z = 0.
    """
    assert ray.ndim == 1, "TEMP: This should handle arbitrary batch axes!"

    o, d = ray.o, ray.d

    out = safeop.div(-o[..., 2], d[..., 2])
    t, defined = out.v, out.d
    ray_ = ray.propagate(t, ior=ior)

    # plane normal is +z, flipped if necessary
    n = jnp.array([0.0, 0.0, 1.0], dtype=o.dtype)
    n = myjnputils.unsqueeze_and_repeat(n, 0, d.shape[0])
    n = jnp.where((jx.vmap(jnp.dot)(n, d) > 0.0)[:, None], -n, n)

    return IntersectionResult(t=t, ray=ray_, normal=n, defined=defined)


@jx.jit
def intersect_spherical(c: float, ray: rayopt.TLensRay, ior: float):
    """
    Intersect a ray with a spherical surface.
    This analytically finds the intersection with careful consideration of numerical stability.

    `c`: Curvature (reciprocal of radius)
    """
    assert ray.ndim == 1, "TEMP: This should handle arbitrary batch axes!"

    def planar():
        return intersect_plane(ray, ior)

    def nonplanar():
        r = 1.0 / c
        cen = jnp.array([0.0, 0.0, r])  # Sphere center
        o = ray.o - cen[None]
        d = ray.d

        # NOTE: This aims to avoid catastrophic cancellation by B^2 - 4AC appearing in quadratic root calculation.

        # Closest-approach parametrization
        t0 = -jx.vmap(jnp.dot)(d, o)  # closest approach to center
        p = o + t0[:, None] * d  # orthogonal residual (d·p = 0)
        # Distance-from-center at closest approach
        p2 = jx.vmap(jnp.dot)(p, p)
        rad2 = r**2
        m2 = rad2 - p2  # must be >= 0 for intersection

        # Handle existence
        sqrt_m2 = jnp.sqrt(jnp.maximum(m2, 0.0))
        s_small = -sqrt_m2
        s_large = sqrt_m2
        t_small = t0 + s_small
        t_large = t0 + s_large

        # Choose root
        use_closer = jnp.logical_xor(d[..., 2] > 0.0, c < 0.0)
        t = jnp.where(use_closer, t_small, t_large)

        defined = m2 >= 0.0

        # Intersection point via small quantities: P' = p + s*d, then shift back
        s = jnp.where(use_closer, s_small, s_large)
        P_local = p + s[:, None] * d  # equals R * n_hat
        P = cen + P_local

        # Normal from local vector (no big subtract)
        n_hat = P_local / jnp.where(r == 0.0, 1.0, r)  # already unit if |r| > 0
        n = jnp.where((jx.vmap(jnp.dot)(n_hat, d) > 0.0)[:, None], -n_hat, n_hat)

        ray_ = ray.propagate(t, ior=ior)
        # overwrite origin with numerically stable P (same point but better)
        ray_ = ray_.set_o(o=P)

        return IntersectionResult(t=t, ray=ray_, normal=n, defined=defined)

    return jx.cond(c == 0.0, planar, nonplanar)


def _evenasph_sag(c: JArray, k: JArray, coefs: JArray, x: JArray, y: JArray):
    """
    TODO: Currently, polynomial starts from 4th order. Zemax seems to start from 2nd order.
          We need generalization!
    """
    temp_unit = 1e-3

    r2 = x * x + y * y
    a = safeop.sqrt(1.0 - (k + 1.0) * (c**2) * r2).v
    z = c * r2 / (1.0 + a)

    coef_unit = coefs * (temp_unit ** (2 * jnp.arange(2, coefs.shape[0] + 2)))
    r2_unit = r2 / temp_unit**2

    # Horner — mirrors safeop.dot: Inf products are zeroed, not propagated

    def horner_step(acc, coeff):
        # Invalid (out-of-aperture) rays can have huge r2_unit -> Inf.
        # Zero out Inf accumulator values, same as safeop.dot zeroes Inf product terms.
        prod = acc * r2_unit
        prod_safe = jnp.where(jnp.isfinite(prod), prod, jnp.zeros_like(prod))
        return prod_safe + coeff, None

    poly_sum, _ = jax.lax.scan(horner_step, jnp.zeros_like(r2), coef_unit[::-1])
    z_asp_raw = poly_sum * r2_unit * r2_unit
    z_asp = jnp.where(jnp.isfinite(z_asp_raw), z_asp_raw, jnp.zeros_like(z_asp_raw))

    return z + z_asp


def _evenasph_f(c: JArray, k: JArray, coefs: JArray, p: JArray):
    return p[:, 2] - _evenasph_sag(c, k, coefs, p[:, 0], p[:, 1])


def _evenasph_df(c: JArray, k: JArray, coefs: JArray, p: JArray):
    """
    TODO: Currently, polynomial starts from 4th order. Zemax seems to start from 2nd order.
          We need generalization!
    """
    temp_unit = 1e-3

    # sphere + conic contribution
    r2 = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1]
    a = safeop.sqrt(1.0 - (k + 1.0) * (c**2) * r2).v
    e = c / a

    df_coef = (2 * jnp.arange(2, coefs.shape[0] + 2)) * coefs
    df_coef_unit = df_coef * (temp_unit ** (2 * jnp.arange(1, coefs.shape[0] + 1)))
    r2_unit = r2 / temp_unit**2

    # original (direct dot product)
    e_asp = safeop.dot(
        df_coef_unit[None],
        r2_unit[:, None] ** jnp.arange(1, coefs.shape[0] + 1)[None],
    ).v

    e_tot = e + e_asp
    return jnp.stack(
        [-e_tot * p[:, 0], -e_tot * p[:, 1], jnp.ones_like(p[:, 0])], axis=-1
    )


def intersect_spencer(
    f_df: Callable[[JArray], Tuple[JArray, JArray]],
    p0: JArray,
    d: JArray,
    eps: float,
    z_dir: int = 1,
):
    """
    NOTE: Inspired by https://github.com/mjhoptics/ray-optics/blob/032ddd1fd60a89e31bd457af5d924d283537142d/src/rayoptics/elem/profiles.py

    From Spencer and Murty, `General Ray-Tracing Procedure <https://doi.org/10.1364/JOSA.52.000672>`
    """

    assert p0.ndim - 1 == 1, "TEMP: This should handle arbitrary batch axes!"
    assert eps >= 0

    class _Carry(NamedTuple):
        s: JArray
        delta: JArray
        defined: BoolArray
        iter: int

    f, df = f_df(p0)
    s = safeop.div(-f, jx.vmap(jnp.dot)(d, df)).v
    delta = jnp.full_like(s, 2 * eps)
    iter = 0

    x0 = _Carry(s, delta, jnp.ones(s.shape, dtype=bool), iter)

    def cond_fun(x: _Carry):
        # NOTE: This may bring difference by batching size. But it'd be very subtle.
        return jnp.any(x.delta > eps) & (x.iter < 1000)

    def body_fun(x: _Carry):
        s = x.s
        p_ = p0 + s[:, None] * d
        f, df = f_df(p_)
        a = jx.vmap(safeop.div)(-f, jx.vmap(jnp.dot)(d, df))
        s_ = s + a.v
        delta_ = abs(s_ - s)
        defined_ = x.defined & a.d
        return _Carry(s_, delta_, defined_, x.iter + 1)

    x_final = jx.while_loop(cond_fun, body_fun, x0)
    p_final = p0 + x_final.s[:, None] * d

    return x_final.s, p_final, x_final.defined


@jx.jit
def clamp_abs(x: JArray, max_val: Union[int, float]) -> JArray:
    """
    Clamps the absolute value of elements in a JAX array to a maximum limit.

    If |x| > max_val, the output element y will satisfy |y| = max_val.
    The sign of the original element is preserved.
    """
    # if max_val < 0:
    #    raise ValueError("max_val must be non-negative.")

    x_abs = jnp.abs(x)
    clamped_abs = jnp.clip(x_abs, a_min=None, a_max=max_val)
    result = clamped_abs * jnp.sign(x)

    return result


def intersect_even_asphere(
    c: JArray,
    k: JArray,
    coeffs: JArray,
    ray: rayopt.TLensRay,
    ior: float = 1.0,
):
    """
    TODO: Currently, polynomial starts from 4th order. Zemax seems to start from 2nd order.
          We need generalization!
    """
    assert ray.ndim == 1, "TEMP: This should handle arbitrary batch axes!"

    def f_df(p: JArray):
        f = _evenasph_f(c, k, coeffs, p)
        df = _evenasph_df(c, k, coeffs, p)
        return f, df

    # Find an initial point via spherical intersection.
    # I believe this usually helps to find a good initial guess.

    # NOTE 260404:
    # I believed analytic spherical intersection works well.
    # However, I found some edge cases that spherical surrogate is too small so some rays are improperly blocked.
    # So I replaced it with planar intersect, which is always well-defined (unless ray is perfectly perpendicular).
    # I believe planar initial value also works well. If I find edge cases again, I'll find out a new way.
    t_init, ray_init, _, def_init = intersect_plane(ray, ior)

    t_iter, p, def_iter = intersect_spencer(f_df, ray_init.o, ray_init.d, eps=1e-12)

    # Compute final intersection point and normal
    n = _evenasph_df(c, k, coeffs, p)

    # NOTE: Normdir is dangerous if it didn't converge!
    #       For this case, the ray is blocked in general, so we wouldn't need to compute exact values.
    #       Prevent infinity coordinates by just clipping.
    def_conv = jnp.all(jnp.isfinite(n))
    n = clamp_abs(n, max_val=1e6)

    n, def_n = jx.vmap(safeop.normdir)(n).vd

    # Flip if same direction as ray
    n = jnp.where((jx.vmap(jnp.dot)(n, ray.d) > 0)[:, None], -n, n)

    t = t_init + t_iter
    ray_ = ray.propagate(t, ior=ior)
    defined = def_init & def_iter & def_conv & def_n[..., 0]

    return t, ray_, n, defined
