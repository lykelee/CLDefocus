from dataclasses import dataclass

import jax.numpy as jnp

from datasyn.jaxutils import debug as debugjax
from datasyn.jaxutils import jx
from datasyn.optics.typing import *


def _abcd(a, b, c, d):
    return jnp.array([[a, b], [c, d]])


def _mat_propagation(n: float, z: float):
    return _abcd(1, z / n, 0, 1)


def _mat_refraction(n1: float, n2: float, c: float):
    return _abcd(1, 0, c * (n1 - n2), 1)


def _mat_thinlens(power: float):
    return _abcd(1, 0, -power, 1)


def _mat_inv_det1(M: JArray):
    return jnp.array([[M[..., 1, 1], -M[..., 0, 1]], [-M[..., 1, 0], M[..., 0, 0]]])


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class RTM2:
    """
    An optical mapping is written as an ABCD matrix. This assumes:

    - No propagation flipping (i.e. no reflection)
    - No translation & tilt (i.e. no change of the optical axis)

    This would be suitable for most refractive lenses.

    NOTE: This uses "reduced angle". The matrix is always symplectic (implying det = 1).
          Refer to "(y-nu) method" in Kingslake & Johnson - Lens Design Fundamentals.
    """

    M: JArray  # 2x2, always det M = 1
    z_enter: JArray
    z_exit: JArray

    @property
    def A(self):
        return self.M[..., 0, 0]

    @property
    def B(self):
        return self.M[..., 0, 1]

    @property
    def C(self):
        return self.M[..., 1, 0]

    @property
    def D(self):
        return self.M[..., 1, 1]

    @property
    def rev(self):
        return RTM2(M=_mat_inv_det1(self.M), z_enter=self.z_exit, z_exit=self.z_enter)

    @staticmethod
    def empty():
        """Useful for dummy"""
        return RTM2(M=jnp.identity(2), z_enter=jnp.array(0.0), z_exit=jnp.array(0.0))

    @staticmethod
    def identity(z_enter: JArray = 0.0, z_exit: JArray = 0.0, n: float = 1.0):
        """A RTM that preserves an incoming ray."""
        return RTM2(
            M=_mat_propagation(n, z_exit - z_enter),
            z_enter=jnp.asarray(z_enter),
            z_exit=jnp.asarray(z_exit),
        )

    def z2e(self, z: float):
        """
        Converts an absolute z position to a position relative to the entrance plane.
        """
        return z - self.z_enter

    def e2z(self, e: float):
        """
        Converts a position relative to the entrance plane to an absolute z position.
        """
        return e + self.z_enter

    def z2x(self, z: float):
        """
        Converts an absolute z position to a position relative to the exit plane.
        """
        return z - self.z_exit

    def x2z(self, x: float):
        """
        Converts a position relative to the exit plane to an absolute z position.
        """
        return x + self.z_exit

    def power(self):
        """
        Optical

        NOTE: The exact definition is not settled yet! How to consider IOR?
        """
        return -self.C

    def vl(self):
        """Vertex length: exit z - enter z"""
        return self.z_exit - self.z_enter

    def efl(self):
        """
        Effective focal length

        NOTE: The exact definition is not settled yet! How to consider IOR?
        """
        return 1 / -self.C

    def ffl(self, n_in: float = 1.0):
        """
        Front focal length: the displacement of the front focal point from the entrance plane

        n_in: IOR of incident space
        """
        return n_in * self.D / self.C

    def ffp(self, n_in: float = 1.0):
        """
        Front focal point

        n_in: IOR of incident space
        """
        return self.z_enter + n_in * self.D / self.C

    def bfl(self, n_out: float = 1.0):
        """
        Back focal length: the displacement of the back focal point from the exit plane

        n_out: IOR of outgoing space
        """
        return -n_out * self.A / self.C

    def bfp(self, n_out: float = 1.0):
        """
        Back focal point

        n_out: IOR of outgoing space
        """
        return self.z_exit - n_out * self.A / self.C

    def fpl(self, n_in: float = 1.0):
        """
        Front principal length (displacement of front principal point from the entrance plane)

        n_in: IOR of incident space
        """
        return n_in * (self.D - 1) / self.C

    def fpp(self, n_in: float = 1.0):
        """
        Front principal point

        n_in: IOR of incident space
        """
        return self.z_enter + n_in * (self.D - 1) / self.C

    def bpl(self, n_out: float = 1.0):
        """
        Back principal length (displacement of back principal point from the exit plane)

        n_out: IOR of outgoing space
        """
        return n_out * (1 - self.A) / self.C

    def bpp(self, n_out: float = 1.0):
        """
        Back principal point

        n_out: IOR of outgoing space
        """
        return self.z_exit + n_out * (1 - self.A) / self.C

    def ttl(self, n_out: float = 1.0):
        """Total track length: distance from the entrance to the back focal point."""
        return self.bfp(n_out) - self.z_enter

    def to_thinlens(self):
        """
        Converts this to an equivalent thin-lens.
        The matrix is a thin-lens matrix with the same optical power, and the entrance and exit are the front and back principal points respectively.
        """
        return VirtualThinLens(power=self.power(), fpp=self.fpp(), bpp=self.bpp())

    def obj2img(self, s_obj: float, n_obj: float = 1.0, n_img: float = 1.0):
        """
        Inputs:
        - s_obj: Object z relative to the entrance plane
        - n_obj: Object-side IOR
        - n_img: Image-side IOR

        Return: (s_img, mag)
        - s_img: Image z relative to the exit plane
        - mag: Signed magnification (i.e. y_img = mag * y_obj)
        """
        s_img = (
            n_img
            * (n_obj * self.B - s_obj * self.A)
            / (s_obj * self.C - n_obj * self.D)
        )
        mag = n_obj / (n_obj * self.D - s_obj * self.C)
        return s_img, mag

    def obj2img_s0(self, n_img: float = 1.0):
        """
        If the object relative z = 0, the imaging depends only on image-side IOR (independent of object-side IOR).
        This is particularly useful for pupil calculations (imaging a stop that coincides to the entrance plane).
        """
        s_img = -n_img * self.B / self.D
        mag = 1 / self.D
        return s_img, mag

    def img2obj(self, s_img: float, n_obj: float = 1.0, n_img: float = 1.0):
        """
        TODO: This is not verified!
        """
        s_obj = (
            n_obj
            * (n_img * self.B + s_img * self.D)
            / (n_img * self.A + s_img * self.C)
        )
        mag = n_img / (n_img * self.A + s_img * self.C)
        return s_obj, mag


def concat_rtm(m1: RTM2, m2: RTM2, n_inter: float = 1.0):
    """
    m1 -> m2 (= m2 * m1)

    - n_inter: IOR of medium between m1 and m2
    """
    d = m2.z_enter - m1.z_exit
    M = m2.M @ _mat_propagation(n_inter, d) @ m1.M
    return RTM2(M=M, z_enter=m1.z_enter, z_exit=m2.z_exit)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VirtualThinLens:
    """
    A thin-lens model that allows different entrance and exit planes.
    Every RTM with nonzero optical power is reduced to this form.

    TODO: I didn't consider non-air media on object/image sides yet!

    TODO 260714: VirtualThinLens is outdated! PupilModel is more consistent. Refactor this perhaps in the future.
    """

    power: float
    fpp: float
    bpp: float

    def to_rtm(self):
        return RTM2(M=_mat_thinlens(self.power), z_enter=self.fpp, z_exit=self.bpp)

    def o2i(self, z_obj: float):
        """
        NOTE: z positions are relative to principal planes.
        """
        z_img = 1 / (1 / z_obj + self.power)
        mag = z_img / z_obj
        return z_img, mag

    def i2o(self, z_img: float):
        """
        NOTE: z positions are relative to principal planes.
        """
        z_obj = 1 / (1 / z_img - self.power)
        mag = z_obj / z_img
        return z_obj, mag

    def __getitem__(self, idx):
        return VirtualThinLens(
            power=self.power[idx], fpp=self.fpp[idx], bpp=self.bpp[idx]
        )


def ior_curv_thick_seq(
    ns: Sequence[float], cs: Sequence[float], ts: Sequence[float], z0: float
):
    """
    Converts a sequence of IORs, curvatures, and thicknesses to a RTM.
    This is useful for compound lenses.

    - z0: The z position of the front element.
    """
    ns, cs, ts = map(jnp.asarray, (ns, cs, ts))
    n_surf = cs.size
    debugjax.safe_assert(n_surf > 0)
    debugjax.safe_assert((ns.size == n_surf + 1) and (ts.size == n_surf - 1))

    if n_surf == 0:
        return RTM2.identity()

    M0 = _mat_refraction(ns[0], ns[1], cs[0])

    if n_surf == 1:
        return RTM2(M=M0, z_enter=z0, z_exit=z0)

    def body(i: int, Mz: Tuple[JArray, float]):
        M, dz = Mz
        n1, n2, c, thick = (ns[i], ns[i + 1], cs[i], ts[i - 1])

        dz_ = dz + thick
        M_ = _mat_refraction(n1=n1, n2=n2, c=c) @ _mat_propagation(n=n1, z=thick) @ M
        return M_, dz_

    M, dz = jx.fori_loop(1, n_surf, body, (M0, 0.0))
    return RTM2(M=M, z_enter=z0, z_exit=z0 + dz)
