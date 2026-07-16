from typing import NamedTuple

import jax.numpy as jnp

from datasyn.optics.paraxial import RTM2, Pupil


class PupilModel(NamedTuple):
    """
    A 5-parameter system definition with EP, XP, and optical power.
    This fully determines a paraxial system.
    """

    ep: Pupil
    xp: Pupil
    op: float  # Optical power

    @property
    def rtm(self):
        return RTM2(
            M=jnp.array(
                [[self.xp.r / self.ep.r, 0], [-self.op, self.ep.r / self.xp.r]]
            ),
            z_enter=self.ep.z,
            z_exit=self.xp.z,
        )

    def projmat(self, z_sen: float):
        s_sen = z_sen - self.xp.z
        invmp = self.ep.r / self.xp.r
        fp = s_sen * invmp
        return jnp.array([[fp, 0, 0, 0], [0, fp, 0, 0], [0, 0, -1, self.ep.z]])

    def obj2img_rel(self, s_obj: float):
        """Relative to entrance/exit pupils"""
        rtm = self.rtm
        return rtm.x2z(rtm.obj2img(rtm.z2e(s_obj + self.ep.z))[0]) - self.xp.z

    def sgncoc(self, s_img: float, s_obs: float):
        return self.xp.r * (s_obs - s_img) / s_img
