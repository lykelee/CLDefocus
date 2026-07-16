"""
Paraxial properties of compound lenses.
"""

from __future__ import annotations

import jax.numpy as jnp

from datasyn.optics.paraxial import Pupil
from datasyn.optics.paraxial.rtm2 import RTM2


def find_ep_xp(
    rtm_front: RTM2, rtm_back: RTM2, stop_radius: float, n_obj: float, n_img: float
):
    rtm_front_rev = rtm_front.rev
    ep_s, ep_mag = rtm_front_rev.obj2img_s0(n_img=n_obj)
    ep_z = rtm_front_rev.x2z(ep_s)
    ep_r = jnp.abs(ep_mag) * stop_radius
    ep = Pupil(z=ep_z, r=ep_r)

    xp_s, xp_mag = rtm_back.obj2img_s0(n_img=n_img)
    xp_z = rtm_back.x2z(xp_s)
    xp_r = jnp.abs(xp_mag) * stop_radius
    xp = Pupil(z=xp_z, r=xp_r)

    return ep, xp
