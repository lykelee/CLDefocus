"""
For each lens (from JSON config), compute a (field * defocus) grid of PSFs and
save per-lens outputs named by the lens:
  - NPZ: the PSF grid itself + field/defocus axes.
  - PNG: a grid visualization.

Config JSON structure:
  {
    "wvl": <meters>, "pixsize": <meters>, "ks": <int>, "Nx": <int>, "Ny": <int>,
    "lenses": [
      {
        "name": "<lens>",
        "fields": {"type": "angle"|"height", "values": [...]},
        "defocus": [<meters>, ...]
      },
      ...
    ]
  }

`fields` x `defocus` form the 2D grid.
- field type "angle":  values are degrees.
- field type "height": values are the normalized field height ([-1, 1]).
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import datasyn.optics.zernike as zernlib
from datasyn.jaxutils.imgutils import make_grid
from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
from datasyn.optics.imaging.imaging import (
    DebyeProper,
    make_imaging,
    wh_physics_to_image,
)
from datasyn.utils.files import ensure_clean_directory, ensure_directory
from datasyn.utils.img.convert import arr1f_to_pil
from datasyn.utils.img.shift import align_to_centroid_dft_layout
from datasyn.utils.parallel.easy_tqdm import easy_tqdm
from datasyn.utils.time import easy_timer


def unique_keep_order(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()

    parser = argparse.ArgumentParser(description="Compute PSF grids.")
    parser.add_argument(
        "--lens-root",
        type=Path,
        help="Directory holding <name>.mytable lens files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Lens/field/defocus + global-params JSON config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output root (image/ + npz/ subdirs).",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    wvl = float(cfg["wvl"])
    pixsize = float(cfg["pixsize"])
    ks = int(cfg["ks"])
    Nx, Ny = int(cfg["Nx"]), int(cfg["Ny"])
    lenses = cfg["lenses"]

    zer_cfg = zernlib.make_zernike_config(n_max=15, kind=zernlib.ZernikeKind.REAL)
    proper = DebyeProper(zer_cfg=zer_cfg)

    save_dir_img = Path(args.out) / "image"
    save_dir_npz = Path(args.out) / "npz"
    ensure_clean_directory(save_dir_img)
    ensure_clean_directory(save_dir_npz)

    failed = []

    for entry in easy_tqdm(lenses):
        name = entry["name"]
        ftype = entry["fields"]["type"]
        fvalues = unique_keep_order([float(x) for x in entry["fields"]["values"]])
        defocuses = unique_keep_order([float(x) for x in entry["defocus"]])
        nF, nD = len(fvalues), len(defocuses)

        with easy_timer(f"Load lens {name}"):
            lens = load_tabular_lens_from_mytable(args.lens_root / f"{name}.mytable")

        imaging = make_imaging(
            lens=lens,
            sen_wh=(Nx, Ny),
            wvl_ref=wvl,
            proper=proper,
            pixsize=pixsize,
        )

        grid_psf = np.zeros((nF, nD, ks, ks), dtype=np.float32)
        grid_viz = np.zeros((nF, nD, ks, ks), dtype=np.float32)

        for i, fval in enumerate(fvalues):
            try:
                # Normalized field height: from angle (deg) or given directly.
                if ftype == "angle":
                    height = imaging.imghelp.field_of_angle(jnp.deg2rad(fval))
                elif ftype == "height":
                    height = fval
                else:
                    raise ValueError(f"Unknown field type: {ftype!r}")
                f_xy = jnp.asarray([0.0, -height])

                # Wavefront: once per field, reused across all defocus.
                with easy_timer("Generate Wavefront"):
                    wf_out = imaging.generate_wavefront(1e3, f_xy, wvl=wvl)
            except Exception:
                failed.append(f"FAILED wavefront: lens={name}, field={fval}")
                continue

            for j, defocus in enumerate(defocuses):
                try:
                    sensor_s = lens.imgpos - imaging.xp.z + defocus
                    with easy_timer("Compute PSF"):
                        out = imaging.propagate_wavefront(
                            wf_out, s_prop=sensor_s, ks=ks
                        )
                        I_psf = align_to_centroid_dft_layout(out.I_psf, cval=0.0)
                        I_psf = I_psf / I_psf.sum()  # SUM normalized
                        I_psf = wh_physics_to_image(I_psf, 0, 1)

                    grid_psf[i, j] = np.asarray(I_psf, dtype=np.float32)

                    I_viz = I_psf / I_psf.max()
                    grid_viz[i, j] = np.asarray(I_viz)

                except Exception:
                    failed.append(
                        f"FAILED psf: lens={name}, field={fval}, defocus={defocus}"
                    )

        with easy_timer(f"Save {name}"):
            file = save_dir_npz / f"{name}.npz"
            ensure_directory(file.parent)
            np.savez(
                file,
                psf=grid_psf,
                fields=np.asarray(fvalues, dtype=np.float32),
                field_type=ftype,
                defocus=np.asarray(defocuses, dtype=np.float32),
            )
            montage = make_grid(jnp.asarray(grid_viz), border_width=2, border_color=1.0)
            file = save_dir_img / f"{name}.png"
            ensure_directory(file.parent)
            arr1f_to_pil(montage).save(save_dir_img / f"{name}.png")

    for fail in failed:
        print(fail)
    print(f"Total failed: {len(failed)}")
    print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
