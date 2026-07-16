import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from datasyn.jaxutils.imgutils import arr1f_to_pil, make_grid
from datasyn.utils.img.shift import align_to_centroid_dft_layout


def main():
    parser = argparse.ArgumentParser(description="Compare with Zemax.")
    parser.add_argument(
        "--psf-npz-root",
        type=Path,
        help="Root of PSFs in npz.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output image file path.",
    )
    args = parser.parse_args()

    root_debye = args.psf_npz_root
    defocus_idx = 2
    names_debye = [
        ("prime/EP3029504_Example04P", 0),
        ("prime/EP3029504_Example04P", 1),
        ("prime/EP3029504_Example04P", 2),
        ("prime/JP2014-026184_Example04P", 0),
        ("prime/JP2014-026184_Example04P", 1),
        ("prime/JP2014-026184_Example04P", 2),
        ("prime/US20150241657_Example01P", 0),
        ("prime/US20150241657_Example01P", 1),
        ("prime/US20150241657_Example01P", 2),
    ]

    root_zemax = Path("./zemax")
    names_zemax = """EP3029504_Example04P__f1
EP3029504_Example04P__f2
EP3029504_Example04P__f3
JP2014-026184_Example04P__f1
JP2014-026184_Example04P__f2
JP2014-026184_Example04P__f3
US20150241657_Example01P__f1
US20150241657_Example01P__f2
US20150241657_Example01P__f3"""

    names_zemax = names_zemax.splitlines()

    # Cache per-lens grid npz
    _debye_cache = {}

    def load_debye(lens: str, field_idx: int):
        if lens not in _debye_cache:
            _debye_cache[lens] = np.load(root_debye / (lens + ".npz"))["psf"]
        return _debye_cache[lens][field_idx, defocus_idx]

    grid = []

    for name_zemax, (lens_debye, field_idx) in tqdm(
        list(zip(names_zemax, names_debye))
    ):
        psf_zemax = np.load(root_zemax / (name_zemax + ".npz"))["psf"]
        psf_debye = load_debye(lens_debye, field_idx)

        psf_zemax = align_to_centroid_dft_layout(psf_zemax, cval=0.0)
        psf_debye = align_to_centroid_dft_layout(psf_debye, cval=0.0)

        psf_zemax = psf_zemax[16:-16, 16:-16]

        psf_zemax = psf_zemax / psf_zemax.sum()
        psf_debye = psf_debye / psf_debye.sum()

        scale = 1 / max(np.max(psf_zemax), np.max(psf_debye))
        col = np.stack([psf_zemax, psf_debye], axis=0)
        col = scale * col
        grid.append(col)

    grid = np.stack(grid, axis=1)

    plot = make_grid(grid, border_width=2, border_color=1)
    arr1f_to_pil(plot).save(args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
