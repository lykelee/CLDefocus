from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from datasyn.utils.files import get_files_by_names, create_symlink


def main():
    parser = argparse.ArgumentParser(description="Make a view for DPDD RAW files.")
    parser.add_argument(
        "--png-root",
        type=Path,
        help="PNG DPDD root.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        help="RAW file root.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        help="RAW file root.",
    )

    args = parser.parse_args()
    png_root = args.png_root
    src_root = args.src_root
    dst_root = args.dst_root

    names = get_files_by_names(
        png_root,
        pred=lambda x: x.parts[-2] == "target" and x.suffix == ".png",
        remove_exts=True,
    )

    for name, png in tqdm(names.items()):
        src = src_root / (name + ".CR2")
        dst = dst_root / (png.parts[-3] + "_target") / (name + ".CR2")
        create_symlink(src, dst, relative=True)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
