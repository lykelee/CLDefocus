"""
Convert ZMX formats to mytable formats, which is used in our codes.
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from datasyn.utils.files import walk_and_apply, makedirs_and_chmod, IterFilesOut
from datasyn.optics.zemax.zmx_to_mytable import zmx_to_mytable_file
from datasyn.optics.zemax.glass import NdVdGlassCatalog, read_catalog


def convert_zmx_to_mytable(src_root: Path, dst_root: Path, catalog: NdVdGlassCatalog):
    src_root = src_root.resolve()

    stat_success = 0
    stat_fail = 0

    def step(out: IterFilesOut):
        nonlocal stat_success
        nonlocal stat_fail

        try:
            src_path = out.path.resolve()
            rel_path = src_path.relative_to(src_root)
            dst_rel = rel_path.with_suffix(".mytable")
            dst_path = dst_root / dst_rel
            makedirs_and_chmod(dst_path.parent, exist_ok=True)
            zmx_to_mytable_file(src_path, dst_path, catalog)
            stat_success += 1
        except Exception as e:
            stat_fail += 1
            print(e)
            print(f"File: {src_path}")

    walk_and_apply(
        src_root, step, exts=[".zmx"], loop_wrapper=tqdm, aware_of_count=True
    )

    stat_total = stat_success + stat_fail
    print("-" * 70)
    print("Finished")
    print(
        f"Stats: total = {stat_total}, succeeded = {stat_success}, failed = {stat_fail}"
    )


def main():
    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()

    parser = argparse.ArgumentParser(description="Convert ZMX format to mytable.")
    parser.add_argument(
        "zmx_root",
        type=Path,
        help="ZMX file root.",
    )
    parser.add_argument(
        "dst_root",
        type=Path,
        help="Directory to save converted mytable files.",
    )
    parser.add_argument(
        "--glass-catalog",
        type=Path,
        default=Path(__file__).parent / "zemax_glass_catalog.json",
        help="Zemax glass catalog json.",
    )

    args = parser.parse_args()
    src_root = args.zmx_root
    dst_root = args.dst_root
    catalog_file = args.glass_catalog

    catalog = read_catalog(catalog_file)
    convert_zmx_to_mytable(src_root, dst_root, catalog)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
