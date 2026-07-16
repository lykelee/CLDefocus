from __future__ import annotations

import argparse
from pathlib import Path

from datasyn.pipeline import Pipeline
from datasyn.synthesis.lens.classify import make_classify_subprocess_node
from datasyn.synthesis.lens.stats import StatsLensNode


def main():
    from datasyn.jaxutils.configs import set_preallocation

    set_preallocation(False)

    parser = argparse.ArgumentParser(description="Analyze lens designs.")
    parser.add_argument(
        "lens_root",
        type=Path,
        help="Lens file (*.mytable) root.",
    )
    parser.add_argument(
        "out_root",
        type=Path,
        help="Output root.",
    )

    args = parser.parse_args()
    lens_root = args.lens_root
    out_root = args.out_root

    classify_path = out_root / "classify"
    stats_path = out_root / "stats"

    classify_node = make_classify_subprocess_node(
        name="classify",
        store_dir=classify_path,
        deps=[],
        lens_root=lens_root,
        classify_db_path=classify_path / "data.db",
        n_workers=2,
    )
    stats_node = StatsLensNode(
        name="stats",
        store_dir=stats_path,
        deps=[],
        lens_root=lens_root,
        stats_db_path=stats_path / "data.db",
    )

    pipe = Pipeline(
        [
            classify_node,
            stats_node,
        ]
    )
    pipe.run_batch(fail_fast=True)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
