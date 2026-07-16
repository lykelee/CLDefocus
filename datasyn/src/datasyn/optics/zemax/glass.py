import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pydantic as pyd

from datasyn.optics.zemax.stanza import parse_stanza
from datasyn.utils.files import (
    iter_files_with_exts,
    makedirs_and_chmod,
    read_file_string_auto,
    write_file_string,
)


class NdVdGlass(pyd.BaseModel):
    """
    (Nd, Vd)
    """

    name: str
    Nd_Vd: Tuple[float, float]


class NdVdGlassCatalog(pyd.BaseModel):
    glasses: Dict[str, NdVdGlass]

    def __getitem__(self, name: str) -> Optional[NdVdGlass]:
        return self.glasses.get(name, None)

    def names(self):
        return sorted(self.glasses.keys())


def read_catalog(file: os.PathLike):
    j, _ = read_file_string_auto(file)
    catalog = NdVdGlassCatalog.model_validate_json(j)
    return catalog


def write_catalog(file: os.PathLike, catalog: NdVdGlassCatalog):
    j = catalog.model_dump_json(indent=4)
    write_file_string(file, j)


def load_agf(text: str):
    """Load a .agf file and return its parsed representation."""
    parsed = parse_stanza(text)

    mats: Dict[str, NdVdGlass] = {}
    d = {}

    def flush():
        name = d["name"]
        mat = NdVdGlass(name=name, Nd_Vd=d["Nd_Vd"])
        mats[name] = mat
        d.clear()

    for entry in parsed:
        k, p = entry.keyword, entry.params

        match k:
            case "NM":
                if len(d) != 0:  # Finish the previous material
                    flush()

                assert len(p) >= 2
                d["name"] = p[0]
                d["Nd_Vd"] = (p[3], p[4])

    flush()
    return mats


def main():
    from tqdm.auto import tqdm

    dir = Path("/data/to/zemax/glass/catalogs")
    save_file = Path("./temp/glass_catalog.json")

    glasses = {}

    files = iter_files_with_exts(dir, [".AGF", ".agf"])
    for file in tqdm(files):
        agf, _ = read_file_string_auto(file)
        glasses = glasses | load_agf(agf)

    glasses = dict(sorted(glasses.items()))
    catalog = NdVdGlassCatalog(glasses=glasses)

    makedirs_and_chmod(save_file.parent, exist_ok=True)
    write_catalog(save_file, catalog)

    catalog_loaded = read_catalog("./temp/glass_catalog.json")

    print("Done")


if __name__ == "__main__":
    raise SystemExit(main())
