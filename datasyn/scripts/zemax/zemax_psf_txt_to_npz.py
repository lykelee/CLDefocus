"""
Convert a PSF computed by Zemax in txt form to a npz file.
In Zemax's PSF tools, you can export txt data by switching the bottom tab to "Text" and press the save button.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


MARKER = "Values are relative intensity."
TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252")


def _parse_float_token(token: str) -> float:
    """Parse Zemax numeric tokens such as ``6.078XE-06``."""
    normalized = token.strip().replace("X", "").replace("x", "")
    return float(normalized)


def _read_text_with_marker(txt_path: Path) -> str:
    """Read text using common Zemax export encodings."""
    raw = txt_path.read_bytes()
    fallback = raw.decode("utf-8", errors="replace")

    for encoding in TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if MARKER in text:
            return text

    return fallback


def parse_zemax_psf_txt(txt_path: Path) -> tuple[np.ndarray, str]:
    """
    Parse a Zemax Huygens PSF text export.

    Returns:
        A tuple of ``(psf, header)``. ``psf`` is a 2D float64 array containing
        the relative-intensity image, and ``header`` is the text before the
        numeric data block.
    """
    lines = _read_text_with_marker(txt_path).splitlines()
    marker_idx = next(
        (idx for idx, line in enumerate(lines) if MARKER in line),
        None,
    )
    if marker_idx is None:
        raise ValueError(f"Marker not found: {MARKER!r}")

    rows: list[list[float]] = []
    started = False
    for line_no, line in enumerate(lines[marker_idx + 1 :], start=marker_idx + 2):
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue

        tokens = stripped.split()
        try:
            row = [_parse_float_token(token) for token in tokens]
        except ValueError as exc:
            if started:
                raise ValueError(
                    f"Non-numeric line after data at line {line_no}"
                ) from exc
            continue

        rows.append(row)
        started = True

    if not rows:
        raise ValueError("No numeric PSF rows found after marker.")

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("PSF data is not a rectangular 2D array.")

    header = "\n".join(lines[: marker_idx + 1])
    return np.asarray(rows, dtype=np.float64), header


def downsample2d_poly(x, factor):
    """
    Downsample a 2D signal by an integer factor using polyphase FIR filtering.
    Works for real or complex arrays.
    """
    from scipy.signal import resample_poly

    y = resample_poly(x, up=1, down=factor, axis=0)
    y = resample_poly(y, up=1, down=factor, axis=1)
    return y


def convert_file(
    txt_path: Path,
    input_dir: Path,
    output_dir: Path,
    downsample: int,
) -> tuple[Path, Path]:
    """Convert one text file to ``.npz`` and normalized ``.png`` files."""
    from datasyn.utils.img.convert import arr1f_to_pil

    psf, header = parse_zemax_psf_txt(txt_path)
    rel_path = txt_path.relative_to(input_dir)
    out_path = output_dir / rel_path.with_suffix(".npz")
    png_path = output_dir / rel_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if downsample > 1:
        psf = downsample2d_poly(psf, downsample)

    np.savez_compressed(
        out_path,
        psf=psf,
        source_path=str(txt_path),
        header=header,
    )
    arr1f_to_pil(psf / psf.max()).save(png_path)
    return out_path, png_path


def iter_input_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    """Collect input files from a directory."""
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Zemax Huygens PSF text exports to compressed NPZ files. "
            f"The parser reads the 2D array after {MARKER!r}."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing txt files.")
    parser.add_argument("output_dir", type=Path, help="Directory to write npz files.")
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Input filename glob pattern. Default: *.txt",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input_dir and preserve relative subdirectories.",
    )
    parser.add_argument("--downsample", type=int, default=1)

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(
            f"input_dir does not exist or is not a directory: {args.input_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = iter_input_files(args.input_dir, args.pattern, args.recursive)
    if not input_files:
        raise SystemExit(f"No files matched {args.pattern!r} under {args.input_dir}")

    n_ok = 0
    failures: list[tuple[Path, str]] = []
    for txt_path in input_files:
        try:
            out_path, png_path = convert_file(
                txt_path, args.input_dir, args.output_dir, args.downsample
            )
        except Exception as exc:
            failures.append((txt_path, repr(exc)))
            print(f"[fail] {txt_path}: {exc}", file=sys.stderr)
            continue

        n_ok += 1
        print(f"[done] {txt_path.name} -> {out_path}, {png_path}")

    print(f"[summary] converted {n_ok}/{len(input_files)} files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
