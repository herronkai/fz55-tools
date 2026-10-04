#!/usr/bin/env python3
"""Recover FZ55 diagnostic samples as a grayscale mosaic and preview.

Requires NumPy and Pillow. This preserves sensor values but does not demosaic
or calibrate colors. The original diagnostic file is never changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def decode(source, output, width, height, allow_partial):
    data = source.read_bytes()
    expected = width * height * 2
    if len(data) % 2:
        raise ValueError("File ends inside a 16-bit sample")
    if len(data) != expected and not allow_partial:
        raise ValueError(
            f"Expected {expected} bytes, found {len(data)}. "
            "Use --allow-partial only to recover an incomplete capture."
        )
    samples = np.frombuffer(data, dtype=">u2")
    rows, trailing = divmod(len(samples), width)
    if rows < 1:
        raise ValueError("Not enough data for one complete row")
    if rows > height:
        raise ValueError("Unexpected data beyond the nominal image")
    if samples.max() > 4095:
        raise ValueError("Samples exceed the observed FZ55 12-bit range")
    mosaic = samples[:rows * width].reshape(rows, width)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / source.stem
    tiff_path = prefix.with_name(prefix.name + "-sensor-mosaic.tiff")
    png_path = prefix.with_name(prefix.name + "-preview.png")
    Image.fromarray(mosaic.astype(np.uint16)).save(
        tiff_path, compression="tiff_deflate"
    )
    # Display-only curve: retain native values in the separate TIFF and RAW.
    display = np.log1p(mosaic.astype(np.float32)) / np.log(4096)
    preview = Image.fromarray(np.uint8(np.clip(display, 0, 1) * 255))
    preview.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    preview.save(png_path)
    info = dict(
        source=str(source.resolve()), sha256=hashlib.sha256(data).hexdigest(),
        source_bytes=len(data), nominal_bytes=expected,
        complete=len(data) == expected, width=width, recovered_rows=rows,
        discarded_partial_row_samples=trailing,
        encoding="12-bit values in big-endian 16-bit words",
        tiff=str(tiff_path.resolve()), preview=str(png_path.resolve()),
        color_rendered=False,
    )
    prefix.with_name(prefix.name + "-decode.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=4608)
    parser.add_argument("--height", type=int, default=3456)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("Image dimensions must be positive")
    try:
        decode(args.source, args.output or args.source.parent,
               args.width, args.height, args.allow_partial)
    except ValueError as error:
        parser.exit(1, f"Cannot decode: {error}\n")
