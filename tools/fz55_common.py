"""Shared FZ55 diagnostic-RAW helpers for calibration and DNG conversion."""

import re
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 4608, 3456
WHITE_LEVEL = 4095  # 12-bit samples; the dump is already black-subtracted


def read_raw(path):
    """Return the 12-bit mosaic as uint16 (rows x cols), validating size."""
    data = Path(path).read_bytes()
    expected = WIDTH * HEIGHT * 2
    if len(data) != expected:
        raise ValueError(f"{path} is {len(data)} bytes, expected {expected}; "
                         "incomplete captures cannot be converted")
    mosaic = np.frombuffer(data, dtype=">u2").reshape(HEIGHT, WIDTH)
    if mosaic.max() > WHITE_LEVEL:
        raise ValueError("Samples exceed the observed FZ55 12-bit range")
    return mosaic


def warp_norm(cx, cy):
    """DNG WarpRectilinear radius normaliser: centre to farthest corner."""
    return max(np.hypot(x - cx, y - cy)
               for x in (0, WIDTH - 1) for y in (0, HEIGHT - 1))


def model_points(params, pts):
    """Corrected (JPEG) pixel -> sensor pixel, DNG WarpRectilinear form.

    params: (cx, cy, scale, k1, k2, k3) with the centre in pixels.
    """
    cx, cy, scale, k1, k2, k3 = params
    m = warp_norm(cx, cy)
    dx = (pts[:, 0] - cx) / m
    dy = (pts[:, 1] - cy) / m
    r2 = dx * dx + dy * dy
    f = scale * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    return np.stack([cx + m * f * dx, cy + m * f * dy], 1)


def lens_params(entry):
    """Profile lens entry -> model_points parameters."""
    kr = entry["kr"]
    cx = entry["center"][0] * (WIDTH - 1)
    cy = entry["center"][1] * (HEIGHT - 1)
    return (cx, cy, kr[0], kr[1] / kr[0], kr[2] / kr[0], kr[3] / kr[0])


def aligned_planes(mosaic, params, step=4):
    """Resample R, G, B sensor planes onto a coarse grid of JPEG pixels."""
    import cv2

    ys, xs = np.mgrid[0:HEIGHT // step, 0:WIDTH // step].astype(np.float64)
    pts = np.stack([xs.ravel() * step + (step - 1) / 2,
                    ys.ravel() * step + (step - 1) / 2], 1)
    src = model_points(params, pts)
    mosaic = mosaic.astype(np.float32)
    planes = ((mosaic[0::2, 0::2], 0.0),
              ((mosaic[0::2, 1::2] + mosaic[1::2, 0::2]) / 2, 0.5),
              (mosaic[1::2, 1::2], 1.0))
    out = []
    for plane, offset in planes:
        mx = ((src[:, 0] - offset) / 2).reshape(xs.shape).astype(np.float32)
        my = ((src[:, 1] - offset) / 2).reshape(xs.shape).astype(np.float32)
        out.append(cv2.remap(cv2.blur(plane, (2, 2)), mx, my,
                             cv2.INTER_LINEAR))
    return np.stack(out, -1)


def tone(x, knots, values):
    """Camera tone curve: linear value -> JPEG code value (0-1)."""
    return np.interp(np.log2(np.clip(x, 1e-6, None)), knots, values)


def find_companion_jpeg(raw_path):
    """The camera saves IMGnnnn.RAW, then the JPEG as ddd_(nnnn+1).JPG.

    On the card the RAW sits at the root and the JPEG in DCIM/dddKFZ55, so
    both the RAW's folder and any DCIM subfolders beside it are searched.
    """
    raw_path = Path(raw_path)
    match = re.fullmatch(r"IMG(\d{4})", raw_path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer JPEG for {raw_path.name}; use --jpeg")
    number = int(match.group(1)) + 1
    folders = [raw_path.parent]
    dcim = raw_path.parent / "DCIM"
    if dcim.is_dir():
        folders += sorted(p for p in dcim.iterdir() if p.is_dir())
    found = [p for folder in folders for p in folder.iterdir()
             if p.suffix.lower() == ".jpg" and not p.name.startswith("._")
             and re.fullmatch(rf"\d{{3}}_{number:04d}", p.stem)]
    if len(found) != 1:
        raise ValueError(f"No unique JPEG numbered {number:04d} next to "
                         f"{raw_path.name}; use --jpeg")
    return found[0]


def read_jpeg_exif(path):
    """Shooting metadata from the camera JPEG (the RAW dump has none)."""
    from PIL import Image

    exif = Image.open(path).getexif()
    sub = exif.get_ifd(0x8769)

    def text(value):
        return value.strip().rstrip("\x00").strip() if value else None

    def number(value):
        return float(value) if value is not None else None

    flash = sub.get(0x9209)
    return dict(
        make=text(exif.get(0x010F)), model=text(exif.get(0x0110)),
        firmware=text(exif.get(0x0131)),
        datetime=text(sub.get(0x9003) or exif.get(0x0132)),
        exposure_time=number(sub.get(0x829A)),
        f_number=number(sub.get(0x829D)),
        iso=sub.get(0x8827), focal_length=number(sub.get(0x920A)),
        focal_length_35mm=sub.get(0xA405),
        flash=flash, flash_fired=bool(flash & 1) if flash is not None else None,
    )
