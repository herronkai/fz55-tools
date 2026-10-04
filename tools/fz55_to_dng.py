#!/usr/bin/env python3
"""Convert an FZ55 diagnostic RAW into a DNG that Lightroom can develop.

The DNG keeps the untouched 12-bit Bayer mosaic and adds:
  * colour calibration (ColorMatrix1) and white balance (AsShotNeutral),
  * the lens distortion profile as a WarpRectilinear opcode, which Adobe
    Camera Raw / Lightroom apply automatically, so framing matches the
    camera JPEG,
  * shooting metadata copied from the camera's companion JPEG.

The RAW dump carries no metadata, so the companion JPEG (saved by the camera
as the next file number) supplies focal length, exposure and per-shot white
balance. Without it the DNG still converts, using profile defaults and no
distortion correction unless --focal-length is given.

Requires NumPy and Pillow; per-shot white balance also needs SciPy and
OpenCV (otherwise the profile's default white balance is used).
"""

import argparse
import json
import struct
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from fz55_common import (
    HEIGHT, WIDTH, find_companion_jpeg, lens_params, read_jpeg_exif,
    read_raw, tone,
)

SOFTWARE = "canary-firmware fz55_to_dng 1.0"
CFA_CODES = {"R": 0, "G": 1, "B": 2}
# Kodak PIXPRO FZ55: 5.1-25.5 mm, f/3.9-6.3
LENS_INFO = (5.1, 25.5, 3.9, 6.3)

BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SRATIONAL, FLOAT = (
    1, 2, 3, 4, 5, 7, 10, 11)


def rational(value, signed=False, limit=1000000):
    frac = Fraction(value).limit_denominator(limit)
    lo, hi = (-2 ** 31, 2 ** 31 - 1) if signed else (0, 2 ** 32 - 1)
    while not (lo <= frac.numerator <= hi and frac.denominator <= 2 ** 32 - 1):
        limit //= 10
        frac = Fraction(value).limit_denominator(limit)
    return frac.numerator, frac.denominator


class Ifd:
    """Collects TIFF entries; values are packed little-endian."""

    def __init__(self):
        self.entries = {}

    def add(self, tag, kind, values):
        if kind == ASCII:
            data = values.encode("ascii", "replace") + b"\0"
            self.entries[tag] = (kind, len(data), data)
            return
        if kind == UNDEFINED or (kind == BYTE and isinstance(values, bytes)):
            self.entries[tag] = (kind, len(values), bytes(values))
            return
        if not isinstance(values, (list, tuple)):
            values = [values]
        if kind in (RATIONAL, SRATIONAL):
            fmt = "<ii" if kind == SRATIONAL else "<II"
            data = b"".join(struct.pack(fmt, *rational(v, kind == SRATIONAL))
                            for v in values)
        else:
            fmt = {BYTE: "<B", SHORT: "<H", LONG: "<I", FLOAT: "<f"}[kind]
            data = b"".join(struct.pack(fmt, v) for v in values)
        self.entries[tag] = (kind, len(values), data)

    def size(self):
        extra = sum(len(d) + len(d) % 2 for _, _, d in self.entries.values()
                    if len(d) > 4)
        return 2 + 12 * len(self.entries) + 4 + extra

    def pack(self, offset, next_ifd=0):
        """Serialise this IFD at absolute file `offset`."""
        tags = sorted(self.entries)
        head = struct.pack("<H", len(tags))
        extra = b""
        extra_at = offset + 2 + 12 * len(tags) + 4
        for tag in tags:
            kind, count, data = self.entries[tag]
            if len(data) <= 4:
                field = data.ljust(4, b"\0")
            else:
                field = struct.pack("<I", extra_at + len(extra))
                extra += data + b"\0" * (len(data) % 2)
            head += struct.pack("<HHI", tag, kind, count) + field
        return head + struct.pack("<I", next_ifd) + extra


def warp_opcode_list(entry):
    """OpcodeList3 holding a single WarpRectilinear (DNG 1.3, opcode 1)."""
    params = struct.pack(">I", 1)  # one coefficient set for all planes
    params += struct.pack(">6d", *entry["kr"], 0.0, 0.0)  # no tangential
    params += struct.pack(">2d", *entry["center"])
    optional = 1  # readers without opcode support may still open the file
    opcode = struct.pack(">IIII", 1, 0x01030000, optional, len(params))
    return struct.pack(">I", 1) + opcode + params


def lens_entry(profile, focal_length):
    """Exact, or linearly interpolated, lens entry for a focal length."""
    lenses = profile.get("lens", [])
    for entry in lenses:
        if abs(entry["focal_length"] - focal_length) <= 0.05:
            return entry, "calibrated"
    below = [e for e in lenses if e["focal_length"] < focal_length]
    above = [e for e in lenses if e["focal_length"] > focal_length]
    if not below or not above:
        return None, None
    a, b = below[-1], above[0]
    t = (focal_length - a["focal_length"]) / (b["focal_length"] - a["focal_length"])
    mix = lambda u, v: [x + t * (y - x) for x, y in zip(u, v)]
    return dict(focal_length=focal_length, kr=mix(a["kr"], b["kr"]),
                center=mix(a["center"], b["center"])), (
        f"interpolated {a['focal_length']}-{b['focal_length']} mm")


def _camera_on_jpeg_grid(mosaic, jpeg, entry):
    """Sensor RGB resampled onto the (downscaled) JPEG's pixel grid.

    Uses the lens profile when there is one; otherwise a feature-matched
    homography, which is close enough for white balance at any zoom.
    """
    import cv2
    from fz55_common import aligned_planes

    h, w = jpeg.shape[:2]
    step = WIDTH // w
    if entry is not None:
        cam = aligned_planes(mosaic, lens_params(entry), step=step)
        return cv2.resize(cam, (w, h), interpolation=cv2.INTER_AREA)
    planes = np.stack([mosaic[0::2, 0::2],
                       (mosaic[0::2, 1::2].astype(np.float32)
                        + mosaic[1::2, 0::2]) / 2,
                       mosaic[1::2, 1::2]], -1).astype(np.float32)
    big = cv2.resize(planes, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
    # bicubic overshoot can go negative, which would turn ** 2.2 into NaN
    ref = np.clip(cv2.resize(jpeg, (w * 2, h * 2),
                             interpolation=cv2.INTER_CUBIC), 0, 1)

    def gray8(x):
        x = x.mean(-1)
        x = (x / max(np.percentile(x, 99.5), 1e-6)) ** (1 / 2.2)
        return cv2.createCLAHE(3.0, (8, 8)).apply(np.uint8(np.clip(x, 0, 1) * 255))

    sift = cv2.SIFT_create(4000)
    kc, dc = sift.detectAndCompute(gray8(big), None)
    kj, dj = sift.detectAndCompute(gray8(ref ** 2.2), None)
    if dc is None or dj is None:
        return None
    pairs = [m for m, n in cv2.BFMatcher().knnMatch(dj, dc, k=2)
             if m.distance < 0.75 * n.distance]
    if len(pairs) < 20:
        return None
    src = np.float32([kj[m.queryIdx].pt for m in pairs])
    dst = np.float32([kc[m.trainIdx].pt for m in pairs])
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None:
        return None
    cam = cv2.warpPerspective(big, homography, (w * 2, h * 2),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_AREA)


def shot_neutral(mosaic, jpeg, entry, colour):
    """Recover this shot's camera white balance from its JPEG.

    `jpeg` is the camera JPEG as float RGB (0-1), at any reduced size. The
    camera's own idea of white is read from the pixels it rendered neutral:
    their sensor RGB is, by definition, AsShotNeutral, so Lightroom renders
    exactly those areas neutral too. A whole-image fit is the fallback for
    scenes with no neutral areas; it relies on the single-illuminant colour
    matrix and drifts green away from the calibration light.
    Returns (neutral, method) or (None, None).
    """
    try:
        import cv2
        from scipy.optimize import least_squares
    except ImportError:
        return None, None
    cam = _camera_on_jpeg_grid(mosaic, jpeg, entry)
    if cam is None:
        return None, None
    cam = cam / 4095.0
    lab = cv2.cvtColor(jpeg.astype(np.float32), cv2.COLOR_RGB2LAB)
    exposed = ((jpeg.max(-1) < 0.95) & (jpeg.min(-1) > 0.04)
               & (cam.max(-1) < 0.9) & (cam.min(-1) > 4 / 4095))
    gray = (exposed & (lab[..., 0] > 20) & (lab[..., 0] < 90)
            & (np.hypot(lab[..., 1], lab[..., 2]) < 5))
    if gray.sum() >= max(50, gray.size // 2000):
        neutral = np.median(cam[gray] / cam[gray][:, 1:2], axis=0)
        return neutral / neutral.max(), f"neutral areas ({gray.mean():.0%} of frame)"

    if exposed.sum() < 500:
        return None, None
    m = np.array(colour["jpeg_matrix"])
    knots, values = colour["tone_log2_knots"], colour["tone_values"]
    c, j = cam[exposed], jpeg[exposed]

    def resid(log_gain):
        return (tone((c * np.exp(log_gain)) @ m.T, knots, values) - j).ravel()

    sol = least_squares(resid, np.zeros(3), loss="soft_l1", f_scale=0.03)
    neutral = np.linalg.inv(m) @ np.ones(3) / np.exp(sol.x)
    if np.any(neutral <= 0):
        return None, None
    return neutral / neutral.max(), "whole-image fit (no neutral areas)"


def load_jpeg(path, width=576):
    image = Image.open(path).convert("RGB")
    image = image.resize((width, width * HEIGHT // WIDTH),
                         Image.Resampling.LANCZOS)
    return np.asarray(image, np.float32) / 255


def thumbnail(mosaic, neutral, jpeg_path, width=256):
    height = width * HEIGHT // WIDTH
    if jpeg_path:
        image = Image.open(jpeg_path).convert("RGB")
    else:
        rgb = np.stack([mosaic[0::2, 0::2],
                        (mosaic[0::2, 1::2].astype(np.float32)
                         + mosaic[1::2, 0::2]) / 2,
                        mosaic[1::2, 1::2]], -1) / 4095.0 / np.array(neutral)
        rgb = np.clip(rgb / max(np.percentile(rgb, 99), 1e-6), 0, 1) ** (1 / 2.2)
        image = Image.fromarray(np.uint8(rgb * 255))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image.tobytes(), width, height


def build_dng(mosaic, profile, exif, neutral, lens, thumb, source_name):
    colour = profile["colour"]
    thumb_bytes, tw, th = thumb
    raw_bytes = mosaic.astype("<u2").tobytes()
    make = exif.get("make") or "JK Imaging, Ltd."
    model = exif.get("model") or profile["camera"]
    stamp = exif.get("datetime") or datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    main = Ifd()  # IFD0: thumbnail plus camera-wide DNG tags
    main.add(254, LONG, 1)
    main.add(256, LONG, tw)
    main.add(257, LONG, th)
    main.add(258, SHORT, [8, 8, 8])
    main.add(259, SHORT, 1)
    main.add(262, SHORT, 2)
    main.add(271, ASCII, make)
    main.add(272, ASCII, model)
    main.add(273, LONG, 0)  # patched below
    main.add(274, SHORT, 1)
    main.add(277, SHORT, 3)
    main.add(278, LONG, th)
    main.add(279, LONG, len(thumb_bytes))
    main.add(284, SHORT, 1)
    main.add(305, ASCII, SOFTWARE)
    main.add(306, ASCII, stamp)
    main.add(330, LONG, 0)  # SubIFDs, patched below
    main.add(34665, LONG, 0)  # Exif IFD, patched below
    main.add(50706, BYTE, bytes([1, 4, 0, 0]))
    main.add(50707, BYTE, bytes([1, 3, 0, 0]))  # WarpRectilinear needs 1.3
    main.add(50708, ASCII, "Kodak PIXPRO FZ55 (diagnostic RAW)")
    main.add(50721, SRATIONAL, [v for row in colour["colour_matrix"] for v in row])
    main.add(50728, RATIONAL, list(neutral))
    main.add(50730, SRATIONAL, colour["baseline_exposure"])
    main.add(50736, RATIONAL, list(LENS_INFO))
    main.add(50778, SHORT, colour["calibration_illuminant_code"])
    main.add(50827, BYTE, source_name.encode() + b"\0")  # OriginalRawFileName
    look = profile.get("look")
    if look:
        # Embedded camera profile: Lightroom lists it as "Embedded" and uses
        # it by default. The look table holds per-hue/saturation corrections
        # (hue shift in degrees, saturation scale, value scale) applied in
        # linear ProPhoto HSV, the same mechanism as Adobe's own profiles.
        main.add(50936, ASCII, look["name"])  # ProfileName
        main.add(50941, LONG, 0)  # ProfileEmbedPolicy: allow copying
        main.add(50981, LONG, list(look["dims"]))  # ProfileLookTableDims
        main.add(50982, FLOAT, [v for cell in look["data"] for v in cell])
        main.add(51108, LONG, 0)  # ProfileLookTableEncoding: linear

    ex = Ifd()
    ex.add(36864, UNDEFINED, b"0230")
    if exif.get("exposure_time"):
        ex.add(33434, RATIONAL, exif["exposure_time"])
    if exif.get("f_number"):
        ex.add(33437, RATIONAL, exif["f_number"])
    if exif.get("iso"):
        ex.add(34855, SHORT, int(exif["iso"]))
    if exif.get("datetime"):
        ex.add(36867, ASCII, exif["datetime"])
        ex.add(36868, ASCII, exif["datetime"])
    if exif.get("flash") is not None:
        ex.add(37385, SHORT, int(exif["flash"]))
    if exif.get("focal_length"):
        ex.add(37386, RATIONAL, exif["focal_length"])
    if exif.get("focal_length_35mm"):
        ex.add(41989, SHORT, int(exif["focal_length_35mm"]))
    ex.add(42035, ASCII, "Kodak")  # LensMake
    ex.add(42036, ASCII, "FZ55 5.1-25.5 mm f/3.9-6.3")  # LensModel

    raw = Ifd()
    pattern = [CFA_CODES[c] for c in profile["cfa_pattern"]]
    raw.add(254, LONG, 0)
    raw.add(256, LONG, WIDTH)
    raw.add(257, LONG, HEIGHT)
    raw.add(258, SHORT, 16)
    raw.add(259, SHORT, 1)
    raw.add(262, SHORT, 32803)  # CFA
    raw.add(273, LONG, 0)  # patched below
    raw.add(277, SHORT, 1)
    raw.add(278, LONG, HEIGHT)
    raw.add(279, LONG, len(raw_bytes))
    raw.add(284, SHORT, 1)
    raw.add(33421, SHORT, [2, 2])
    raw.add(33422, BYTE, bytes(pattern))
    raw.add(50710, BYTE, bytes([0, 1, 2]))
    raw.add(50711, SHORT, 1)
    raw.add(50714, LONG, profile["black_level"])
    raw.add(50717, LONG, profile["white_level"])
    raw.add(50719, LONG, [0, 0])
    raw.add(50720, LONG, [WIDTH, HEIGHT])
    if lens is not None:
        raw.add(51022, UNDEFINED, warp_opcode_list(lens))

    # Layout: header, IFD0, Exif IFD, raw IFD, thumbnail strip, raw strip.
    main_at = 8
    exif_at = main_at + main.size()
    exif_at += exif_at % 2
    raw_at = exif_at + ex.size()
    raw_at += raw_at % 2
    thumb_at = raw_at + raw.size()
    thumb_at += thumb_at % 2
    data_at = thumb_at + len(thumb_bytes)
    data_at += data_at % 2
    main.add(273, LONG, thumb_at)
    main.add(330, LONG, raw_at)
    main.add(34665, LONG, exif_at)
    raw.add(273, LONG, data_at)

    out = bytearray(b"II*\0" + struct.pack("<I", main_at))
    for at, blob in ((main_at, main.pack(main_at)), (exif_at, ex.pack(exif_at)),
                     (raw_at, raw.pack(raw_at)), (thumb_at, thumb_bytes),
                     (data_at, raw_bytes)):
        out += b"\0" * (at - len(out))
        assert len(out) == at
        out += blob
    return bytes(out)


def convert(raw_path, output, profile_path, jpeg_path, focal_length,
            lens_correction):
    profile = json.loads(profile_path.read_text())
    mosaic = read_raw(raw_path)
    exif = read_jpeg_exif(jpeg_path) if jpeg_path else {}
    if focal_length is not None:
        exif["focal_length"] = focal_length
    notes = []

    lens = how = None
    if not lens_correction:
        notes.append("lens correction disabled")
    elif exif.get("focal_length") is None:
        notes.append("WARNING: no focal length known: lens correction skipped")
    else:
        lens, how = lens_entry(profile, exif["focal_length"])
        if lens is None:
            calibrated = [e["focal_length"] for e in profile.get("lens", [])]
            notes.append(f"WARNING: {exif['focal_length']} mm is outside calibrated "
                         f"focal lengths {calibrated}: lens correction skipped")
        else:
            notes.append(f"lens correction {how} for {exif['focal_length']} mm")

    neutral = None
    if jpeg_path:
        neutral, how = shot_neutral(mosaic, load_jpeg(jpeg_path), lens,
                                    profile["colour"])
        if neutral is not None:
            notes.append(f"white balance from camera JPEG: {how}")
    if neutral is None:
        neutral = profile["colour"]["default_neutral"]
        notes.append("WARNING: white balance is the daylight default"
                     + (" (could not match the camera JPEG)" if jpeg_path
                        else " (no camera JPEG)"))

    dng = build_dng(mosaic, profile, exif, neutral, lens,
                    thumbnail(mosaic, neutral, jpeg_path), raw_path.name)
    output.write_bytes(dng)
    return notes, neutral


def read_dng(path):
    """Read back a DNG written by this tool (mosaic, thumbnail, metadata)."""
    import tifffile

    def ratio(value):
        return value[0] / value[1] if value else None

    with tifffile.TiffFile(path) as tif:
        main_ifd = tif.pages[0]
        tags = main_ifd.tags
        exif = tags[34665].value if 34665 in tags else {}
        raw_ifd = main_ifd.pages[0]
        lens = None
        if 51022 in raw_ifd.tags:
            values = struct.unpack(">8d", bytes(raw_ifd.tags[51022].value)[24:88])
            lens = dict(kr=list(values[:4]), center=list(values[6:8]))
        neutral = np.array(tags[50728].value, float).reshape(3, 2)
        name = tags[50827].value if 50827 in tags else b""
        return dict(
            mosaic=raw_ifd.asarray(),
            thumb=main_ifd.asarray().astype(np.float32) / 255,
            thumb_bytes=main_ifd.asarray().tobytes(),
            thumb_size=(main_ifd.imagewidth, main_ifd.imagelength),
            # the thumbnail is the camera JPEG exactly when EXIF came with it
            thumb_is_jpeg="ExposureTime" in exif,
            lens=lens,
            neutral=neutral[:, 0] / neutral[:, 1],
            source_name=bytes(name).rstrip(b"\0").decode() or Path(path).stem + ".RAW",
            exif=dict(
                make=tags[271].value, model=tags[272].value,
                datetime=exif.get("DateTimeOriginal"),
                exposure_time=ratio(exif.get("ExposureTime")),
                f_number=ratio(exif.get("FNumber")),
                iso=exif.get("ISOSpeedRatings"),
                focal_length=ratio(exif.get("FocalLength")),
                focal_length_35mm=exif.get("FocalLengthIn35mmFilm"),
                flash=exif.get("Flash"),
            ),
        )


def refresh(dng_path, profile):
    """Rebuild an existing DNG with the current profile and white balance.

    Works without the original RAW/JPEG: the DNG holds the mosaic, and its
    thumbnail is the camera JPEG when one was available at conversion time.
    Returns (old neutral, new neutral, white-balance note).
    """
    dng = read_dng(dng_path)
    neutral = how = None
    if dng["thumb_is_jpeg"]:
        neutral, how = shot_neutral(dng["mosaic"], dng["thumb"], dng["lens"],
                                    profile["colour"])
    if neutral is None:
        neutral = profile["colour"]["default_neutral"]
        how = "profile default (no camera JPEG)"
    lens = dng["lens"]
    if lens is None and dng["exif"]["focal_length"]:
        lens, _ = lens_entry(profile, dng["exif"]["focal_length"])
    thumb = (dng["thumb_bytes"], *dng["thumb_size"])
    data = build_dng(dng["mosaic"], profile, dng["exif"], neutral, lens,
                     thumb, dng["source_name"])
    tmp = Path(dng_path).with_suffix(".dng.tmp")
    tmp.write_bytes(data)
    tmp.replace(dng_path)
    return dng["neutral"], neutral, how


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file (one input) or directory")
    parser.add_argument("--jpeg", type=Path,
                        help="companion JPEG (default: next file number)")
    parser.add_argument("--no-jpeg", action="store_true",
                        help="ignore companion JPEGs")
    parser.add_argument("--focal-length", type=float,
                        help="override focal length in mm")
    parser.add_argument("--no-lens-correction", action="store_true")
    parser.add_argument("--profile", type=Path,
                        default=Path(__file__).with_name("fz55_profile.json"))
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert even when the DNG is already up to date")
    parser.add_argument("--refresh", action="store_true",
                        help="inputs are existing DNGs: rebuild them in place "
                             "with the current profile and white balance")
    args = parser.parse_args()
    if args.refresh:
        profile = json.loads(args.profile.read_text())
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        for path in args.raw:
            old, new, how = refresh(path, profile)
            print(f"{path.name}: AsShotNeutral {fmt(old)} -> {fmt(new)} ({how})")
        return
    if args.jpeg and len(args.raw) > 1:
        parser.error("--jpeg applies to a single RAW")

    failed = skipped = 0
    for raw_path in args.raw:
        if args.output and args.output.is_dir():
            out = args.output / (raw_path.stem + ".dng")
        elif args.output and len(args.raw) == 1:
            out = args.output
        else:
            out = raw_path.with_suffix(".dng")
        if (not args.overwrite and out.exists()
                and out.stat().st_mtime >= raw_path.stat().st_mtime):
            skipped += 1
            continue
        jpeg = None
        if not args.no_jpeg:
            try:
                jpeg = args.jpeg or find_companion_jpeg(raw_path)
            except ValueError as error:
                print(f"{raw_path.name}: {error}")
        try:
            notes, neutral = convert(raw_path, out, args.profile, jpeg,
                                     args.focal_length,
                                     not args.no_lens_correction)
        except (ValueError, OSError) as error:
            print(f"{raw_path.name}: cannot convert: {error}")
            failed += 1
            continue
        print(f"{raw_path.name} -> {out}"
              + (f" (metadata from {jpeg.name})" if jpeg else ""))
        for note in notes:
            print(f"  {note}")
        print("  AsShotNeutral " + " ".join(f"{v:.4f}" for v in neutral))
    if skipped:
        print(f"Skipped {skipped} already converted (use --overwrite to redo)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
