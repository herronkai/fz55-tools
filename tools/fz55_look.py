#!/usr/bin/env python3
"""Fit and install the FZ55 camera profiles (look tables) for Lightroom.

A bare colour matrix leaves Camera Raw's rendering of this camera unevenly
muted: blues, greens and magentas lose far more colour than skin and
yellows, so the Saturation slider pushes skin orange long before skies look
right. Adobe's own camera profiles fix this with a look table: per-hue and
per-saturation corrections in linear ProPhoto HSV. This tool fits one so
Camera Raw reproduces the camera's colour, with Camera Raw itself in the
loop (Photoshop renders each iteration headlessly).

  fit       fit "Natural" to camera JPEGs and derive "Vivid"; saves both
            into fz55_profile.json (Natural becomes the embedded default)
  install   write .dcp files so Lightroom's profile browser offers
            FZ55 Natural, FZ55 Vivid and FZ55 Matrix (no look table)

Requires Photoshop 2026 (fit only), NumPy, SciPy, OpenCV and tifffile.
"""

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from fz55_to_dng import (
    ASCII, FLOAT, LONG, SHORT, SRATIONAL, Ifd, _camera_on_jpeg_grid,
    read_dng, refresh,
)

HUES, SATS = 36, 8  # look table divisions (10 degree hue steps)
PHOTOSHOP = "Adobe Photoshop 2026"
PROFILE_DIR = (Path.home() / "Library/Application Support/Adobe/CameraRaw"
               / "CameraProfiles")
UNIQUE_MODEL = "Kodak PIXPRO FZ55 (diagnostic RAW)"

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])
D50 = np.array([0.9642, 1.0, 0.8249])
D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_PROPHOTO = np.array([[1.3459433, -0.2556075, -0.0511118],
                            [-0.5445989, 1.5081673, 0.0205351],
                            [0.0, 0.0, 1.2118128]])


def adapt(src, dst):
    return (np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ dst) / (BRADFORD @ src))
            @ BRADFORD)


SRGB_TO_PROPHOTO = XYZ_TO_PROPHOTO @ adapt(D65, D50) @ SRGB_TO_XYZ


def srgb_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def hsv(rgb):
    """DNG-style hexcone HSV: hue in [0, 6), value = max channel."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-9)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, (g - b) / d,
                 np.where(mx == g, 2 + (b - r) / d, 4 + (r - g) / d)) % 6
    return h, s, mx


def identity_table():
    return np.tile([0.0, 1.0, 1.0], (HUES * SATS, 1))


def with_look(profile, table, name):
    profile = json.loads(json.dumps(profile))
    if table is None:
        profile.pop("look", None)
    else:
        profile["look"] = dict(name=name, dims=[HUES, SATS, 1],
                               data=np.round(table, 5).tolist())
    return profile


def render(folder, out_dir):
    """Render every DNG in `folder` through Camera Raw to sRGB JPEGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir.with_suffix(".jsx")
    script.write_text(f"""app.displayDialogs = DialogModes.NO;
var files = Folder("{folder}").getFiles("*.dng");
for (var i = 0; i < files.length; i++) {{
  var d = app.open(files[i]);
  d.convertProfile("sRGB IEC61966-2.1", Intent.RELATIVECOLORIMETRIC, true, false);
  d.bitsPerChannel = BitsPerChannelType.EIGHT;
  d.resizeImage(UnitValue(1152, "px"), UnitValue(864, "px"), 72,
                ResampleMethod.BICUBIC);
  var o = new JPEGSaveOptions(); o.quality = 11;
  d.saveAs(new File("{out_dir}/" + files[i].name.replace(".dng", ".jpg")),
           o, true);
  d.close(SaveOptions.DONOTSAVECHANGES);
}}
files.length;""")
    subprocess.run(["osascript", "-e", "with timeout of 3600 seconds", "-e",
                    f'tell application "{PHOTOSHOP}" to do javascript file '
                    f'(POSIX file "{script}")', "-e", "end timeout"],
                   check=True, capture_output=True)


class Shot:
    """Pre-look-table ProPhoto values and camera-JPEG target for one DNG."""

    def __init__(self, path, profile):
        dng = read_dng(path)
        self.name = Path(path).stem
        self.thumb = dng["thumb"]
        cam = _camera_on_jpeg_grid(dng["mosaic"], self.thumb, dng["lens"]) / 4095
        # Camera Raw's matrix path: inverse ColorMatrix, then adapt the
        # as-shot white to D50, with the neutral at Y = 1
        cam_to_xyz = np.linalg.inv(np.array(profile["colour"]["colour_matrix"]))
        white = cam_to_xyz @ dng["neutral"]
        to_xyz = adapt(white / white[1], D50) @ cam_to_xyz
        to_xyz /= (to_xyz @ dng["neutral"])[1]
        exposure = 2.0 ** profile["colour"]["baseline_exposure"]
        self.source = cam @ (XYZ_TO_PROPHOTO @ to_xyz).T * exposure
        self.target = srgb_linear(self.thumb) @ SRGB_TO_PROPHOTO.T
        grey = cv2.cvtColor(self.thumb, cv2.COLOR_RGB2GRAY)
        edges = cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, np.ones((3, 3)))
        self.mask = ((edges < 0.06) & (self.thumb.max(-1) < 0.97)
                     & (self.thumb.min(-1) > 0.02) & (self.source.max(-1) < 1)
                     & (self.source.min(-1) > 0))

    def compare(self, render_dir):
        r = cv2.imread(str(render_dir / f"{self.name}.jpg"))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        r = cv2.resize(r, self.thumb.shape[1::-1], interpolation=cv2.INTER_AREA)
        ok = self.mask & (r.max(-1) < 0.97) & (r.min(-1) > 0.02)
        lab = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[ok]
        return (self.source[ok], self.target[ok],
                (srgb_linear(r) @ SRGB_TO_PROPHOTO.T)[ok],
                lab(r), lab(self.thumb))


def report(lab_render, lab_target, label):
    cr = np.hypot(*lab_render[:, 1:].T)
    ct = np.hypot(*lab_target[:, 1:].T)
    hue = np.degrees(np.arctan2(lab_target[:, 2], lab_target[:, 1])) % 360
    parts = []
    for lo, hi, name in ((0, 40, "red"), (40, 90, "yellow"), (90, 150, "green"),
                         (150, 210, "cyan"), (210, 280, "blue"),
                         (280, 360, "magenta")):
        k = (ct > 10) & (hue >= lo) & (hue < hi)
        if k.sum() > 200:
            parts.append(f"{name} {np.median(cr[k] / ct[k]):.2f}")
    de = np.linalg.norm(lab_render - lab_target, axis=1)
    print(f"{label:<14} colour error {de.mean():.2f} | colour kept vs "
          f"camera: {', '.join(parts)}")


def update_table(table, source, target, rendered, damp=0.8, min_weight=30):
    """Move each table cell toward the camera-JPEG colour it should give."""
    hi, si, _ = hsv(source)
    ht, st, vt = hsv(target)
    hr, sr, vr = hsv(rendered)
    k = (st > 0.04) & (sr > 0.04)
    x = hi[k] * HUES / 6
    y = np.clip(si[k], 0, 1) * (SATS - 1)
    # hue is preserved by Camera Raw's tone curve, so shifts carry over
    # exactly; saturation and value ratios carry over approximately
    corrections = (((ht[k] - hr[k] + 3) % 6 - 3) * 60,
                   np.log(st[k] / sr[k]),
                   np.log(vt[k] / vr[k]))
    corrections = corrections[:2] + (corrections[2] - np.median(corrections[2]),)
    acc = np.zeros((HUES, SATS, 4))
    x0 = np.floor(x).astype(int)
    fx = x - x0
    y0 = np.clip(np.floor(y).astype(int), 0, SATS - 2)
    fy = np.clip(y - y0, 0, 1)
    for ox, wx in ((0, 1 - fx), (1, fx)):
        for oy, wy in ((0, 1 - fy), (1, fy)):
            w = wx * wy
            cell = ((x0 + ox) % HUES, y0 + oy)
            for c, v in enumerate(corrections):
                np.add.at(acc[..., c], cell, w * v)
            np.add.at(acc[..., 3], cell, w)
    # smooth across neighbouring cells (hue wraps around)
    smooth = np.stack([gaussian_filter(np.concatenate([acc[..., c]] * 3),
                                       (1.2, 0.8), mode="nearest")[HUES:2 * HUES]
                       for c in range(4)], -1)
    count = smooth[..., 3]
    trust = count / (count + min_weight)  # sparse cells stay near identity
    mean = smooth[..., :3] / np.maximum(count, 1e-9)[..., None]
    t = table.reshape(HUES, SATS, 3).copy()
    t[..., 0] = np.clip(t[..., 0] + damp * trust * mean[..., 0], -25, 25)
    t[..., 1] = np.clip(t[..., 1] * np.exp(damp * trust * mean[..., 1]), 0.75, 1.9)
    t[..., 2] = np.clip(t[..., 2] * np.exp(damp * trust * mean[..., 2]), 0.85, 1.15)
    t[:, 0] = (0, 1, 1)  # greys stay grey
    return t.reshape(-1, 3)


def vivid_from(natural, strength=0.22):
    """Natural plus vibrance: muted colours lifted most, skin protected."""
    t = natural.reshape(HUES, SATS, 3).copy()
    hue = np.arange(HUES) * 360 / HUES
    sat = np.arange(SATS) / (SATS - 1)
    skin = np.exp(-0.5 * (((hue - 25 + 180) % 360 - 180) / 18) ** 2)
    boost = 1 + strength * (1 - sat[None]) ** 1.3 * (1 - 0.7 * skin[:, None])
    boost[:, 0] = 1
    t[..., 1] = np.minimum(t[..., 1] * boost, 2.2)
    return t.reshape(-1, 3)


def collect(dngs, count, seed):
    usable = []
    for path in dngs:
        dng = read_dng(path)
        if dng["thumb_is_jpeg"] and dng["lens"] is not None:
            usable.append(path)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(usable))
    return [usable[i] for i in sorted(pick[:count])], \
           [usable[i] for i in sorted(pick[count:count + count // 2])]


def fit(dngs, profile_path, iterations=2, count=30):
    profile = json.loads(profile_path.read_text())
    train, holdout = collect(dngs, count, seed=1)
    if len(train) < 10:
        raise SystemExit(f"Need at least 10 DNGs with camera JPEGs, "
                         f"found {len(train)}")
    work = Path(tempfile.mkdtemp(prefix="fz55_look_"))
    folders = {}
    for name, paths in (("train", train), ("holdout", holdout)):
        folders[name] = work / name
        folders[name].mkdir()
        for p in paths:
            shutil.copy(p, folders[name])
    shots = [Shot(p, profile) for p in sorted(folders["train"].glob("*.dng"))]
    print(f"Fitting on {len(shots)} shots, checking on {len(holdout)} held out")

    table = identity_table()
    for it in range(iterations + 1):
        trial = with_look(profile, None if it == 0 else table, "FZ55 Natural")
        for p in folders["train"].glob("*.dng"):
            refresh(p, trial)
        out = work / f"render{it}"
        render(folders["train"], out)
        parts = [s.compare(out) for s in shots]
        source, target, rendered, lr, lt = (np.concatenate(x) for x in zip(*parts))
        report(lr, lt, "before" if it == 0 else f"iteration {it}")
        if it < iterations:
            table = update_table(table, source, target, rendered)

    natural, vivid = table, vivid_from(table)
    held = [Shot(p, profile) for p in sorted(folders["holdout"].glob("*.dng"))]
    for label, t in (("held out, old", None), ("held out, new", natural)):
        trial = with_look(profile, t, "FZ55 Natural")
        for p in folders["holdout"].glob("*.dng"):
            refresh(p, trial)
        out = work / label.replace(" ", "_").replace(",", "")
        render(folders["holdout"], out)
        parts = [s.compare(out) for s in held]
        report(*(np.concatenate(x) for x in list(zip(*parts))[3:]), label)

    profile = with_look(profile, natural, "FZ55 Natural")
    profile["looks"] = {
        "FZ55 Natural": np.round(natural, 5).tolist(),
        "FZ55 Vivid": np.round(vivid, 5).tolist(),
    }
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    shutil.rmtree(work)
    print(f"Saved FZ55 Natural (embedded default) and FZ55 Vivid to "
          f"{profile_path}")


def write_dcp(path, profile, name, table):
    """A DNG camera profile file: TIFF layout with the 'IIRC' signature."""
    colour = profile["colour"]
    ifd = Ifd()
    ifd.add(50708, ASCII, UNIQUE_MODEL)
    ifd.add(50721, SRATIONAL, [v for row in colour["colour_matrix"] for v in row])
    ifd.add(50778, SHORT, colour["calibration_illuminant_code"])
    ifd.add(50936, ASCII, name)
    ifd.add(50941, LONG, 0)
    if table is not None:
        ifd.add(50981, LONG, [HUES, SATS, 1])
        ifd.add(50982, FLOAT, [v for cell in table for v in cell])
        ifd.add(51108, LONG, 0)
    path.write_bytes(b"IIRC" + struct.pack("<I", 8) + ifd.pack(8))


def install(profile_path):
    profile = json.loads(profile_path.read_text())
    looks = profile.get("looks")
    if not looks:
        raise SystemExit("No looks in the profile; run `fit` first")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    entries = list(looks.items()) + [("FZ55 Matrix", None)]
    for name, table in entries:
        out = PROFILE_DIR / f"{name.replace(' ', '_')}.dcp"
        write_dcp(out, profile, name, table)
        print(f"Installed {out}")
    print("Restart Lightroom; the profiles appear under Profile Browser > "
          "Profiles for these DNGs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit", "install"))
    parser.add_argument("dngs", nargs="*", type=Path,
                        help="converted DNGs to fit on (fit only)")
    parser.add_argument("--profile", type=Path,
                        default=Path(__file__).with_name("fz55_profile.json"))
    args = parser.parse_args()
    if args.command == "fit":
        if not args.dngs:
            parser.error("fit needs DNGs, e.g. output/*.dng")
        fit(args.dngs, args.profile)
    else:
        install(args.profile)
