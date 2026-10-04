#!/usr/bin/env python3
"""Derive an FZ55 DNG profile from a diagnostic RAW and its camera JPEG.

The camera's JPEG engine corrects lens distortion, so matching the JPEG
against the uncorrected sensor mosaic recovers the distortion model at the
shot's focal length. The same aligned pair also yields a colour matrix and
the camera's tone curve.

Each run adds or replaces one focal-length entry in the profile JSON. Run it
on pairs shot at several zoom positions to cover the zoom range; the
converter interpolates between calibrated focal lengths.

Requires NumPy, SciPy, Pillow and OpenCV (opencv-python-headless).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from fz55_common import (
    HEIGHT, WIDTH, WHITE_LEVEL, aligned_planes, find_companion_jpeg,
    model_points, read_jpeg_exif, read_raw, warp_norm,
)

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])
WHITE_XYZ = {  # Y = 1
    "D65": np.array([0.95047, 1.0, 1.08883]),
    "D55": np.array([0.95682, 1.0, 0.92149]),
    "A": np.array([1.09850, 1.0, 0.35585]),
}
# EXIF LightSource codes used by DNG CalibrationIlluminant
ILLUMINANT_CODE = {"D65": 21, "D55": 20, "A": 17, "Flash": 4}
# Camera Raw's default rendering puts a given scene level ~0.93 EV brighter
# than the camera JPEG's curve does; measured by rendering 10 calibration
# DNGs in Camera Raw (Photoshop 2026) and matching median L* to the JPEGs.
ACR_EXPOSURE_OFFSET = -0.93


def srgb_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_encode(v):
    v = np.clip(v, 0, None)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def log_image(x, ref_percentile=50):
    x = np.log1p(x / max(np.percentile(x, ref_percentile), 1e-6) * 20)
    return cv2.GaussianBlur(x, (0, 0), 1.5).astype(np.float32)


def fit_model(pj, pr, p0):
    sol = least_squares(lambda p: (model_points(p, pj) - pr).ravel(), p0,
                        loss="soft_l1", f_scale=1.0)
    err = np.linalg.norm(model_points(sol.x, pj) - pr, axis=1)
    return sol.x, err


def sift_matches(raw_lum, jpeg_lum):
    """Coarse correspondences at half resolution, returned in full-res px."""
    clahe = cv2.createCLAHE(3.0, (8, 8))

    def prep(x):
        x = x[::2, ::2]
        x = (x / np.percentile(x, 99.5)) ** (1 / 2.2)
        return clahe.apply(np.uint8(np.clip(x, 0, 1) * 255))

    a, b = prep(raw_lum), prep(jpeg_lum)
    sift = cv2.SIFT_create(20000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [x for x, y in pairs if x.distance < 0.75 * y.distance]
    if len(good) < 50:
        raise ValueError(f"Only {len(good)} feature matches; scene too dark "
                         "or RAW/JPEG are not the same shot")
    pj = np.float64([kb[x.queryIdx].pt for x in good]) * 2
    pr = np.float64([ka[x.trainIdx].pt for x in good]) * 2
    _, inl = cv2.findHomography(pj, pr, cv2.RANSAC, 16.0)
    inl = inl.ravel().astype(bool)
    return pj[inl], pr[inl]


def dense_matches(raw_lum, jpeg_lum, params, step=48, half=72, search=24):
    """Refine a full-frame grid by patch correlation around the coarse model."""
    a, b = log_image(raw_lum), log_image(jpeg_lum)
    pts = []
    lo = half + search + 4
    for y in range(lo, HEIGHT - lo, step):
        for x in range(lo, WIDTH - lo, step):
            tpl = b[y - half:y + half, x - half:x + half]
            if tpl.std() < 0.04:
                continue
            (rx, ry), = model_points(params, np.array([[x, y]], float))
            rx, ry = int(round(rx)), int(round(ry))
            if not (lo <= rx < WIDTH - lo and lo <= ry < HEIGHT - lo):
                continue
            win = a[ry - half - search:ry + half + search,
                    rx - half - search:rx + half + search]
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (lx, ly) = cv2.minMaxLoc(res)
            if peak < 0.6 or not (0 < lx < res.shape[1] - 1
                                  and 0 < ly < res.shape[0] - 1):
                continue

            def sub(l, c, r):
                d = l - 2 * c + r
                return 0.0 if d == 0 else 0.5 * (l - r) / d

            ox = sub(res[ly, lx - 1], res[ly, lx], res[ly, lx + 1])
            oy = sub(res[ly - 1, lx], res[ly, lx], res[ly + 1, lx])
            pts.append((x, y, rx - search + lx + ox, ry - search + ly + oy))
    pts = np.array(pts)
    return pts[:, :2], pts[:, 2:]


def fit_colour(cam, jpeg, knots=14, samples=150000):
    """Fit JPEG ~= tone(M @ cam); M[1,1] fixed to 1 to pin the scale."""
    ok = ((jpeg.max(-1) < 0.97) & (jpeg.min(-1) > 0.02)
          & (cam.max(-1) < 0.95) & (cam.min(-1) > 2 / WHITE_LEVEL))
    idx = np.flatnonzero(ok)
    if idx.size < 5000:
        raise ValueError("Too few well-exposed pixels for a colour fit")
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, min(samples, idx.size), replace=False)
    c = cam.reshape(-1, 3)[idx]
    j = jpeg.reshape(-1, 3)[idx]
    m0, *_ = np.linalg.lstsq(c, srgb_linear(j), rcond=None)
    m0 = m0.T / m0.T[1, 1]
    lx = np.log2(np.clip(c @ m0.T, 1e-5, None))
    kn = np.linspace(np.percentile(lx, 0.5) - 0.5,
                     np.percentile(lx, 99.9) + 0.3, knots)

    def tone(x, tp):
        vals = np.concatenate([[tp[0]], tp[0] + np.cumsum(np.exp(tp[1:]))])
        return np.interp(np.log2(np.clip(x, 1e-6, None)), kn, vals)

    def unpack(p):
        m = p[:9].reshape(3, 3).copy()
        m[1, 1] = 1.0
        return m, p[9:]

    def resid(p):
        m, tp = unpack(p)
        return (tone(c @ m.T, tp) - j).ravel()

    v0 = srgb_encode(2.0 ** kn)
    tp0 = np.concatenate([[v0[0]], np.log(np.maximum(np.diff(v0), 1e-4))])
    sol = least_squares(resid, np.concatenate([m0.ravel(), tp0]),
                        loss="soft_l1", f_scale=0.03, max_nfev=400)
    m, tp = unpack(sol.x)
    vals = np.concatenate([[tp[0]], tp[0] + np.cumsum(np.exp(tp[1:]))])
    err = np.abs(resid(sol.x)) * 255
    return m, kn, vals, err


def colour_matrix(m, white):
    """XYZ (under `white`) -> camera, normalised so white maps to max 1."""
    w = WHITE_XYZ[white]
    lms_src, lms_dst = BRADFORD @ w, BRADFORD @ WHITE_XYZ["D65"]
    adapt = np.linalg.inv(BRADFORD) @ np.diag(lms_dst / lms_src) @ BRADFORD
    cm = np.linalg.inv(m) @ np.linalg.inv(SRGB_TO_XYZ) @ adapt
    neutral = cm @ w
    cm /= neutral.max()
    return cm, neutral / neutral.max()


def calibrate(raw_path, jpeg_path, profile_path, illuminant):
    mosaic = read_raw(raw_path).astype(np.float32)
    exif = read_jpeg_exif(jpeg_path)
    if exif.get("focal_length") is None:
        raise ValueError("JPEG has no focal length in EXIF")
    jpeg_bgr = cv2.imread(str(jpeg_path))
    if jpeg_bgr.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError("JPEG must be full resolution (4608 x 3456)")
    jpeg = cv2.cvtColor(jpeg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    raw_lum = cv2.filter2D(mosaic, -1, np.full((2, 2), 0.25, np.float32),
                           anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
    jpeg_lum = srgb_linear(jpeg).mean(-1).astype(np.float32)

    pj, pr = sift_matches(raw_lum, jpeg_lum)
    coarse, _ = fit_model(pj, pr, [WIDTH / 2, HEIGHT / 2, 1, 0, 0, 0])
    # A wide first search tolerates the coarse model's corner error; narrower
    # passes then stop the sparse, noisy corners from biasing the fit.
    params = coarse
    for search, half in ((24, 72), (10, 72), (6, 48)):
        pj, pr = dense_matches(raw_lum, jpeg_lum, params, half=half,
                               search=search)
        params, err = fit_model(pj, pr, params)
    cx, cy = params[:2]
    radius = np.hypot(pj[:, 0] - cx, pj[:, 1] - cy) / warp_norm(cx, cy)
    print(f"Distortion: {len(pj)} correspondences, median residual "
          f"{np.median(err):.2f} px, coverage to r={radius.max():.2f}")
    if radius.max() < 0.8:
        print("Warning: corners poorly constrained; shoot a brighter, "
              "more textured scene for this focal length")

    cam = aligned_planes(mosaic, params) / WHITE_LEVEL
    jpeg_coarse = cv2.resize(jpeg, (cam.shape[1], cam.shape[0]),
                             interpolation=cv2.INTER_AREA)
    m, kn, vals, cerr = fit_colour(cam, jpeg_coarse)
    print(f"Colour: median error {np.median(cerr):.2f}/255, "
          f"90th percentile {np.percentile(cerr, 90):.2f}/255")
    white = "D55" if illuminant == "Flash" else illuminant
    cm, neutral = colour_matrix(m, white)

    # Mid-grey (sRGB 0.46) level after DNG white balance, relative to white
    mid = 2.0 ** np.interp(0.46, vals, kn)
    u = np.linalg.inv(m) @ np.ones(3)
    baseline = float(np.log2(0.18 / (mid * u.max()))) + ACR_EXPOSURE_OFFSET

    profile = (json.loads(profile_path.read_text()) if profile_path.exists()
               else {"camera": "KODAK PIXPRO FZ55", "lens": []})
    profile.update(
        cfa_pattern="RGGB", black_level=0, white_level=WHITE_LEVEL,
        colour=dict(
            calibration_illuminant=illuminant,
            calibration_illuminant_code=ILLUMINANT_CODE[illuminant],
            colour_matrix=np.round(cm, 6).tolist(),
            default_neutral=np.round(neutral, 6).tolist(),
            baseline_exposure=round(baseline, 3),
            source=f"{raw_path.name} + {jpeg_path.name}",
            jpeg_median_error_8bit=round(float(np.median(cerr)), 2),
            # camera rendering, used to estimate each shot's white balance
            jpeg_matrix=np.round(m, 6).tolist(),
            tone_log2_knots=np.round(kn, 4).tolist(),
            tone_values=np.round(vals, 5).tolist(),
        ),
    )
    entry = dict(
        focal_length=exif["focal_length"],
        center=[round(cx / (WIDTH - 1), 6), round(cy / (HEIGHT - 1), 6)],
        kr=[round(float(params[2] * k), 7)
            for k in (1, params[3], params[4], params[5])],
        median_residual_px=round(float(np.median(err)), 3),
        max_radius_fitted=round(float(radius.max()), 3),
        correspondences=int(len(pj)),
        source=f"{raw_path.name} + {jpeg_path.name}",
        calibrated=date.today().isoformat(),
    )
    profile["lens"] = sorted(
        [e for e in profile["lens"]
         if abs(e["focal_length"] - entry["focal_length"]) > 0.05] + [entry],
        key=lambda e: e["focal_length"])
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"Saved {entry['focal_length']} mm lens entry to {profile_path}")


def calibrate_colour_from_dngs(paths, profile_path, max_samples=3000):
    """Refit the colour profile jointly over many converted DNGs.

    Each DNG's thumbnail is its camera JPEG, so every shot contributes
    (sensor RGB, camera output) samples under its own light. One shared
    matrix and tone curve are fitted, with a free white balance per shot;
    the median white balance, which for mostly outdoor shots is daylight,
    becomes the D65 calibration point.
    """
    from scipy.sparse import lil_matrix

    from fz55_to_dng import _camera_on_jpeg_grid, read_dng

    shots = []
    for path in paths:
        dng = read_dng(path)
        if dng["thumb_is_jpeg"] is False:
            continue
        thumb = dng["thumb"]
        cam = _camera_on_jpeg_grid(dng["mosaic"], thumb, dng["lens"])
        if cam is None:
            continue
        cam = cam / WHITE_LEVEL
        grey = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        edges = cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, np.ones((3, 3)))
        # flat, unclipped areas only: thumbnail-scale misregistration at
        # edges would otherwise mix colours
        ok = ((thumb.max(-1) < 0.96) & (thumb.min(-1) > 0.03)
              & (cam.max(-1) < 0.9) & (cam.min(-1) > 3 / WHITE_LEVEL)
              & (edges < 0.08))
        idx = np.flatnonzero(ok)
        if idx.size < 500:
            continue
        rng = np.random.default_rng(len(shots))
        idx = rng.choice(idx, min(max_samples, idx.size), replace=False)
        shots.append((Path(path).name, cam.reshape(-1, 3)[idx],
                      thumb.reshape(-1, 3)[idx]))
    if len(shots) < 5:
        raise ValueError(f"Only {len(shots)} usable DNGs with camera JPEG "
                         "thumbnails; need at least 5 varied shots")
    n = len(shots)
    c = np.concatenate([s[1] for s in shots])
    j = np.concatenate([s[2] for s in shots])
    sid = np.concatenate([[i] * len(s[1]) for i, s in enumerate(shots)])

    m0, *_ = np.linalg.lstsq(c, srgb_linear(j), rcond=None)
    m0 = m0.T / m0.T[1, 1]
    knots = 14
    lx = np.log2(np.clip(c @ m0.T, 1e-5, None))
    kn = np.linspace(np.percentile(lx, 0.5) - 1.5,
                     np.percentile(lx, 99.9) + 1.5, knots)

    def unpack(p):
        m = p[:9].reshape(3, 3).copy()
        m[1, 1] = 1.0  # pins the matrix/tone-curve scale
        tp = p[9:9 + knots]
        # per-shot log gains, constrained to zero mean so they cannot trade
        # off against the matrix columns
        lw = p[9 + knots:].reshape(n - 1, 3)
        return m, tp, np.vstack([lw, -lw.sum(0)])

    def tone_curve(x, tp):
        vals = np.concatenate([[tp[0]], tp[0] + np.cumsum(np.exp(tp[1:]))])
        return np.interp(np.log2(np.clip(x, 1e-6, None)), kn, vals)

    def resid(p):
        m, tp, lw = unpack(p)
        return (tone_curve((c * np.exp(lw[sid])) @ m.T, tp) - j).ravel()

    head = 9 + knots
    sparsity = lil_matrix((c.shape[0] * 3, head + 3 * (n - 1)), dtype=int)
    sparsity[:, :head] = 1
    for i in range(n):
        rows = np.flatnonzero(sid == i)
        cols = (slice(head, None) if i == n - 1
                else slice(head + 3 * i, head + 3 * i + 3))
        for ch in range(3):
            sparsity[rows * 3 + ch, cols] = 1
    v0 = srgb_encode(2.0 ** kn)
    tp0 = np.concatenate([[v0[0]], np.log(np.maximum(np.diff(v0), 1e-4))])
    sol = least_squares(resid, np.concatenate([m0.ravel(), tp0,
                                               np.zeros(3 * (n - 1))]),
                        jac_sparsity=sparsity, loss="soft_l1", f_scale=0.03,
                        max_nfev=200, x_scale="jac")
    m, tp, lw = unpack(sol.x)
    vals = np.concatenate([[tp[0]], tp[0] + np.cumsum(np.exp(tp[1:]))])
    err = np.abs(resid(sol.x)) * 255
    print(f"Colour: {n} shots, median error {np.median(err):.2f}/255, "
          f"90th percentile {np.percentile(err, 90):.2f}/255")

    u = np.linalg.inv(m) @ np.ones(3)
    neutrals = np.exp(-lw) * u
    neutrals /= neutrals.max(1, keepdims=True)
    ref = np.exp(np.median(np.log(neutrals), 0))
    ref /= ref.max()
    cm = np.diag(ref / u) @ np.linalg.inv(m) @ np.linalg.inv(SRGB_TO_XYZ)
    cm /= (cm @ WHITE_XYZ["D65"]).max()

    # Mid-grey (sRGB 0.46) level after DNG white balance, relative to white
    mid = 2.0 ** np.interp(0.46, vals, kn)
    levels = mid * (np.exp(-lw) * u).max(1)
    baseline = float(np.log2(0.18 / np.median(levels))) + ACR_EXPOSURE_OFFSET

    profile = json.loads(profile_path.read_text())
    profile["colour"] = dict(
        calibration_illuminant="D65",
        calibration_illuminant_code=ILLUMINANT_CODE["D65"],
        colour_matrix=np.round(cm, 6).tolist(),
        default_neutral=np.round(ref, 6).tolist(),
        baseline_exposure=round(baseline, 3),
        source=f"{n} DNGs ({shots[0][0]} .. {shots[-1][0]})",
        calibrated=date.today().isoformat(),
        jpeg_median_error_8bit=round(float(np.median(err)), 2),
        jpeg_matrix=np.round(m, 6).tolist(),
        tone_log2_knots=np.round(kn, 4).tolist(),
        tone_values=np.round(vals, 5).tolist(),
    )
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"Saved colour profile to {profile_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+",
                        help="RAW to calibrate, or DNGs with --colour-from-dng")
    parser.add_argument("--jpeg", type=Path,
                        help="matching camera JPEG (default: next file number)")
    parser.add_argument("--profile", type=Path,
                        default=Path(__file__).with_name("fz55_profile.json"))
    parser.add_argument("--illuminant", choices=list(ILLUMINANT_CODE),
                        help="scene light (default: Flash if it fired, else D55)")
    parser.add_argument("--colour-from-dng", action="store_true",
                        help="refit only the colour profile from many "
                             "converted DNGs (recommended: 20+ varied shots)")
    args = parser.parse_args()
    try:
        if args.colour_from_dng:
            calibrate_colour_from_dngs(args.raw, args.profile)
            raise SystemExit(0)
        if len(args.raw) > 1:
            parser.error("lens calibration takes a single RAW")
        raw = args.raw[0]
        jpeg = args.jpeg or find_companion_jpeg(raw)
        light = args.illuminant or (
            "Flash" if read_jpeg_exif(jpeg).get("flash_fired") else "D55")
        print(f"Calibrating {raw.name} against {jpeg.name} ({light})")
        calibrate(raw, jpeg, args.profile, light)
    except ValueError as error:
        parser.exit(1, f"Cannot calibrate: {error}\n")
