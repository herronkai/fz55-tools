# fz55-tools

RAW capture and a Lightroom-ready DNG workflow for the **Kodak PIXPRO FZ55**,
a compact camera that officially shoots JPEG only.

The FZ55's firmware (v1.06) contains a factory diagnostic that saves the
unprocessed sensor data alongside each JPEG. A one-line startup script on the
SD card switches it on. **No firmware is modified or flashed**; the setting
lives in RAM and is cleared when the camera powers off.

These tools turn those diagnostic dumps into DNG files that open in Lightroom
and Camera Raw with:

- **Lens distortion correction**, embedded as a DNG `WarpRectilinear` opcode.
  It reproduces the camera's own correction to within about half a pixel.
- **Per-shot white balance**, read from the camera's own JPEG.
- **Camera profiles** (`FZ55 Natural`, `FZ55 Vivid`, `FZ55 Matrix`): hue-by-hue
  colour tables like Adobe's own profiles, so saturation behaves naturally.
- **Shooting metadata** (exposure, aperture, ISO, focal length, flash, date),
  copied from the JPEG, because the RAW dump has none.

The sensor data is stored untouched: 4608 × 3456, 12-bit, RGGB.

> Unofficial project, not affiliated with or endorsed by Kodak or JK Imaging.
> It uses an undocumented diagnostic feature; use it at your own risk.

## Enabling RAW capture

Tested only on firmware **v1.06**.

1. Copy [`sd-card/script.txt`](sd-card/script.txt) to the **root** of an SD
   card. It contains a single line: `CMD ss.raw 1 1`.
2. Put the card in the camera and turn it on. Shoot as normal.
3. Each photo now produces `IMGnnnn.RAW` in the card root, plus the usual JPEG
   in `DCIM/…`, numbered `nnnn+1`.

Things to know:

- Each RAW is 31.85 MB. Saving takes noticeably longer than a JPEG alone.
- If the card fills up, the RAW gets truncated and the JPEG can come out empty.
- To turn RAW off, delete `script.txt` and restart the camera.
- The camera never records orientation, so portrait shots need rotating.

## Converting to DNG

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python tools/fz55_to_dng.py /Volumes/<card>/IMG*.RAW -o output/
```

Keep each RAW and its JPEG together (on the card, or copied with the same
layout). The converter pairs them by file number and also searches `DCIM/*`
next to the RAW. RAWs already converted are skipped; `--overwrite` redoes
them.

The converter prints a `WARNING` when it falls back to a default:

- no companion JPEG: daylight white balance and no metadata
- a focal length that has no lens calibration: no distortion correction

Other modes:

| Command | Purpose |
|---|---|
| `fz55_to_dng.py --refresh output/*.dng` | Rebuild existing DNGs with the current profile. The originals aren't needed; the DNG keeps the camera JPEG as its thumbnail. |
| `fz55_to_dng.py --no-lens-correction …` | Skip the distortion opcode. |
| `fz55_to_dng.py --focal-length 5.1 …` | Override the focal length. |

## Lightroom profiles

```sh
.venv/bin/python tools/fz55_look.py install
```

This writes three `.dcp` files to
`~/Library/Application Support/Adobe/CameraRaw/CameraProfiles`. Restart
Lightroom, then open the Profile Browser:

| Profile | Look |
|---|---|
| **FZ55 Natural** | Matches the camera's colour. This is the default embedded in every DNG. |
| **FZ55 Vivid** | Natural plus vibrance: muted colours lifted most, skin protected. |
| **FZ55 Matrix** | Colour matrix only; a flat starting point for grading. |

To add more colour, use **Vibrance** rather than Saturation.

## Calibration

The shipped `tools/fz55_profile.json` was calibrated on one FZ55. You can
refine it from your own shots:

```sh
# Lens distortion and a first colour fit, from one RAW+JPEG pair
.venv/bin/python tools/fz55_calibrate.py IMGnnnn.RAW

# Colour matrix fitted jointly over many converted DNGs (20+ varied scenes)
.venv/bin/python tools/fz55_calibrate.py --colour-from-dng output/*.dng

# Look tables, fitted with Camera Raw in the loop (macOS + Photoshop 2026)
.venv/bin/python tools/fz55_look.py fit output/*.dng
```

**Lens distortion is calibrated at 5.1 mm (full wide) only.** Distortion
changes with zoom. To cover other focal lengths, shoot a bright, textured
scene at each zoom step and run `fz55_calibrate.py` on each pair. The
converter interpolates between calibrated focal lengths.

## How it works

- **RAW format:** big-endian 16-bit words holding 12-bit values, RGGB, with
  black already subtracted. White level is 4095.
- **Lens correction:** the camera's JPEG is already distortion-corrected. The
  calibration matches thousands of points between the mosaic and the JPEG,
  then fits the DNG radial model (scale ≈ 1.054, k1 ≈ −0.165).
- **White balance:** the sensor RGB of areas the camera rendered neutral grey.
  By the DNG maths, Lightroom then renders those areas grey too.
- **Colour:** a matrix and tone curve fitted jointly across many shots, with a
  free white balance per shot. On top of that sits a 36 × 8 ProPhoto-HSV
  look table, fitted by rendering with Camera Raw, measuring each hue against
  the camera JPEG, and iterating.
- **Exposure:** `BaselineExposure` includes a −0.93 EV offset, measured so
  Camera Raw's default rendering matches the camera's brightness.

## Other tools

- `tools/inspect_fz55.py`: offline parser for Kodak's official firmware update
  file. It never talks to a camera, and no firmware is included in this repo.
- `tools/decode_fz55_raw.py`: dumps a RAW to a 16-bit grayscale TIFF mosaic.

## License

MIT. See [LICENSE](LICENSE).
