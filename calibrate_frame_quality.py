"""
Calibrate the frame-quality gate against a real video BEFORE the demo.

Why this exists:
  BLUR_THRESHOLD / DARKNESS_THRESHOLD in utils/frame_quality.py are generic
  guesses. Different cameras / compression give very different baseline
  Laplacian-variance and brightness values even on footage that looks perfectly
  fine. If the thresholds are wrong for tomorrow's clip, the gate can flag
  *every* frame -> detection never runs at full confidence, and the demo looks
  broken.

What it does:
  Samples frames across the whole video and reports the real distribution of
  both metrics, plus the percentage of frames that would be flagged at the
  current thresholds. If that percentage is much higher than the video actually
  looks blurry/dark, lower the threshold(s) in utils/frame_quality.py, then
  re-run this until the flag rate looks sane.

Do this ONCE, tonight, on a clip similar to what you expect tomorrow.
Don't discover it live.

Usage:
    python calibrate_frame_quality.py /path/to/video.mp4 [--samples 300]
"""

import argparse
import statistics
import sys

import cv2

from utils.frame_quality import (
    BLUR_THRESHOLD,
    DARKNESS_THRESHOLD,
    laplacian_variance,
    mean_brightness,
)


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(name, values, threshold):
    """Print a distribution summary and the % of frames below `threshold`
    (both metrics flag a frame when the value is BELOW its threshold)."""
    vs = sorted(values)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  count    : {len(values)}")
    print(f"  min      : {min(values):.1f}")
    print(f"  p05      : {percentile(vs, 5):.1f}")
    print(f"  median   : {statistics.median(values):.1f}")
    print(f"  p95      : {percentile(vs, 95):.1f}")
    print(f"  max      : {max(values):.1f}")
    print(f"  mean     : {statistics.mean(values):.1f}")
    flagged = sum(1 for v in values if v < threshold)
    pct = 100.0 * flagged / len(values)
    print(f"  threshold: {threshold:.1f}  ->  {pct:.1f}% of sampled frames "
          f"flagged (below threshold)")
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--samples", type=int, default=300,
                    help="max number of frames to sample across the video")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.video}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // args.samples) if total else 1

    blur_vals, bright_vals = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            blur_vals.append(laplacian_variance(frame))
            bright_vals.append(mean_brightness(frame))
        idx += 1
    cap.release()

    if not blur_vals:
        print("ERROR: no frames read.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSampled {len(blur_vals)} frames from {args.video} "
          f"(total ~{total} frames).")

    blur_pct = summarize("Laplacian variance (sharpness)  [lower = blurrier]",
                         blur_vals, BLUR_THRESHOLD)
    dark_pct = summarize("Mean brightness (0-255)         [lower = darker]",
                         bright_vals, DARKNESS_THRESHOLD)

    print("\nRead this as:")
    print(f"  - {blur_pct:.1f}% of frames would be flagged BLURRY   at "
          f"BLUR_THRESHOLD={BLUR_THRESHOLD:.0f}")
    print(f"  - {dark_pct:.1f}% of frames would be flagged TOO DARK at "
          f"DARKNESS_THRESHOLD={DARKNESS_THRESHOLD:.0f}")
    print("\nIf either percentage is much higher than the footage actually "
          "looks bad,\nlower that threshold in utils/frame_quality.py and "
          "re-run this script.\n")


if __name__ == "__main__":
    main()
