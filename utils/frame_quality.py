"""
Frame-quality gate for the footfall demo.

This is the BINARY CLASSIFICATION step of the pipeline (see docs/TASK_TYPES.md):
two independent two-class decisions made on every frame *before* the detector
runs --

    is_blurry(frame)   -> True / False   (blurry  vs. sharp,   Laplacian variance)
    is_too_dark(frame) -> True / False   (dark    vs. well-lit, mean brightness)

A frame flagged by either check is NOT thrown away. The tracking scripts raise
the detector's confidence threshold for that frame (and log it) so a soft or
dark frame produces fewer low-confidence spurious boxes. This is the concrete
"how does the code handle low-quality images" answer -- see
docs/INFERENCE_AND_ACCURACY.md.

The two thresholds below are GENERIC STARTING GUESSES, not numbers tuned
against real footage. Run `calibrate_frame_quality.py` against your actual
video first and adjust them if the flag rate is far higher than the footage
actually looks bad -- a compressed / low-res / CCTV-style clip can read as
"blurry" everywhere under a threshold that was fine for a phone recording,
which would flag every frame and stop detection from ever running at full
confidence.
"""

import cv2

# Laplacian-variance below this  => frame is treated as blurry.
# Higher variance = more high-frequency detail = sharper image.
BLUR_THRESHOLD = 40.0

# Mean pixel brightness (0-255) below this  => frame is treated as too dark.
DARKNESS_THRESHOLD = 50.0


def laplacian_variance(frame):
    """Sharpness score: variance of the Laplacian of the grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_brightness(frame):
    """Average brightness (0-255) of the grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_blurry(frame, threshold=BLUR_THRESHOLD):
    """Binary decision: True if the frame is blurrier than `threshold`."""
    return laplacian_variance(frame) < threshold


def is_too_dark(frame, threshold=DARKNESS_THRESHOLD):
    """Binary decision: True if the frame is darker than `threshold`."""
    return mean_brightness(frame) < threshold
