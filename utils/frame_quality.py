import cv2
BLUR_THRESHOLD = 40.0
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
