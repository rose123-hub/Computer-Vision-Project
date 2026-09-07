"""
Person detection + tracking with TEMPORARY IDs (ByteTrack).

Detector : YOLOv8n, COCO-pretrained, filtered to the `person` class (id 0).
Tracker  : ByteTrack -> motion association only (Kalman filter + IoU).
           No appearance memory. When a person leaves and re-enters the frame
           they get a BRAND-NEW incrementing ID, because ByteTrack has no way
           to recognize "this is the same person as before."

Contrast this with detect_track_persistent_id.py, which is the SAME script
with one differing line (the tracker config) and therefore KEEPS the old ID
on re-entry. See that file's header for the exact line-by-line diff.

Also drawn on every frame:
  - a per-track hit-streak confirmation gate  (false-positive mitigation)
  - a running FPS readout
  - a live occupancy readout:  "In view: N"  and  "Peak: M"
    (how many confirmed people are on screen right now / the max seen so far
     this run). This is a live, in-memory-only count -- it resets when the
     script restarts and is deliberately NOT a unique cumulative footfall
     count (that would need line-crossing logic, which is out of scope).

Usage:
    python detect_track_temporary_id.py /path/to/video.mp4
    (press q in the window to quit)
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from utils.frame_quality import is_blurry, is_too_dark

# ---- Tunables -------------------------------------------------------------
MODEL_PATH   = "yolov8n.pt"   # COCO-pretrained nano; auto-downloads once
PERSON_CLASS = 0              # COCO class id for "person"
BASE_CONF    = 0.35           # detector confidence on normal frames.
                              #   Tuned above YOLO's ~0.25 default to cut
                              #   low-confidence spurious boxes  (FP fix #1)
LOWQ_CONF    = 0.55           # raised confidence on blurry/dark frames
MIN_HITS     = 3              # a track must be seen this many CONSECUTIVE
                              #   frames before it is confirmed & drawn.
                              #   Kills single-frame flicker (poster/reflection
                              #   misdetected for one frame)     (FP fix #2)
MIN_BOX_HEIGHT = 120         # px; ignore detections shorter than this.
                              #   Far-away street pedestrians produce short
                              #   boxes -- this keeps "In view" to people who
                              #   are actually near the camera/entrance.
                              #   RAISE to count only close people, LOWER to
                              #   include more distant ones. Tune per camera.

# >>> THE ONE LINE THAT DEFINES THIS SCRIPT'S BEHAVIOR <<<
TRACKER = "bytetrack.yaml"   # motion-only association, NO Re-ID
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    args = ap.parse_args()

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open {args.video}")

    hit_counts = {}     # track_id -> consecutive frames seen
    confirmed  = set()  # track_ids that have passed the hit-streak gate
    peak_count = 0      # max simultaneous confirmed people this run (add-on)

    frame_idx = 0
    t_prev = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ---- BINARY CLASSIFICATION: frame-quality gate (BEFORE detection) --
        blurry = is_blurry(frame)
        dark   = is_too_dark(frame)
        conf   = LOWQ_CONF if (blurry or dark) else BASE_CONF
        if blurry or dark:
            flags = ", ".join(f for f, on in (("blurry", blurry), ("dark", dark)) if on)
            print(f"[frame {frame_idx}] low-quality ({flags}) "
                  f"-> conf raised to {conf}")

        # ---- OBJECT DETECTION + TRACKING (the core call) ------------------
        # persist=True keeps tracker state across these manual per-frame calls.
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,            # <-- ByteTrack (see TRACKER above)
            classes=[PERSON_CLASS],
            conf=conf,
            verbose=False,
        )

        boxes = results[0].boxes
        seen_this_frame = set()

        if boxes is not None and boxes.id is not None:
            ids  = boxes.id.int().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                # ---- skip far-away / tiny detections (street pedestrians) --
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue
                seen_this_frame.add(tid)

                # ---- FALSE-POSITIVE MITIGATION: consecutive-hit streak -----
                hit_counts[tid] = hit_counts.get(tid, 0) + 1
                if hit_counts[tid] >= MIN_HITS:
                    confirmed.add(tid)

                # draw only once a track is confirmed as a real person
                if tid in confirmed:
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # reset the streak for any track NOT seen this frame (must be CONSECUTIVE)
        for tid in list(hit_counts):
            if tid not in seen_this_frame:
                hit_counts[tid] = 0

        # ---- OCCUPANCY / PEAK (add-on) ------------------------------------
        current_count = len(seen_this_frame & confirmed)   # boxes drawn now
        peak_count = max(peak_count, current_count)         # only ever rises

        # ---- FPS (exponential moving average) -----------------------------
        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps else (1.0 / dt)

        # ---- Overlays -----------------------------------------------------
        cv2.putText(frame, f"FPS: {fps:4.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"In view: {current_count}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Peak: {peak_count}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Footfall - TEMPORARY IDs (ByteTrack)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Peak simultaneous people in view this run: {peak_count}")


if __name__ == "__main__":
    main()
