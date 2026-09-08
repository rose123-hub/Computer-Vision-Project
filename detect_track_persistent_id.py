import argparse
import time

import cv2
from ultralytics import YOLO

from utils.frame_quality import is_blurry, is_too_dark

# ---- Tunables -------------------------------------------------------------
MODEL_PATH   = "yolov8n.pt"  
PERSON_CLASS = 0             
BASE_CONF    = 0.35           
LOWQ_CONF    = 0.55          
MIN_HITS     = 3              
MIN_BOX_HEIGHT = 120         
TRACKER = "botsort.yaml"     
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
            tracker=TRACKER,            # <-- BoT-SORT + Re-ID (see TRACKER above)
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

        cv2.imshow("Footfall - PERSISTENT IDs (BoT-SORT + Re-ID)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Peak simultaneous people in view this run: {peak_count}")


if __name__ == "__main__":
    main()
