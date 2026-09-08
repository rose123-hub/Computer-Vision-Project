import argparse
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS = 0
CONF = 0.35
MIN_HITS = 3
MIN_BOX_HEIGHT = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    args = ap.parse_args()

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open {args.video}")

    hit_counts = {}
    confirmed = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            conf=CONF,
            verbose=False
        )

        boxes = results[0].boxes
        seen_this_frame = set()

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue

                seen_this_frame.add(tid)

                hit_counts[tid] = hit_counts.get(tid, 0) + 1

                if hit_counts[tid] >= MIN_HITS:
                    confirmed.add(tid)

                if tid in confirmed:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID {tid}",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        for tid in list(hit_counts):
            if tid not in seen_this_frame:
                hit_counts[tid] = 0

        people_tracked = len(seen_this_frame & confirmed)

        cv2.putText(
            frame,
            f"People Tracked: {people_tracked}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Footfall - YOLOv8 + ByteTrack", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
