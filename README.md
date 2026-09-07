# Footfall Analysis Demo — Person Detection & Tracking

A CPU-only, real-time person detection and multi-object tracking pipeline built with **YOLOv8n** and **Ultralytics' built-in trackers (ByteTrack / BoT-SORT)**. Built as a technical demo for a QSR (quick-service restaurant) footfall analysis use case.

Given any input video, the pipeline detects people frame-by-frame and tracks them across frames, drawing bounding boxes and IDs live in an on-screen window. Two scripts demonstrate the core trade-off in tracking design: **temporary IDs** (no memory of lost tracks) vs. **persistent IDs** (appearance-based re-identification).

---

## What's in this repo

| Script | Tracker | Behavior |
|---|---|---|
| `detect_track_temporary_id.py` | ByteTrack (motion-only) | A person who leaves and re-enters frame gets a **new** ID |
| `detect_track_persistent_id.py` | BoT-SORT + Re-ID | A person who leaves and re-enters frame (within a short window) **keeps** their ID |

Both scripts use the same YOLOv8n detector, the same `model.track()` call, and the same frame-quality gate — only the tracker configuration differs. This is intentional: it makes the ID-persistence behavior a one-line, directly comparable difference.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with temporary IDs (ByteTrack)
python detect_track_temporary_id.py --source path/to/video.mp4

# 4. Run with persistent IDs (BoT-SORT + Re-ID)
python detect_track_persistent_id.py --source path/to/video.mp4
```

Press `q` to quit the live window. Running FPS is printed to the console.

### Requirements
- Python 3.9+
- CPU only — no GPU/CUDA required (this is a design constraint, not a fallback)
- See `requirements.txt`: `ultralytics`, `opencv-python`

---

## How it works

1. **Frame-quality gate** (`utils/frame_quality.py`) — every frame is checked for blur (Laplacian variance) and darkness (mean brightness) *before* detection. Flagged frames are skipped or run at a raised confidence threshold, and this is logged to the console.
2. **Detection** — YOLOv8n (`ultralytics`, pretrained on COCO), filtered to the `person` class (`classes=[0]`).
3. **Tracking** — Ultralytics' built-in `model.track()`, using either:
   - `bytetrack.yaml` — pure motion association (Kalman filter + IoU)
   - `botsort.yaml` with `with_reid: True` — motion association + an appearance embedding matched against a short-term gallery of recently-lost tracks
4. **False-positive suppression** — a track is only confirmed and drawn after **N consecutive detections** (hit-streak), which filters out single-frame flicker (e.g. a poster or reflection briefly misdetected as a person).
5. **Output** — live `cv2.imshow` window with boxes + track IDs, plus running FPS printed to console.

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── utils/
│   └── frame_quality.py          # binary classification: is_blurry(), is_too_dark()
├── detect_track_temporary_id.py  # YOLOv8n + ByteTrack
├── detect_track_persistent_id.py # YOLOv8n + BoT-SORT (Re-ID)
└── docs/
    ├── TASK_TYPES.md              # detection / classification / feature extraction, mapped to code
    ├── DATA_STRATEGY.md           # synthetic train / real val / real test — documentation only
    ├── INFERENCE_AND_ACCURACY.md  # frame handling, false positives, benchmark accuracy
    └── DATABASE_DESIGN.md         # why no DB is used now, and what a vector DB extension would look like
```

---

## Design decisions at a glance

- **Model:** YOLOv8n — the only COCO-pretrained detector family that's reliably real-time on CPU, with tracking built in.
- **No fine-tuning:** the demo uses pretrained COCO weights as-is; the `person` class is large/diverse enough to transfer to a QSR entrance camera without retraining. Training/data strategy is documented ( `docs/DATA_STRATEGY.md`) but not executed here.
- **No frame skipping:** every frame is processed by default; skipping is only added if benchmarking shows it's needed for smooth FPS on a given machine.
- **No database:** the Re-ID gallery is in-memory and session-scoped. A vector-database design (FAISS/Chroma/Milvus) for true cross-session identity persistence is documented as a future extension, not implemented.
- **Scope:** detection + tracking only. No in/out line-crossing footfall counter in this version.

Full rationale for each of these — including rejected alternatives, accuracy benchmarks, and false-positive handling — is in [`docs/`](./docs).

## Docs index

- [`docs/TASK_TYPES.md`](./docs/TASK_TYPES.md) — object detection, binary classification, feature extraction, mapped to exact code locations
- [`docs/DATA_STRATEGY.md`](./docs/DATA_STRATEGY.md) — synthetic vs. real data strategy, annotation tooling
- [`docs/INFERENCE_AND_ACCURACY.md`](./docs/INFERENCE_AND_ACCURACY.md) — frame handling, false-positive mitigation, benchmark accuracy figures
- [`docs/DATABASE_DESIGN.md`](./docs/DATABASE_DESIGN.md) — current in-memory approach and the vector-DB extension for persistent identity
