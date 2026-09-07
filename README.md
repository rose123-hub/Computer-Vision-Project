# Computer-Vision-Project
# Footfall Analysis Demo — Person Detection & Tracking (Proglint)

## Context
This is a take-home/demo deliverable for a computer-vision problem statement from Proglint (QSR footfall analysis). Tomorrow the user will be handed a random video and must run person detection + tracking live in front of the company, and must be able to answer deep technical questions about the ML pipeline (task types used, data strategy, database choices, false-positive handling, accuracy, frame handling) — not just show working code.

Decisions confirmed with the user so far:
- **CPU only** (no CUDA GPU) — model choice must be real-time on CPU.
- **Video file input only** (matches tomorrow's "random video" test — no webcam/RTSP needed).
- **Core scope = detection + tracking only** — no in/out line-crossing footfall counter.
- **Live output = on-screen window** (cv2.imshow with boxes + IDs drawn).
- **Training/data strategy = documentation only**, no fine-tuning code — the live demo uses the pretrained model as-is; synthetic-data/training is explained as a design the user can articulate, not executed before tomorrow.
- **Frame sampling = process every frame** by default; only add frame-skipping if benchmarking on the user's machine shows it's genuinely needed for smooth live FPS.

The directory is currently empty — from-scratch build. Deliverable is now a small but thorough project: working demo code + a documentation set covering the full ML lifecycle story, since the user must defend design choices live, not just show a script running.

## Model choice: Ultralytics YOLOv8n + built-in trackers
**Detector:** YOLOv8n (nano), via the `ultralytics` pip package, pretrained on COCO (already includes a `person` class — no retraining needed for the demo).

**Why YOLOv8n over alternatives:**
- Real-time on CPU — nano variant is the only COCO-pretrained detector family that reliably hits usable FPS on CPU; Faster/Mask R-CNN, SSD, YOLOv5m/l, Detectron2 are GPU-oriented and too slow for a live CPU demo.
- Tracking is *built in* — `model.track()` wires up ByteTrack and BoT-SORT directly, so "temporary ID" vs "persistent ID" is a one-line config change, not two pipelines.
- Actively maintained, trivial `pip install ultralytics`, exportable to ONNX/TensorRT later for real edge deployment (Jetson at a QSR entrance).
- Rejected alternatives to cite if asked: YOLOv5 (superseded, no built-in ReID tracker), Faster/Mask R-CNN (too slow on CPU), SSD-MobileNet (lower accuracy, no integrated tracker), MediaPipe/OpenPose (pose estimation, not multi-person ID tracking), Detectron2 (GPU-oriented, heavier setup).
- License note to mention if asked: Ultralytics is AGPL-3.0 — fine for a demo/POC; a closed-source commercial product would need an Ultralytics Enterprise license.

**Trackers (the "persistent vs temporary ID" contrast):**
- **Temporary ID → ByteTrack** (`bytetrack.yaml`): pure motion association (Kalman filter + IoU), no appearance memory. Lost track → ID gone. Re-appearance → brand-new incrementing ID, because ByteTrack has no way to recognize "this is the same person."
- **Persistent ID → BoT-SORT with Re-ID enabled** (`botsort.yaml`, `with_reid: True`): same motion association, *plus* an appearance embedding (built-in Re-ID CNN) per track, kept in a short-term in-memory gallery of recently-lost tracks. A new detection is compared by appearance similarity against that gallery before a new ID is minted — if it matches closely enough, the old ID is restored.
- Both are demoed with the same detector and the same `model.track()` call — only the tracker file/flag differs, so the code comments can point at the exact line that causes the behavioral difference.
- Known limit to state honestly: the Re-ID gallery is short-term (seconds, buffer-configurable), not a long-term recognition system. Absences of minutes+ will get a new ID either way — true long-term persistence needs a stored embedding database (see Database doc below), which is out of scope here.

## Files to create

### Code
- `requirements.txt` — `ultralytics`, `opencv-python`
- `utils/frame_quality.py` — small shared module used by both scripts. Implements the **binary classification** step in this pipeline: `is_blurry(frame) -> bool` (Laplacian variance threshold) and `is_too_dark(frame) -> bool` (mean brightness threshold). This is the concrete answer to "how does the code handle low-quality images": flagged frames are either skipped for detection or have their confidence threshold raised, and this is logged so it's visible during the demo.
- `detect_track_temporary_id.py` — self-contained script: loads YOLOv8n, per-frame quality check via `utils/frame_quality.py`, runs `model.track(source=video_path, tracker="bytetrack.yaml", classes=[0], conf=..., ...)`, draws boxes + track IDs, shows live window, prints running FPS. Heavy inline comments at the tracker call explaining ByteTrack's behavior.
- `detect_track_persistent_id.py` — near-identical script; only the tracker config differs (`botsort.yaml` + Re-ID). Top-of-file comment block diffs the two files line-by-line.
- Both scripts confirm a track only after **N consecutive detections** (small hit-streak counter) before drawing/counting it as a real person — this is the concrete false-positive mitigation (see Inference doc), stopping single-frame spurious detections (e.g. a poster, a reflection) from flickering a box onto the screen.

### Documentation set (root `README.md` + `docs/`)
- `README.md` — top-level overview: what the two scripts demonstrate, how to run them, and a short index linking to each doc below. This is the file to skim right before walking in tomorrow.
- `docs/TASK_TYPES.md` — maps each ML "task type" the user asked about to the exact code location:
  - **Object detection** (localization + multi-class classification combined): the core YOLOv8n forward pass — bounding box regression + softmax over COCO classes, filtered to `classes=[0]` (person). Point to the `model.track(...)` call in both scripts.
  - **Binary classification**: the frame-quality gate in `utils/frame_quality.py` (blurry/not-blurry, dark/not-dark) — a simple two-class decision that runs *before* the detector on every frame.
  - **Feature extraction**: the Re-ID embedding step inside BoT-SORT (`with_reid: True`) — a CNN produces an appearance feature vector per detected person, used for cosine-similarity matching against the short-term track gallery. Point to the tracker config line in `detect_track_persistent_id.py`.
  - A short table summarizing all three: task type | where in code | input | output | purpose.
- `docs/DATA_STRATEGY.md` — documentation-only (no training code, per user's decision), covering:
  - **Training set — synthetic data**: why synthetic is needed here — real QSR camera footage with diverse angles/lighting/crowd-density/occlusion is expensive and privacy-sensitive to collect and label at volume; synthetic generation (simulated store scenes, domain randomization of lighting/camera angle/clothing/occlusion) gives cheap, perfectly-labeled, edge-case-rich data (e.g. deliberately generating crowded/occluded/backlit scenes that are rare to capture for real).
  - **Validation set — real data only**: must be real to catch the synthetic-to-real domain gap; used for model/tracker hyperparameter selection (confidence threshold, Re-ID similarity threshold, NMS IoU) — never synthetic, or you'd just be validating against your own generator's biases.
  - **Test set — real data only, held out**: final unbiased accuracy read, collected from a different time/location than validation data if possible so it isn't quietly re-used for tuning.
  - Annotation format & tooling: bounding-box annotation in YOLO `.txt` format (`class x_center y_center width height`, normalized) for detection; explain CVAT (free, self-hosted, has video interpolation — well suited to tracking ground truth so you don't hand-label every frame) as the recommended annotation tool, with Roboflow named as a faster cloud-based alternative for managing synthetic+augmented sets.
  - Explicit statement that the demo itself skips this whole pipeline by using pretrained COCO weights, because the `person` class in COCO is already large and diverse enough to transfer well to a QSR entrance camera without fine-tuning — this doc exists so the user can explain *how they would* extend it, not because tomorrow's demo needs it.
- `docs/INFERENCE_AND_ACCURACY.md`:
  - **Frame handling**: confirms every frame is analyzed (no skipping) by default, states the measured FPS from local benchmarking (to be filled in after running once on the user's machine), and documents the frame-quality gate's effect on which frames get analyzed at full confidence vs. flagged.
  - **False positives in live inference**: concrete cases — a poster/mannequin/reflection misdetected as a person, a stationary object briefly classified as person 1-frame flicker — and the two mitigations actually in the code: (1) confidence threshold tuned above YOLO's default to cut low-confidence spurious boxes, (2) the N-consecutive-frame hit-streak before a track is confirmed, which kills single-frame flicker false positives.
  - **Accuracy**: published benchmark figures with explicit caveats that they're benchmark-dataset numbers, not a guarantee for the user's own footage — YOLOv8n COCO mAP50-95 ≈ 37 all-classes (person-class AP typically mid-50s%, dataset-dependent); ByteTrack ≈ 76-80 MOTA on MOT17; BoT-SORT ≈ 80-81 MOTA with measurably fewer ID switches due to Re-ID. Recommends the user visually count ID switches on their own test clip as the real proxy metric, since there's no ground truth for a random video handed to them tomorrow.
- `docs/DATABASE_DESIGN.md`:
  - States plainly: **no external database is used** in this demo. The Re-ID gallery BoT-SORT keeps is an in-memory, session-scoped structure (cleared when the script exits) — appropriate because current scope is single-video, single-session tracking with no requirement to recognize a person across separate visits/days.
  - Explains what *would* be needed if scope grew to true cross-session persistent footfall identity (e.g. recognizing a repeat customer days later): a **vector database** (FAISS for a local/self-hosted option, or a managed one like Chroma/Milvus) storing each person's Re-ID embedding keyed by ID, queried by approximate-nearest-neighbor similarity search — chosen over a relational/SQL database because the lookup is "find embeddings similar to this one," not an exact-match query, which SQL isn't built for.
  - Names this explicitly as a **documented future extension, not implemented**, matching the confirmed scope (detection + tracking only).

## Verification
- Run both scripts against a short local test clip (e.g., a walk-in/walk-out phone recording) on CPU and confirm:
  - `detect_track_temporary_id.py` assigns a new ID number each time the same person re-enters frame.
  - `detect_track_persistent_id.py` keeps the same ID number when that person re-enters within a short time window.
- Record the actual measured FPS from this run and fill it into `docs/INFERENCE_AND_ACCURACY.md` (replacing any placeholder) so the accuracy/performance doc reflects the real machine, not just a benchmark citation.
- Sanity-check on a second clip with two-plus people to confirm boxes/IDs behave correctly with multiple simultaneous tracks, and deliberately test one occlusion (one person briefly walks behind another) to observe and note whether an ID switch occurs — this becomes a live example for the edge-cases discussion.
- Test the frame-quality gate by darkening/blurring a clip (or a portion of one) and confirming the console/log shows frames being flagged.
