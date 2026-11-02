# YOLO Soccer Recognition

A Python computer vision pipeline that turns soccer footage into annotated video with player tracking, team assignments, estimated ball control, and player speed and distance. The project combines YOLO object detection with ByteTrack, jersey-color clustering, optical flow, and perspective transformation.

## Features

- **Object detection and tracking:** Detect players, goalkeepers, referees, and the ball with a custom YOLO model. ByteTrack maintains player and referee IDs across frames; goalkeepers are grouped with players for downstream processing.
- **Team classification:** Use two-stage K-means clustering to extract jersey colors from player crops and assign players to two teams.
- **Ball-control estimates:** Associate ball detections with nearby players and display each team's cumulative share of estimated possession.
- **Ball interpolation:** Fill gaps in detected ball bounding boxes with pandas interpolation and backfilling.
- **Camera-motion estimation:** Track image features with pyramidal Lucas–Kanade optical flow and use median displacement to adjust object positions.
- **Speed and distance estimates:** Map positions into a calibrated pitch region and estimate player speed in km/h and accumulated distance in meters.
- **Annotated video:** Export team-colored player markers, tracking IDs, ball and possession indicators, camera movement, and movement statistics.
- **Cached processing:** Reuse detection tracks and camera-motion estimates through pickle stubs.

## Technology

Python · Ultralytics YOLO · ByteTrack (`trackers`) · Supervision · OpenCV · NumPy · pandas · scikit-learn

## Getting started

### 1. Set up an environment

```bash
git clone https://github.com/umbraudax/YOLOSoccerRecognition.git
cd YOLOSoccerRecognition
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics supervision trackers opencv-python numpy pandas scikit-learn
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The repository does not currently pin dependency versions; the `trackers` package must provide `ByteTrackTracker` with the API used in `tracker_helpers/tracker.py`.

### 2. Provide model weights and footage

Create the working directories:

```bash
mkdir -p models input output_videos stubs
```

Place a trained detection checkpoint at `models/best.pt` and a soccer video at `input/video1.mp4`. These assets are not included in the repository. The model must expose the class names `ball`, `goalkeeper`, `player`, and `referee`.

### 3. Configure the pipeline

The current implementation is configured through source constants:

| Setting | Location | Default / action |
| --- | --- | --- |
| Input video | `main.py` | `VIDEO_NAME = "video1"`; omit the extension |
| Model checkpoint | `main.py` | `./models/best.pt` |
| Inference device | `tracker_helpers/tracker.py` | `device='mps'`; change to `'cpu'` or a supported CUDA device as needed |
| Track and camera caches | `main.py` | Set both `read_from_stub` arguments to `False` for a new video or changed model |
| Pitch calibration | `view_transformer/view_transformer.py` | Set four image vertices and their corresponding real-world coordinates for your footage |
| Frame rate | `speed_and_distance_estimator/speed_and_distance_estimator.py` and `utils/video_utils.py` | Both currently use 24 FPS; match them to the source video |
| Team initialization | `main.py` | `INIT_FRAME = 3`; choose a frame with usable detections from both teams |

Camera-feature masks and ball-control overlay coordinates are also fixed in pixel space. Adjust them in `camera_movement_estimator/camera_movement_estimator.py` and `tracker_helpers/tracker.py` for different resolutions or camera views. Give camera caches video-specific paths when working with multiple clips.

### 4. Run

From the repository root:

```bash
python main.py
```

The pipeline writes `output_videos/output_video.avi` using the XVID codec. It loads video frames into memory, so start with a short clip. Only reuse stubs that match the exact video and processing configuration, and only load trusted pickle files.

## How it works

1. Read the video and run batched YOLO inference, or load matching cached tracks.
2. Use ByteTrack to associate player and referee detections across frames.
3. Extract foot positions for people and center positions for the ball.
4. Estimate camera displacement, adjust positions, and transform valid positions into pitch coordinates.
5. Estimate player speed and distance over five-frame windows.
6. Interpolate missing ball detections, classify teams by jersey color, and assign ball control using player proximity.
7. Render annotations and save the output video.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Pipeline entry point and video configuration |
| `tracker_helpers/` | YOLO inference, ByteTrack association, interpolation, and annotations |
| `team_assigner/` | Jersey-color extraction and team clustering |
| `player_assigner/` | Ball-to-player association |
| `camera_movement_estimator/` | Optical flow and camera-displacement estimates |
| `view_transformer/` | Image-to-pitch perspective transformation |
| `speed_and_distance_estimator/` | Player movement estimates and overlays |
| `utils/` | Video I/O and bounding-box geometry helpers |
| `devanalysis/` | Training and color-analysis notebooks, plus saved training logs |
| `football-players-detection-1/` | Dataset configuration, attribution, and committed label files |
| `stubs/` | Cached tracking results |
| `yolo_inference.py` | Standalone inference experiment |

## Training

`devanalysis/yolo_train.ipynb` contains a Roboflow download workflow and an Ultralytics training example using `yolo26n.pt` for 30 epochs. To use the notebook, install its additional dependencies:

```bash
python -m pip install roboflow python-dotenv jupyter
```

Set `ROBOFLOW_API_KEY` in your environment, download the dataset, and update the notebook's dataset path and device for your machine. The checked-in training path references `../training/football-players-detection-1/data.yaml`, which does not match the current repository layout. Ensure the YAML's train, validation, and test image paths resolve to your downloaded data, then copy the resulting best checkpoint to `models/best.pt`.

Dataset: [Football Players Detection, version 1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1), provided by a Roboflow user under **CC BY 4.0**, as recorded in the dataset metadata. Training images and trained weights must be obtained separately.

## Current limitations

This is an experimental offline analysis pipeline. Speed, distance, and ball control are estimates, and depend on detection quality, track continuity, frame rate, and pitch calibration.

- The pitch transform is manually calibrated to one camera view. Significant camera changes require recalibration; the current camera adjustment subtracts per-frame displacement rather than cumulative motion.
- Team classification assumes two visually distinguishable jersey-color clusters. Lighting, occlusion, and goalkeeper kits can affect assignments.
- Ball control uses a pixel-distance heuristic. The current assignment loop does not update its running minimum distance, so it can select the last qualifying player rather than the closest one. Unassigned frames retain the previous team, with Team 1 used initially.
- The speed estimator can encounter a zero-duration final window for some clip lengths; skip windows where the ending frame equals the starting frame if this occurs.
- The repository has no pinned environment or packaged command-line interface. Setup requires the assets and configuration changes described above.

## Acknowledgments

Built with [Ultralytics](https://github.com/ultralytics/ultralytics), [Supervision](https://github.com/roboflow/supervision), [Trackers](https://github.com/roboflow/trackers), [OpenCV](https://opencv.org/), and [scikit-learn](https://scikit-learn.org/). Dataset attribution and licensing are included in `football-players-detection-1/`.
