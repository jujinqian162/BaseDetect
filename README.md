# BaseDetect

[中文文档](README.zh.md)

YOLOv8-based pipeline for detecting and tracking machine bases in monochrome video feeds. The repo bundles training scripts, inference utilities, and reference assets so you can fine-tune Ultralytics models on the Roboflow Base Inspection dataset and replay detections on sample footage.

## Repository Layout
- `scripts/train.py` — CLI for training that bootstraps a lightweight demo dataset and launches Ultralytics with sensible defaults.
- `scripts/predict.py` — CLI that pulls the newest weights (or a fallback checkpoint), converts frames to grayscale, and streams/saves annotated video.
- `basedetect/` — lightweight package used for `python -m basedetect` smoke tests or future shared helpers.
- `configs/data.yaml` — dataset + class metadata shared by the trainer.
- `datasets/` — expected Roboflow export with `train/`, `valid/`, and `test/` splits.
- `test/` — short demo clips for quick regression checks.
- `artifacts/` — auto-created experiment logs (`artifacts/runs/`) and rendered videos (`artifacts/outputs/`).
- `weights/pretrained/` — cached YOLO checkpoints used to bootstrap training.
- `AGENTS.md` — contributor guide covering style, testing, and review expectations.

### Project Structure Explained
```
BaseDetect/
├─ basedetect/             # Python package entry point and shared helpers
├─ scripts/                # Operational scripts (training, inference)
├─ configs/                # Dataset + experiment configuration files
├─ datasets/               # Roboflow exports: train/ valid/ test/ splits
├─ test/                   # Small demo clips for quick regressions
├─ weights/pretrained/     # Cached YOLO checkpoints
├─ artifacts/              # Generated runs/, metrics, and output videos
├─ README.md / README.zh.md# Project documentation
└─ AGENTS.md               # Contributor guide
```
The separation keeps code, configs, raw data, and generated artifacts isolated so the workspace stays tidy and large files remain outside version control.

## Setup
1. Install Python 3.9+ and a recent CUDA toolkit if you plan to train on GPU.
2. Prefer [`uv`](https://github.com/astral-sh/uv) for reproducible environments:
   ```bash
   uv sync
   ```
   or fall back to pip:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Bootstrap folders and generate the synthetic demo dataset:
   ```bash
   uv run python -m basedetect
   ```
4. Verify `nvidia-smi` shows the intended device before long runs (`CUDA_VISIBLE_DEVICES=0` is respected by Ultralytics).

## Train & Evaluate
Fine-tune from the provided Ultralytics checkpoints (defaults to a synthetic demo dataset so the command succeeds immediately):
```bash
uv run python scripts/train.py
```
Key outputs land under `artifacts/runs/basedetect/` (check `results.csv`, `weights/best.pt`, and rendered plots). Switch to your real Roboflow export by pointing at the bundled config:
```bash
uv run python scripts/train.py --config configs/data.yaml
```
Add other overrides (epochs, batch size, model variant) through CLI flags, e.g. `--epochs 50 --model yolov8s.pt`.

### Train CLI options
```
uv run python scripts/train.py [options]
```
- `--config PATH` — dataset YAML; the default `'auto'` generates a synthetic dataset at `datasets/demo/data.yaml`.
- `--model SOURCE` — initial weights (file path or Ultralytics model name). Default points to `weights/pretrained/yolov8n.pt`.
- `--epochs N` — number of training epochs (default `10`).
- `--batch N` — batch size per iteration (default `8`).
- `--imgsz N` — square image resolution used for training (default `640`).
- `--device VALUE` — Ultralytics device string. `auto` (default) maps to `0` when CUDA is available, otherwise `cpu`.
- `--workers N` — dataloader worker processes (default `4`).
- `--project PATH` — parent directory for Ultralytics experiment folders (default `artifacts/runs`).
- `--name RUN_NAME` — experiment sub-directory name (default `basedetect`).
- `--patience N` — early-stopping patience in epochs (default `10`).
- `--resume` — resume the most recent checkpoint saved in the run directory.

Examples:
```bash
# GPU training on Roboflow export with more epochs
uv run python scripts/train.py --config configs/data.yaml --epochs 50 --device 0

# CPU-only experiment with a smaller model
uv run python scripts/train.py --model yolov8n.pt --device cpu --batch 4
```

## Run Tracking Demo
Use the grayscale tracker on bundled footage (auto-picks the latest trained weights, otherwise falls back to `yolov8n.pt`):
```bash
uv run python scripts/predict.py
```
By default it reads `test/test3.mp4`, writes `artifacts/outputs/output.avi`, and runs headless. Add `--show` for a live window or `--weights /path/to/best.pt` to inspect a specific checkpoint. Use `--source 0` to attach a webcam.

### Predict CLI options
```
uv run python scripts/predict.py [options]
```
- `--weights SOURCE` — path or model name for inference weights; `auto` (default) selects the newest `artifacts/runs/**/weights/best.pt` or falls back to `weights/pretrained/yolov8n.pt`.
- `--source INPUT` — video path or camera index (default `test/test3.mp4`).
- `--output PATH` — annotated video destination when saving is enabled (default `artifacts/outputs/output.avi`).
- `--device VALUE` — Ultralytics device string; `auto` (default) maps to `0` when CUDA is detected, otherwise `cpu`.
- `--conf THRESH` — detection confidence threshold (default `0.25`).
- `--no-save` — disable writing the annotated video to disk.
- `--show` — open an OpenCV window for live preview (press `q` to exit).

Examples:
```bash
# Live webcam preview with GUI window
uv run python scripts/predict.py --source 0 --show

# Benchmark a specific checkpoint without saving output
uv run python scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --no-save
```

## Data Preparation
The synthetic smoke-test dataset lives under `datasets/demo/` with its own `data.yaml` generated automatically. `configs/data.yaml` targets the Roboflow project (`robocon-ozkss/base-inspection-txwpc`). Export it with YOLOv8 format and drop the `train/`, `valid/`, and `test/` folders under `datasets/`. Keep raw data and generated artifacts out of version control; use `.gitignore` to skip large assets.

## Troubleshooting
- **Video fails to open**: confirm `test/*.mp4` exists and codec support via `ffmpeg -codecs`.
- **No detections**: verify your weights path and that inference frames are still 3-channel after grayscale conversion.
- **CPU fallback**: Ultralytics drops to CPU quietly if CUDA is unavailable—watch the console log at startup.

## Contributing
Follow the practices in `AGENTS.md` (Conventional Commits, concise PR summaries, manual test evidence). File issues for architecture questions or dataset access problems so the team can triage quickly.
