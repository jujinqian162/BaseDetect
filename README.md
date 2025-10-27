# BaseDetect

[中文文档](README.zh.md)

BaseDetect packages a YOLOv8-based workflow for detecting and tracking machine bases in monochrome video feeds. The repository ships training and inference scripts, a smoke-test dataset, pretrained checkpoints, and sample footage so you can validate the pipeline quickly and iterate on your own data.

## Repository Layout
```
BaseDetect/
├─ basedetect/             # Package entry point, usable via `uv run --module basedetect`
├─ scripts/                # Operational scripts (training, inference)
├─ configs/                # Dataset / experiment configuration files
├─ datasets/               # Roboflow exports (train/ valid/ test splits)
├─ test/                   # Short demo clips for regression checks
├─ weights/pretrained/     # Cached YOLO checkpoints
├─ artifacts/              # Generated runs/, metrics, and output videos
├─ README.md / README.zh.md# Project documentation
└─ AGENTS.md               # Contributor guide
```
Code, configuration, raw data, and generated artifacts live in separate directories so the workspace stays tidy and large files stay out of git.

## Environment Setup
- Python: 3.9 or newer. Install a recent CUDA toolkit if you plan to train on GPU.
- Dependencies: prefer [`uv`](https://github.com/astral-sh/uv) for reproducible environments.
  ```bash
  uv sync
  ```
  If you need to fall back to pip:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- First run: bootstrap folders and the synthetic demo dataset.
  ```bash
  uv run --module basedetect
  ```
- Before long training jobs, double-check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` to ensure Ultralytics sees the intended GPU.

## Quickstart Checklist
1. `uv sync` to install dependencies.
2. `uv run --module basedetect` to create the demo dataset and required directories.
3. `uv run scripts/train.py` to fine-tune on the demo data; confirm `artifacts/runs/basedetect/` contains logs and `weights/best.pt`.
4. `uv run scripts/predict.py` to process `test/test3.mp4` and produce `artifacts/outputs/output.avi`.

To switch to the Roboflow dataset, drop the exported `train/`, `valid/`, and `test/` folders under `datasets/<name>/` and point the trainer at the matching `configs/*.yaml`.

## Training Workflow
Run the training CLI:
```bash
uv run scripts/train.py [options]
```
- `--config PATH` — dataset YAML. The default `'auto'` builds a demo dataset at `datasets/demo/data.yaml`.
- `--model SOURCE` — initial weights (file path or Ultralytics model name). Defaults to `weights/pretrained/yolov8n.pt`.
- `--epochs / --batch / --imgsz` — standard hyperparameters with defaults `10`, `8`, and `640`.
- `--device` — Ultralytics device string; `auto` prefers the first GPU when CUDA is available.
- `--project / --name` — overrides for the Ultralytics experiment directory (`artifacts/runs/basedetect` by default).
- `--resume`, `--patience` — control checkpoint resumption and early stopping.

Recommended habits:
- After each run, capture metrics from `artifacts/runs/basedetect/results.csv` (mAP, precision, recall).
- Use distinct `--name` values when comparing experiments to avoid overwriting outputs.
- When tuning hyperparameters, note the commands and logs in your PR or experiment diary for reproducibility.

Examples:
```bash
# Longer GPU training on the Roboflow export
uv run scripts/train.py --config configs/data.yaml --epochs 50 --device 0

# Lightweight CPU experiment with a smaller batch
uv run scripts/train.py --model yolov8n.pt --device cpu --batch 4
```

## Tracking & Inference
Run the inference CLI:
```bash
uv run scripts/predict.py [options]
```
- `--weights` — selects the newest `artifacts/runs/**/weights/best.pt` or falls back to `weights/pretrained/yolov8n.pt`.
- `--source` — video path or camera index (`test/test3.mp4` by default).
- `--output` — annotated video destination (`artifacts/outputs/output.avi` by default).
- `--device`, `--conf`, `--no-save`, `--show` — mirror the Ultralytics CLI options.

Common patterns:
```bash
# Live webcam preview with an on-screen window
uv run scripts/predict.py --source 0 --show

# Evaluate a specific checkpoint and keep the rendered video
uv run scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --source test/test3.mp4
```

## Data & Configuration Management
- `configs/` stores dataset descriptors and experiment settings. `configs/data.yaml` points to the Roboflow project `robocon-ozkss/base-inspection-txwpc`. Copy it when creating new variants and adjust paths accordingly.
- Organize Roboflow exports under descriptive directories such as `datasets/roboflow_v1/`. Place the `train/`, `valid/`, and `test/` folders directly inside.
- `.gitignore` already excludes `datasets/` and `artifacts/`. Always review `git status` before committing to ensure large files stay local.

## Manual Validation
- `uv run scripts/predict.py` and confirm `artifacts/outputs/output.avi` updates and contains bounding boxes and tracks.
- Repeat inference for each clip in `test/` to ensure different resolutions behave correctly.
- After training changes, record fresh metrics from `artifacts/runs/<run_name>/results.csv`; grab loss/precision plots when useful for reviews.
- Whenever CLI argument parsing or default paths change, run `uv run --module basedetect` as a smoke test to confirm directories and demo data still initialize correctly.

## Troubleshooting
- **Falls back to CPU** — check `CUDA_VISIBLE_DEVICES` and `nvidia-smi`. Pass `--device 0` explicitly if the environment masks GPUs.
- **Weights not found** — confirm `weights/pretrained/yolov8n.pt` exists or point `--weights` to the desired checkpoint.
- **Missing output directory** — run the smoke test or create `artifacts/outputs/` manually before inference.
- **Label mismatch after Roboflow export** — export in YOLOv8 format and ensure `names` in `configs/*.yaml` matches the dataset.

## Contributing
Follow `AGENTS.md` for Conventional Commits, focused PRs, and manual test evidence. Surface dataset access or architectural questions via issues so the team can coordinate fixes promptly.
