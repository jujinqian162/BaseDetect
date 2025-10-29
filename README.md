# BaseDetect

[中文文档](README.zh.md)

BaseDetect packages a YOLOv8-based workflow for detecting and tracking machine bases in monochrome video feeds. The repository ships training and inference scripts, a smoke-test dataset, pretrained checkpoints, and sample footage so you can validate the pipeline quickly and iterate on your own data.

## Repository Layout
```
BaseDetect/
├─ basedetect/             # Package entry point, usable via `uv run --module basedetect`
├─ scripts/                # Operational scripts (training, inference)
├─ configs/                # Dataset / experiment configuration files
├─ datasets/               # Dataset collections and exports
│  ├─ datasets2/           # Roboflow export referenced by configs/data.yaml
│  ├─ datasets-initial/    # Default dataset referenced by configs/data-initial.yaml
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
- First run: create runtime directories so inference outputs and checkpoints have a home.
  ```bash
  uv run --module basedetect
  ```
- Before long training jobs, double-check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` to ensure Ultralytics sees the intended GPU.

## Quickstart Checklist
1. `uv sync` to install dependencies.
2. Ensure the dataset referenced by `configs/data-initial.yaml` (`datasets/datasets-initial/`) is available (see Data & Configuration Management).
3. `uv run scripts/train.py` to fine-tune using `configs/data-initial.yaml`; confirm `artifacts/runs/basedetect/` contains logs and `weights/best.pt`.
4. `uv run scripts/predict.py` to process `test/test3.mp4` and produce `artifacts/outputs/output.avi`.

## Training Workflow
Run the training CLI:
```bash
uv run scripts/train.py [options]
```
- `--config PATH` — dataset YAML. Defaults to `configs/data-initial.yaml`.
- `--model SOURCE` — initial weights (file path or Ultralytics model name). Defaults to `weights/pretrained/yolov8n.pt`.
- `--epochs / --batch / --imgsz` — standard hyperparameters with defaults `10`, `8`, and `640`.
- `--device` — Ultralytics device string; `auto` prefers the first GPU when CUDA is available.
- `--project / --name` — overrides for the Ultralytics experiment directory (`artifacts/runs/basedetect` by default).
- `--resume`, `--patience` — control checkpoint resumption and early stopping.
- Whenever training falls back to a pretrained checkpoint (the default `yolov8n.pt` or another Ultralytics weight), the CLI emits a yellow bilingual warning so you know initial weights come from pretraining.

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
- `--device`, `--conf`, `--no-save`, `--unshow` — mirror the Ultralytics CLI options.
- When `--weights auto` cannot find a freshly trained checkpoint, the script emits a yellow bilingual warning and falls back to the bundled pretrained model so you know which weights are in use.

Common patterns:
```bash
# Live webcam preview with an on-screen window
uv run scripts/predict.py --source 0

# disable on-screen preview if you're running headless
uv run scripts/predict.py --source 0 --unshow

# Evaluate a specific checkpoint and keep the rendered video
uv run scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --source test/test3.mp4
```

## Data & Configuration Management
- `configs/` stores dataset descriptors and experiment settings. `configs/data-initial.yaml` (default) points to `datasets/datasets-initial/`, while `configs/data.yaml` references the Roboflow export under `datasets/datasets2/`. Copy either when creating new variants and adjust paths accordingly.
- Place all dataset exports under `datasets/` (e.g., `datasets/custom_v1/`). Each dataset folder should contain `train/`, `valid/`, and `test/` subdirectories that match the YAML paths.
- Keep large assets (datasets, artifacts) out of commits—double-check `git status` before pushing.

## Manual Validation
- `uv run scripts/predict.py` and confirm `artifacts/outputs/output.avi` updates, contains detections, and prints a yellow bilingual warning only when auto-falling back to pretrained weights.
- Repeat inference for each clip in `test/` to ensure different resolutions behave correctly.
- After training changes, record fresh metrics from `artifacts/runs/<run_name>/results.csv`; grab loss/precision plots when useful for reviews.
- Whenever CLI argument parsing or default paths change, run `uv run scripts/test_cli.py` for parameter smoke tests and `uv run --module basedetect` to confirm runtime directories are ready.

## Troubleshooting
- **Falls back to CPU** — check `CUDA_VISIBLE_DEVICES` and `nvidia-smi`. Pass `--device 0` explicitly if the environment masks GPUs.
- **Weights not found** — confirm `weights/pretrained/yolov8n.pt` exists or point `--weights` to the desired checkpoint.
- **Missing output directory** — run the smoke test or create `artifacts/outputs/` manually before inference.
- **Label mismatch after Roboflow export** — export in YOLOv8 format and ensure `names` in `configs/*.yaml` matches the dataset.

## Contributing
Follow `AGENTS.md` for Conventional Commits, focused PRs, and manual test evidence. Surface dataset access or architectural questions via issues so the team can coordinate fixes promptly.
