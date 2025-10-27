# Repository Guidelines

## Project Structure & Module Organization
Operational scripts now stay in `scripts/` (`train.py`, `predict.py`), while reusable code sits in the lightweight `basedetect/` package (`uv run --module basedetect` for smoke tests). Dataset configs live in `configs/`, datasets are grouped under `datasets/` (`datasets2/`, `datasets-initial/`, `demo/`), demo clips in `test/`, and generated assets collect in `artifacts/` (`runs/`, `outputs/`). Pretrained checkpoints are grouped beneath `weights/pretrained/` for clarity.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (preferred) or `pip install -r requirements.txt`. Typical workflows:
```
uv run scripts/train.py      # fine-tune YOLO on configs/data.yaml
uv run scripts/predict.py    # run tracking on test/test3.mp4
uv run --module basedetect   # smoke-test entry point
```
Always verify CUDA visibility (`CUDA_VISIBLE_DEVICES`) before kicking off long jobs; the Ultralytics trainer will fall back to CPU otherwise.

## Coding Style & Naming Conventions
Write Python 3.9+ code with 4-space indentation and descriptive snake_case identifiers (e.g., `gray_rgb`). Keep modules single-purpose and prefer helper functions over sprawling scripts. Add inline comments only when logic is not obvious (model configs, video IO). Maintain English log/print strings even if dataset metadata is multilingual.

## Testing Guidelines
There is still no heavy automated suite, so rely on targeted runs: `uv run scripts/predict.py` against each clip in `test/` and confirm the saved `artifacts/outputs/output.avi`. When touching training logic, document metrics from `artifacts/runs/basedetect/results.csv` (or similar) and attach screenshots of key plots. Use `uv run scripts/test_cli.py` for quick parameter smoke tests. Name any new manual-test scripts `test_<feature>.py` and store them near the code they exercise.

## Commit & Pull Request Guidelines
The history is currently empty, so adopt Conventional Commits (e.g., `feat: add thermal preprocessor`) to set the tone. Each PR should include: a concise summary, reproduction steps or command logs, references to tracked issues, and before/after evidence (loss curves, detection screenshots). Keep branches focused; rebase on `main` before requesting review to avoid noisy diffs.

## Security & Configuration Tips
Weights can be large; avoid committing anything under `artifacts/` or raw `datasets/`. Store secrets (API keys, camera URLs) in env vars or `.env` ignored locally. When sharing configs, scrub personal paths and confirm that `configs/data.yaml` only references public locations.
