# BaseDetect 项目指南

BaseDetect 利用 Ultralytics YOLOv8 在单色视频中检测与跟踪机床底座，仓库内同时包含训练脚本、推理脚本、权重与示例视频，可帮助你从 Roboflow 提供的数据中快速微调模型并验证效果。

## 项目结构
```
BaseDetect/
├─ basedetect/             # Python 包，可通过 `python -m basedetect` 进行快速冒烟测试
├─ scripts/                # 训练与推理脚本（train.py、predict.py）
├─ configs/                # 数据集与其他配置文件（data.yaml）
├─ datasets/               # Roboflow 导出的 train/valid/test 数据
├─ test/                   # 演示/回归使用的视频片段
├─ weights/pretrained/     # 预训练的 YOLO 权重缓存
├─ artifacts/              # 训练日志、导出的可视化以及 `outputs/` 视频
├─ AGENTS.md, README*.md   # 贡献者与项目文档
└─ requirements.txt 等     # 依赖与工程元数据
```
该结构将"代码"与"数据/产物"分开，便于清理工作区并避免误提交大型文件。

## 环境准备
1. 安装 Python 3.9+ 和 CUDA（如需 GPU 训练）
2. 推荐使用 [uv](https://github.com/astral-sh/uv)：
   ```bash
   uv sync
   ```
   或使用 venv + pip：
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 初始化目录并生成演示数据集：
   ```bash
   uv run python -m basedetect
   ```
4. 训练前确认 `nvidia-smi` 输出目标 GPU，并在需要时设置 `CUDA_VISIBLE_DEVICES`

## 训练与评估
默认使用自动生成的演示数据集，确保开箱即可运行：
```bash
uv run python scripts/train.py
```
- 如果存在训练历史，Ultralytics 工具会将结果写入 `artifacts/runs/basedetect/`（包含 `results.csv`、`weights/best.pt` 等）
- 默认从 `yolov8n.pt` 微调，可通过 `--model` 更换
- 切换到真实 Roboflow 数据集：
  ```bash
  uv run python scripts/train.py --config configs/data.yaml
  ```
- 其他超参数可通过命令行传入，例如 `--epochs 50 --model yolov8s.pt --batch 16`

### 训练脚本参数
```
uv run python scripts/train.py [参数]
```
- `--config PATH`：数据集配置文件；默认值 `'auto'` 会生成存放在 `datasets/demo/data.yaml` 的演示数据集。
- `--model SOURCE`：初始化权重（文件路径或 Ultralytics 模型名），默认指向 `weights/pretrained/yolov8n.pt`。
- `--epochs N`：训练轮数（默认 `10`）。
- `--batch N`：每批次的图像数量（默认 `8`）。
- `--imgsz N`：训练时的输入尺寸，使用方形图像（默认 `640`）。
- `--device VALUE`：传递给 Ultralytics 的设备字符串；`auto`（默认）在检测到 CUDA 时映射为 `0`，否则回退到 `cpu`。
- `--workers N`：数据加载进程数（默认 `4`）。
- `--project PATH`：Ultralytics 实验目录的父路径（默认 `artifacts/runs`）。
- `--name RUN_NAME`：实验子目录名称（默认 `basedetect`）。
- `--patience N`：早停策略的耐心值（默认 `10`）。
- `--resume`：从最近一次保存的检查点继续训练。

示例：
```bash
# 在 GPU 上训练 Roboflow 数据集，并延长训练轮数
uv run python scripts/train.py --config configs/data.yaml --epochs 50 --device 0

# 在 CPU 上快速试验，使用较小 batch
uv run python scripts/train.py --model yolov8n.pt --device cpu --batch 4
```

## 推理与跟踪
```bash
uv run python scripts/predict.py
```
程序默认读取 `test/test3.mp4`，优先选用最近一次训练生成的 `weights/best.pt`，否则回退到 `yolov8n.pt`，并在后台写出 `artifacts/outputs/output.avi`。常用参数：
- `--show`：在桌面环境中弹出实时窗口
- `--weights /path/to/best.pt`：手动指定权重
- `--source 0`：切换到摄像头或自定义视频源

### 推理脚本参数
```
uv run python scripts/predict.py [参数]
```
- `--weights SOURCE`：权重文件路径或模型名称；默认 `auto` 会选择最新的 `artifacts/runs/**/weights/best.pt`，若不存在则使用 `weights/pretrained/yolov8n.pt`。
- `--source INPUT`：视频路径或摄像头索引（默认 `test/test3.mp4`）。
- `--output PATH`：保存标注视频的输出路径（默认 `artifacts/outputs/output.avi`）。
- `--device VALUE`：推理设备；`auto`（默认）在有 CUDA 时取 `0`，否则取 `cpu`。
- `--conf THRESH`：检测置信度阈值（默认 `0.25`）。
- `--no-save`：不写出标注视频。
- `--show`：弹出 OpenCV 窗口实时显示（按 `q` 退出）。

示例：
```bash
# 打开摄像头并显示实时窗口
uv run python scripts/predict.py --source 0 --show

# 指定某次实验的权重进行离线推理且不保存视频
uv run python scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --no-save
```

## 数据准备
演示数据集位于 `datasets/demo/`，对应的 `data.yaml` 会在初始化时自动生成；`configs/data.yaml` 指向 Roboflow 项目 `robocon-ozkss/base-inspection-txwpc`。导出 YOLOv8 格式后，将 `train/valid/test` 文件夹放入 `datasets/`。为避免仓库膨胀，确保 raw 数据与 `artifacts/` 目录都在 `.gitignore` 中。

## 贡献说明
遵循 `AGENTS.md` 中的提交约定（Conventional Commits、PR 说明、手动测试证据）。在 issue 中同步数据访问或架构上的问题，以便协作。

## 资源
- 英文文档：README.md
- 贡献者指南：AGENTS.md
