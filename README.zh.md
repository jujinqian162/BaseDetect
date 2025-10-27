# BaseDetect 项目指南

BaseDetect 针对机床底座检测与跟踪场景封装了基于 Ultralytics YOLOv8 的训练与推理流程。仓库内提供最小化数据集、脚本工具、预训练权重以及示例视频，帮助快速验证流程并在自有数据上迭代。

## 项目结构
```
BaseDetect/
├─ basedetect/             # Python 包，可 `uv run --module basedetect` 快速冒烟
├─ scripts/                # 训练与推理脚本（train.py、predict.py）
├─ configs/                # 数据/任务配置（data.yaml 等）
├─ datasets/               # Roboflow 导出的 train/valid/test
├─ test/                   # 演示或回归使用的视频片段
├─ weights/pretrained/     # 预训练 YOLO 权重缓存
├─ artifacts/              # 训练日志、推理输出、临时可视化
├─ AGENTS.md, README*.md   # 项目文档
└─ requirements.txt ...    # 依赖说明与工程元数据
```
代码与生成产物分开存放，便于清理工作目录，也能避免无意提交大型文件。

## 环境准备
- Python 版本：3.9 及以上；若需 GPU，请预装 CUDA 驱动并确认 `nvidia-smi` 正常输出。
- 依赖安装：推荐使用 [uv](https://github.com/astral-sh/uv)。
  ```bash
  uv sync
  ```
  如不使用 uv，可手动创建虚拟环境：
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- 初次启动建议运行一次冒烟脚本，用于创建演示数据与输出目录：
  ```bash
  uv run --module basedetect
  ```
- 长时间训练前，检查 `CUDA_VISIBLE_DEVICES` 是否指向目标 GPU；Ultralytics 若检测不到 CUDA 会直接落到 CPU。

## 快速开始
1. `uv sync` 安装依赖。
2. 确保 `configs/data-initial.yaml` 对应的数据集已准备完毕（参见“数据与配置管理”）。
3. `uv run scripts/train.py` 使用 `configs/data-initial.yaml` 进行训练，确认 `artifacts/runs/basedetect/` 写出日志与 `weights/best.pt`。
4. `uv run scripts/predict.py` 跑通 `test/test3.mp4`，验证在 `artifacts/outputs/output.avi` 生成带框视频。

如果需要临时体验 demo 数据集，可运行 `uv run --module basedetect` 并在训练命令中添加 `--config auto`。

## 训练流程
运行训练脚本：
```bash
uv run scripts/train.py [参数]
```
- `--config PATH`：数据集配置文件。默认读取 `configs/data-initial.yaml`；传入 `auto` 会创建 `datasets/demo/data.yaml` 并使用生成的小型数据。
- `--model SOURCE`：初始权重路径或 Ultralytics 模型名。默认读取 `weights/pretrained/yolov8n.pt`。
- `--epochs / --batch / --imgsz`：常规超参，缺省值分别为 `10`、`8`、`640`。
- `--device`：传递给 YOLO 的设备字符串；在 `auto` 模式下优先使用首块 GPU。
- `--project / --name`：实验输出位置，默认为 `artifacts/runs/basedetect`。
- `--resume` 与 `--patience`：对应重训与早停策略。

建议习惯：
- 训练完成后记录 `artifacts/runs/basedetect/results.csv` 中的关键指标（mAP、precision、recall）。
- 若需要对比多次实验，可调整 `--name` 生成单独目录，避免覆盖。
- 调参与模型切换建议在 PR 或笔记中留下一份命令行及训练日志，便于后续复现。

进阶示例：
```bash
# 使用 Roboflow 数据集并延长训练
uv run scripts/train.py --config configs/data.yaml --epochs 50 --device 0

# CPU 上快速试验，调小 batch
uv run scripts/train.py --model yolov8n.pt --device cpu --batch 4
```

## 推理与跟踪
```bash
uv run scripts/predict.py [参数]
```
- `--weights`：自动查找最近的 `artifacts/runs/**/weights/best.pt`，如未找到则回退到 `weights/pretrained/yolov8n.pt`。
- `--source`：输入源，既可以是 `test/test3.mp4` 等文件，也可以直接给摄像头索引。
- `--output`：输出视频路径，默认 `artifacts/outputs/output.avi`。
- `--conf / --device / --no-save / --show` 等参数与 Ultralytics CLI 用法一致，便于迁移。

常见用法：
```bash
# 推理摄像头画面并弹窗查看
uv run scripts/predict.py --source 0 --show

# 指定历史实验的 best 权重，生成离线视频
uv run scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --source test/test3.mp4
```

## 数据与配置管理
- `configs/` 保存数据源与实验设置。`configs/data-initial.yaml` 是训练的默认配置；`configs/data.yaml` 指向 Roboflow 项目 `robocon-ozkss/base-inspection-txwpc`。如需新增数据，只需复制模板并修改路径。
- 数据目录命名建议与配置保持一致，例如 `datasets-initial/`、`datasets/roboflow_v1/` 等，并在目录下放置 `train/valid/test` 子文件夹以匹配 YAML 声明。
- 仓库已在 `.gitignore` 中排除 `datasets/` 与 `artifacts/`，提交前可快速执行 `git status` 确认未包含大文件。

## 手动测试清单
- `uv run scripts/predict.py`，检查 `artifacts/outputs/output.avi` 是否更新，确保视频帧内有检测框与轨迹。
- 针对每个 `test/` 下的演示视频重复上述命令，确认兼容不同分辨率。
- 训练逻辑调整后，记录 `artifacts/runs/<run_name>/results.csv` 最新指标；必要时截取 loss/precision 曲线附在 PR 中。
- 若修改了 CLI 参数解析或路径逻辑，使用 `uv run --module basedetect` 做一次冒烟，验证目录与默认文件都能正确创建。

## 贡献与协作
- 提交信息遵循 Conventional Commits，例如 `feat: add thermal preprocessor`。
- PR 中附上操作命令、复现步骤以及关键信息（模型配置、训练轮数、评估指标）。
- 数据访问或架构讨论请在 issue 中同步，避免口头约定难以追踪。

## 常见问题
- **脚本退回 CPU**：检查 `CUDA_VISIBLE_DEVICES` 是否被设置为空，或 `nvidia-smi` 是否能读取目标 GPU。必要时显式传入 `--device 0`。
- **权重路径找不到**：训练前确认 `weights/pretrained/yolov8n.pt` 存在；若复制到别处，可通过 `--weights` 手动指定。
- **输出目录不存在**：运行推理脚本前确保执行过 `uv run --module basedetect` 或手动创建 `artifacts/outputs/`。
- **Roboflow 标签不匹配**：导出 YOLOv8 格式时保持默认类别顺序，并在 `configs/*.yaml` 中同步 `names` 字段。

## 资源
- 英文文档：README.md
- 贡献者指南：AGENTS.md
