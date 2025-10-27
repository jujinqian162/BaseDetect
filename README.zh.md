# BaseDetect 项目指南

BaseDetect 针对机床底座检测与跟踪场景封装了基于 Ultralytics YOLOv8 的训练与推理流程。仓库内提供最小化数据集、脚本工具、预训练权重以及示例视频，帮助快速验证流程并在自有数据上迭代。

## 项目结构
```
BaseDetect/
├─ basedetect/             # Python 包，可 `uv run --module basedetect` 快速冒烟
├─ scripts/                # 训练与推理脚本（train.py、predict.py）
├─ configs/                # 数据/任务配置（data.yaml 等）
├─ datasets/               # 数据集汇总目录
│  ├─ datasets2/           # configs/data.yaml 指向的 Roboflow 导出
│  ├─ datasets-initial/    # configs/data-initial.yaml 默认使用的数据
│  └─ demo/                # `--config auto` 自动生成的演示数据
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
2. 确保 `configs/data-initial.yaml` 对应的 `datasets/datasets-initial/` 数据集已准备完毕（参见“数据与配置管理”）。
3. `uv run scripts/train.py` 使用 `configs/data-initial.yaml` 进行训练，确认 `artifacts/runs/basedetect/` 写出日志与 `weights/best.pt`。
4. `uv run scripts/predict.py` 跑通 `test/test3.mp4`，验证在 `artifacts/outputs/output.avi` 生成带框视频。

如果需要临时体验 demo 数据集，可运行 `uv run --module basedetect` 生成 `datasets/demo/`，并在训练命令中添加 `--config auto`。

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
- 当 `--weights auto` 未找到最新训练权重时，会打印黄色的中英双语警告并回退到内置预训练模型，确保你了解当前使用的权重来源。

常见用法：
```bash
# 推理摄像头画面并弹窗查看
uv run scripts/predict.py --source 0 --show

# 指定历史实验的 best 权重，生成离线视频
uv run scripts/predict.py --weights artifacts/runs/basedetect/weights/best.pt --source test/test3.mp4
```

## 数据与配置管理
- `configs/` 保存数据源与实验设置。默认使用的 `configs/data-initial.yaml` 指向 `datasets/datasets-initial/`；`configs/data.yaml` 对应 `datasets/datasets2/` 下的 Roboflow 数据。复制模板即可衍生新配置。
- 所有数据集统一存放在 `datasets/` 内，可按需创建如 `datasets/custom_v1/` 的目录，并确保其下包含 `train/valid/test` 子目录，以便 YAML 路径正确指向。
- 演示数据位于 `datasets/demo/`，执行 `uv run --module basedetect` 可随时重新生成。
- 提交前请检查 `git status`，避免意外提交体积较大的数据或产物。

## 手动测试清单
- `uv run scripts/predict.py`，检查 `artifacts/outputs/output.avi` 是否更新并包含检测框，同时仅在自动回退预训练权重时出现黄色中英双语警告。
- 针对每个 `test/` 下的演示视频重复上述命令，确认兼容不同分辨率。
- 训练逻辑调整后，记录 `artifacts/runs/<run_name>/results.csv` 最新指标；必要时截取 loss/precision 曲线附在 PR 中。
- 若修改了 CLI 参数解析或路径逻辑，使用 `uv run scripts/test_cli.py` 做参数冒烟，并运行 `uv run --module basedetect` 确认演示数据仍可生成。

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
