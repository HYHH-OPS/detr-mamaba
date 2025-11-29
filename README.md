# 大豆病害多模态管线

该仓库提供一套完整的计算机视觉与自然语言处理流程，用于识别大豆叶片病害并回答相关农艺问题。框架结合了 DETR 目标检测器（基于 `torchvision` 的前向实现）、Mamba 文本分类器以及多模态融合模型。

## 仓库结构

```
src/
  detr_mamaba/
    cv/                # DETR 训练、评估与数据增强工具
    nlp/               # Mamba 文本数据集与训练脚本
    multimodal/        # 融合数据集、模型、训练与评估
scripts/
  run_pipeline.py      # 统一的命令行入口
requirements.txt       # 核心依赖（CPU/GPU 兼容）
```

## 环境准备

1. **Python**：3.9 及以上。
2. **PyTorch**：如需使用 GPU，请按照官方 [安装指南](https://pytorch.org/get-started/locally/) 安装与本地 CUDA 对应的版本。
3. **Mamba SSM**：NLP 模型依赖 `mamba-ssm` 包，macOS（Intel/Apple Silicon）与 Windows 均提供预编译 wheel。

安装依赖示例：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

如需 Windows GPU 支持，请先按照 pytorch.org 提示安装匹配 CUDA 的 PyTorch，再安装其余依赖。

## 数据集组织

DETR 训练数据的默认目录结构：

```
大豆叶片病害数据集/
  train/
    images/   # 训练图片
    labels/   # YOLO 格式标注
  val/
    images/
    labels/
  test/
    images/
    labels/
```

NLP 文本 CSV 应包含两列（`text`, `label`），标签可为 `Healthy`、`Bean_Rust`、`Angular_Leaf_Spot`。

## 训练与评估流程

所有流程均通过 `scripts/run_pipeline.py` 调度。

### 1. 训练 DETR 视觉模型

```bash
python scripts/run_pipeline.py train-cv --train-root /Users/.../train --val-root /Users/.../val --epochs 100 --batch-size 4
```

检查点默认保存在 `checkpoints/`。验证阶段会记录 mAP（安装 `torchmetrics` 时可用）与各 epoch 的 loss。训练完成后可评估并绘图：

```bash
python scripts/run_pipeline.py eval-cv --checkpoint checkpoints/detr_epoch_100.pt --data-root /Users/.../test --save results/cv_confusion.png
```

### 2. 准备文本数据并训练 Mamba 分类器

收集各病害类别的简短描述，保存在 CSV 后即可微调分类器：

```bash
python scripts/run_pipeline.py train-nlp --train-csv data/soy_disease_text_train.csv --val-csv data/soy_disease_text_val.csv --tokenizer bert-base-uncased
```

检查点会存于 `checkpoints_nlp/`，训练日志会输出精度与验证损失。

### 3. 训练多模态问答模型

多模态数据集将图片与相应文本标签配对，训练命令示例：

```bash
python scripts/run_pipeline.py train-mm --image-root /Users/.../train --text-csv data/soy_disease_text_train.csv --tokenizer bert-base-uncased
```

模型保存在 `checkpoints_multimodal/`，评估与绘图示例：

```bash
python scripts/run_pipeline.py eval-mm --checkpoint checkpoints_multimodal/multimodal_epoch_005.pt --image-root /Users/.../val --text-csv data/soy_disease_text_val.csv --tokenizer bert-base-uncased --save results/multimodal_confusion.png
```

### 4. Windows 注意事项

在 PowerShell 中可使用：

```powershell
.venv\Scripts\Activate.ps1
```

请确保安装 Visual C++ 运行时并更新 GPU 驱动（如使用 CUDA）。路径中的 `/` 可改为转义的 `\` 或用引号包裹。

### 5. macOS 注意事项

默认 `pip install -r requirements.txt` 会安装 CPU 版本。如需 Apple Silicon 的 `mps`，确保系统为 macOS 12.3+，并在出现算子缺失时尝试：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```

## 扩展建议

- **数据增强**：在 `detr_mamaba/cv/augmentation.py` 添加 CutMix、随机裁剪等变换，并传入 `SoyDiseaseDetectionDataset`。
- **超参数调优**：通过对应的 dataclass 配置学习率、批大小与调度策略。
- **知识扩展**：在 NLP CSV 中补充症状、治疗、预防等问答对，让融合模型生成更丰富的回答。

## AI 驱动局域网聊天练习

`scripts/lan_chat_app.py` 提供轻量的 asyncio 聊天示例，支持局域网对话与 `@电影` 触发，亦可提醒队友（如 `@川小农`）。

### 启动服务器

```bash
python scripts/lan_chat_app.py server --host 0.0.0.0 --port 9009 --ai-name 助手
```

可通过 `--movies` 传入逗号分隔的推荐片单，或使用 `--room` 为聊天室命名。

### 加入聊天

```bash
python scripts/lan_chat_app.py client --host <server-ip> --port 9009 --name 川小农
```

输入消息并回车即可广播。提及 `@电影` 可获得电影推荐，@AI 昵称或 `@助手` 会得到即时回复，`@帮助` 可查看可用指令。示例无外部依赖，可与主训练流程并行试验。

## License

本仓库仅用于研究与教学目的，按现状提供。
