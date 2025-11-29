脚本会在 checkpoints_multimodal/ 中创建检查点（checkpoints）。要评估多模态融合模型，运行：

python scripts/run_pipeline.py eval-mm --checkpoint checkpoints_multimodal/multimodal_epoch_005.pt --image-root /Users/.../val --text-csv data/soy_disease_text_val.csv --tokenizer bert-base-uncased --save results/multimodal_confusion.png

4. Windows 相关说明

在 Windows PowerShell 中，路径里的正斜杠需要替换为转义的反斜杠，或者直接把路径用引号括起来。激活虚拟环境的命令为：

.venv\\Scripts\\Activate.ps1


在安装 PyTorch 和 mamba-ssm 之前，请确保已安装最新版本的 Visual C++ 运行库，以及（如果适用）最新的 GPU 驱动程序。

5. macOS 相关说明

在 macOS（无论 Intel 还是 Apple Silicon）上，默认的 pip install -r requirements.txt 会安装 CPU 版本的 wheel。
如果你在 Apple Silicon 上想用 GPU（mps），请确保系统是 macOS 12.3 或更高版本；如果运行时遇到算子缺失错误，可以在运行脚本时设置：

PYTORCH_ENABLE_MPS_FALLBACK=1

扩展该训练流水线

数据增广（Data augmentation）：在 detr_mamaba/cv/augmentation.py 中添加更多图像增强操作（例如 CutMix、随机裁剪等），并在构建 SoyDiseaseDetectionDataset 时传入这些增强。

超参数调优（Hyperparameter tuning）：通过修改对应的 dataclass 配置来调整学习率、batch size 以及学习率调度策略等。

知识集成（Knowledge integration）：在 NLP 使用的 CSV 文件中加入更多经过整理的问答对，内容包括症状描述、治疗方式和预防措施，让多模态融合模块能够给出更丰富、更专业的回答。

AI 驱动的局域网（LAN）聊天练习

为了练习提示中提到的 AI 驱动聊天应用流程，仓库在 scripts/lan_chat_app.py 中提供了一个基于 asyncio 的轻量级局域网聊天服务器与客户端。该应用支持局域网内的多人聊天，并提供一个 @电影 触发词来返回电影推荐（它还会顺便“艾特”同事，比如 @川小农，让 TA 一起来聊天）。

在局域网中启动服务器：

python scripts/lan_chat_app.py server --host 0.0.0.0 --port 9009 --ai-name 助手


在同一网段内启动客户端连接：

python scripts/lan_chat_app.py client --host <server-ip> --port 9009 --name 川小农


键入消息并回车即可广播给所有在线客户端。
在消息中提到 @电影 可以获得 AI 精选的电影推荐，也可以 “@助手”（或你设置的 AI 名字）以获得及时响应。该示例没有额外依赖，可以与主训练流水线同时运行，用于快速实验和练习。
