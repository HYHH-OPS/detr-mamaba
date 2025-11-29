@@ -91,28 +91,46 @@ The script creates checkpoints in `checkpoints_multimodal/`. To evaluate the fus

```bash
python scripts/run_pipeline.py eval-mm --checkpoint checkpoints_multimodal/multimodal_epoch_005.pt --image-root /Users/.../val --text-csv data/soy_disease_text_val.csv --tokenizer bert-base-uncased --save results/multimodal_confusion.png
```

### 4. Windows-specific notes

On Windows PowerShell, replace forward slashes in paths with escaped backslashes or wrap the paths in quotes. Activate the virtual environment with:

```powershell
.venv\\Scripts\\Activate.ps1
```

Ensure that the Visual C++ runtime and GPU drivers (if applicable) are up to date before installing PyTorch and `mamba-ssm`.

### 5. macOS-specific notes

On macOS (Intel or Apple Silicon), the default `pip install -r requirements.txt` command will install CPU wheels. For Apple Silicon GPUs via `mps`, ensure you are on macOS 12.3 or newer and run your scripts with `PYTORCH_ENABLE_MPS_FALLBACK=1` if you experience missing operator errors.

## Extending the pipeline

- **Data augmentation**: extend `detr_mamaba/cv/augmentation.py` with additional transformations (CutMix, random crops) and pass them into `SoyDiseaseDetectionDataset`.
- **Hyperparameter tuning**: update the dataclass configurations for learning rate, batch sizes, and scheduling strategies.
- **Knowledge integration**: expand the NLP CSV files with curated Q&A pairs describing symptoms, treatments, and prevention methods so that the multimodal fusion module can produce richer answers.

## AI-driven LAN chat exercise

To practice the AI-driven chat application flow mentioned in the prompt, a lightweight asyncio chat server and client are available in `scripts/lan_chat_app.py`. The app supports LAN conversations and an `@电影` trigger that returns movie suggestions (it also nudges teammates like `@川小农` to join the conversation).

Start a server on your LAN:

```bash
python scripts/lan_chat_app.py server --host 0.0.0.0 --port 9009 --ai-name 助手
```

Connect a client from the same network segment:

```bash
python scripts/lan_chat_app.py client --host <server-ip> --port 9009 --name 川小农
```

Type messages and press Enter to broadcast. Mention `@电影` to receive AI-curated recommendations, or ping `@助手` (or the AI name you set) to get a prompt response. The example stays dependency-free so it can be run alongside the main training pipeline for quick experimentation.

## License

This repository is provided as-is for research and educational purposes.
