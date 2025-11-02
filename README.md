# Soybean Disease Multimodal Pipeline

This repository provides a complete computer-vision and natural-language-processing pipeline to detect soybean leaf diseases and answer agronomic questions about the findings. The framework combines a DETR detector (forward pass based on `torchvision`'s implementation) with a Mamba-based text classifier and a fusion model for multimodal question answering.

## Repository structure

```
src/
  detr_mamaba/
    cv/                # DETR training, evaluation, augmentation helpers
    nlp/               # Mamba text dataset utilities and training
    multimodal/        # Fusion dataset, model, training, evaluation
scripts/
  run_pipeline.py      # Command line interface for all stages
requirements.txt       # Core dependencies (CPU/GPU compatible)
```

## Prerequisites

1. **Python**: 3.9 or newer.
2. **PyTorch**: install the variant that matches your CUDA toolkit if you plan to use GPUs. Follow the official [installation instructions](https://pytorch.org/get-started/locally/).
3. **Mamba SSM**: the NLP model relies on the `mamba-ssm` package. Prebuilt wheels are available for macOS (Apple Silicon and Intel) and Windows.

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU support on Windows, run the PyTorch installation command suggested on pytorch.org before installing the rest of the requirements to ensure compatible CUDA packages are selected.

## Dataset layout

The code expects the soybean detection dataset to follow the structure described in the prompt:

```
大豆叶片病害数据集/
  train/
    images/   # training images
    labels/   # YOLO-format label files
  val/
    images/
    labels/
  test/
    images/
    labels/
```

The NLP CSV files should contain two columns (`text`, `label`) describing the disease categories: `Healthy`, `Bean_Rust`, `Angular_Leaf_Spot`.

## Training and evaluation

All workflows are orchestrated by `scripts/run_pipeline.py`.

### 1. Train DETR on soybean imagery

```bash
python scripts/run_pipeline.py train-cv --train-root /Users/.../train --val-root /Users/.../val --epochs 100 --batch-size 4
```

Checkpoints are written to `checkpoints/`. During validation, mean average precision (when `torchmetrics` is installed) and per-epoch loss values are recorded. After training, evaluate and produce plots:

```bash
python scripts/run_pipeline.py eval-cv --checkpoint checkpoints/detr_epoch_100.pt --data-root /Users/.../test --save results/cv_confusion.png
```

The evaluation routine emits aggregate precision/recall scores and renders the confusion matrix using Seaborn.

### 2. Prepare textual data and train the Mamba classifier

Collect short agronomic descriptions for each disease type (official agricultural bulletins, extension service documents, etc.), save them into a CSV file, and fine-tune the Mamba classifier:

```bash
python scripts/run_pipeline.py train-nlp --train-csv data/soy_disease_text_train.csv --val-csv data/soy_disease_text_val.csv --tokenizer bert-base-uncased
```

This command stores NLP checkpoints under `checkpoints_nlp/`. Accuracy and validation loss are reported every epoch.

### 3. Train the multimodal QA model

The multimodal dataset pairs each annotated image with textual snippets of the corresponding disease label. Train the fusion model as follows:

```bash
python scripts/run_pipeline.py train-mm --image-root /Users/.../train --text-csv data/soy_disease_text_train.csv --tokenizer bert-base-uncased
```

The script creates checkpoints in `checkpoints_multimodal/`. To evaluate the fused model and export diagnostic plots:

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

## License

This repository is provided as-is for research and educational purposes.
