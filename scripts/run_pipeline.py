"""Command line interface for the soybean disease multimodal pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from detr_mamaba.cv.train import TrainingConfig, train_detr
from detr_mamaba.cv.evaluation import evaluate_model, plot_confusion_matrix
from detr_mamaba.cv.model import DetrConfig
from detr_mamaba.nlp.train import NLPTrainingConfig, train_mamba_classifier
from detr_mamaba.nlp.model import MambaConfig
from detr_mamaba.multimodal.model import MultimodalConfig
from detr_mamaba.multimodal.train import MultimodalTrainingConfig, train_multimodal_model
from detr_mamaba.multimodal.evaluation import evaluate_multimodal_model
from detr_mamaba.multimodal.data import multimodal_collate_fn, MultimodalSoyDataset
from detr_mamaba.nlp.data import LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soybean disease detection multimodal pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cv_parser = subparsers.add_parser("train-cv", help="Train the DETR detector")
    cv_parser.add_argument("--train-root", type=Path, required=True)
    cv_parser.add_argument("--val-root", type=Path)
    cv_parser.add_argument("--epochs", type=int, default=50)
    cv_parser.add_argument("--batch-size", type=int, default=2)

    eval_parser = subparsers.add_parser("eval-cv", help="Evaluate a DETR checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data-root", type=Path, required=True)
    eval_parser.add_argument("--save", type=Path)

    nlp_parser = subparsers.add_parser("train-nlp", help="Train the Mamba text classifier")
    nlp_parser.add_argument("--train-csv", type=Path, required=True)
    nlp_parser.add_argument("--val-csv", type=Path)
    nlp_parser.add_argument("--tokenizer", type=str, required=True)

    mm_parser = subparsers.add_parser("train-mm", help="Train the multimodal fusion model")
    mm_parser.add_argument("--image-root", type=Path, required=True)
    mm_parser.add_argument("--text-csv", type=Path, required=True)
    mm_parser.add_argument("--tokenizer", type=str, required=True)
    mm_parser.add_argument("--detr-checkpoint", type=Path)
    mm_parser.add_argument("--mamba-checkpoint", type=Path)

    mm_eval_parser = subparsers.add_parser("eval-mm", help="Evaluate the multimodal model")
    mm_eval_parser.add_argument("--checkpoint", type=Path, required=True)
    mm_eval_parser.add_argument("--image-root", type=Path, required=True)
    mm_eval_parser.add_argument("--text-csv", type=Path, required=True)
    mm_eval_parser.add_argument("--tokenizer", type=str, required=True)
    mm_eval_parser.add_argument("--save", type=Path)

    return parser.parse_args()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def main() -> None:
    args = parse_args()

    if args.command == "train-cv":
        config = TrainingConfig(train_root=args.train_root, val_root=args.val_root, epochs=args.epochs, batch_size=args.batch_size)
        train_detr(config)
    elif args.command == "eval-cv":
        result = evaluate_model(args.checkpoint, args.data_root, class_names=LABELS)
        if args.save:
            plot_confusion_matrix(result, save_path=args.save)
        else:
            plot_confusion_matrix(result)
    elif args.command == "train-nlp":
        tokenizer = load_tokenizer(args.tokenizer)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(tokenizer.get_vocab())
        config = NLPTrainingConfig(train_csv=args.train_csv, val_csv=args.val_csv, tokenizer=tokenizer)
        model_config = MambaConfig(vocab_size=vocab_size, num_classes=len(LABELS))
        train_mamba_classifier(config, model_config)
    elif args.command == "train-mm":
        tokenizer = load_tokenizer(args.tokenizer)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(tokenizer.get_vocab())
        detr_config = DetrConfig(num_classes=len(LABELS) + 1)
        mamba_config = MambaConfig(vocab_size=vocab_size, num_classes=len(LABELS))
        multi_config = MultimodalConfig(detr_config=detr_config, mamba_config=mamba_config)
        mm_config = MultimodalTrainingConfig(
            image_root=args.image_root,
            text_csv=args.text_csv,
            tokenizer=tokenizer,
            detr_checkpoint=args.detr_checkpoint,
            mamba_checkpoint=args.mamba_checkpoint,
        )
        train_multimodal_model(mm_config, multi_config)
    elif args.command == "eval-mm":
        tokenizer = load_tokenizer(args.tokenizer)
        dataset = MultimodalSoyDataset(args.image_root, args.text_csv, tokenizer=tokenizer)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=multimodal_collate_fn)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint["config"]
        model = MultimodalFusionModel(config)
        model.load_state_dict(checkpoint["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        evaluate_multimodal_model(model, loader, answer_space=LABELS, device=device, save_heatmap=args.save)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
