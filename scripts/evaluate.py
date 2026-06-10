#!/usr/bin/env python
"""
Evaluate the calibrated classifier on the fixed test split.

Produces metrics/evaluation_report.json with Macro F1, per-class F1,
accuracy, ECE, and sample count — the inputs the validate_model gate reads.

Usage (experiment directory — requires config.yaml alongside weights):
  python scripts/evaluate.py --classifier-dir experiments/binary_classifier/efficientnet_b1

Usage (direct weights file — arch read from checkpoint; pass --arch for pre-Plan-A weights):
  python scripts/evaluate.py --weights src/allium_cepa_classifier/weights/classifier_calibrated.pt
  python scripts/evaluate.py --weights <path>.pt --arch efficientnet_b2

Test set source: datasets/crops/binary_classifier/test (ImageFolder).
Future (Plan D): swap for s3://dataset/test_fixed/ via the storage provider.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from allium_cepa_classifier.config.experiment_config import ExperimentConfig, ModelConfig
from allium_cepa_classifier.training.model_builder import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CLASSIFIER_DIR = _ROOT / "experiments/binary_classifier/efficientnet_b2"
_DEFAULT_TEST_DIR = _ROOT / "datasets/crops/binary_classifier/test"


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.sum() / len(labels) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece_val)


def load_calibrated_classifier(
    weights_path: Path,
    device: torch.device,
    arch: str | None = None,
) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    """Load calibrated BackboneWithHead + temperature from a .pt file.

    arch is read from the checkpoint when present (stored by calibrator.py >= Plan A).
    Pass arch explicitly for older weights that pre-date this metadata field.
    """
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    resolved_arch = arch or ckpt.get("timm_model_name")
    if resolved_arch is None:
        raise ValueError(
            f"Checkpoint {weights_path} does not contain 'timm_model_name'. "
            "Pass --arch <arch> explicitly (e.g. --arch efficientnet_b2)."
        )

    model = build_model(ModelConfig(arch=resolved_arch)).to(device)

    # CalibratedClassifier wraps BackboneWithHead, so state dict keys are base_model.*
    base_state = {
        k[len("base_model.") :]: v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("base_model.")
    }
    model.load_state_dict(base_state)
    model.eval()

    temperature = torch.tensor(ckpt["temperature"], dtype=torch.float32).to(device)
    return model, temperature, ckpt


def run_evaluation(weights_path: Path, test_dir: Path, arch: str | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model, temperature, ckpt = load_calibrated_classifier(weights_path, device, arch=arch)

    image_size = tuple(ckpt.get("image_size", (224, 224)))
    mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    log.info(f"Test samples: {len(test_ds)}, class mapping: {test_ds.class_to_idx}")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    num_classes = probs.shape[1]
    class_names = [idx_to_class[i] for i in range(num_classes)]

    f1_per_class = f1_score(labels, preds, average=None)
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    acc = float(accuracy_score(labels, preds))
    ece = _ece(probs, labels)

    log.info(f"Macro F1={macro_f1:.4f}  Accuracy={acc:.4f}  ECE={ece:.4f}")
    for i, name in enumerate(class_names):
        log.info(f"  F1[{name}]={f1_per_class[i]:.4f}")

    return {
        "macro_f1": macro_f1,
        "f1_per_class": {name: float(f1_per_class[i]) for i, name in enumerate(class_names)},
        "accuracy": acc,
        "ece": ece,
        "n_samples": int(len(labels)),
    }


def _resolve_weights(args: argparse.Namespace) -> tuple[Path, str | None]:
    """Return (weights_path, arch_hint) from parsed args."""
    if args.weights:
        return Path(args.weights), args.arch

    classifier_dir = Path(args.classifier_dir)
    weights = classifier_dir / "weights" / "classifier_calibrated.pt"
    # Try to read arch from config.yaml if present (pre-Plan-A checkpoint without timm_model_name)
    config_path = classifier_dir / "config.yaml"
    arch_hint = args.arch
    if arch_hint is None and config_path.exists():
        try:
            cfg = ExperimentConfig.from_yaml(config_path)
            arch_hint = cfg.model.arch
        except Exception:
            pass
    return weights, arch_hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated classifier on test split.")

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--classifier-dir",
        type=Path,
        default=_DEFAULT_CLASSIFIER_DIR,
        help="Experiment directory containing weights/classifier_calibrated.pt (and optionally config.yaml)",
    )
    source.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Direct path to a classifier_calibrated.pt file",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override timm model arch (e.g. efficientnet_b2). Required for pre-Plan-A weights "
        "that do not embed 'timm_model_name' in the checkpoint.",
    )
    parser.add_argument("--test-dir", type=Path, default=_DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights_path, arch = _resolve_weights(args)
    report = run_evaluation(weights_path, args.test_dir, arch=arch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    log.info(f"Written: {args.output}")


if __name__ == "__main__":
    main()
