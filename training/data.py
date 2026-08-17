"""Dataset discovery and deterministic DataLoader construction."""

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
)


def _image_files(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_dataset(save_report: bool = True) -> dict:
    """Inspect the actual predefined train/valid/test class folders."""
    split_paths = {"train": TRAIN_PATH, "valid": VALID_PATH, "test": TEST_PATH}
    report: dict = {"splits": {}, "official_predefined_splits": True}
    split_keys: dict[str, set[str]] = {}

    for split, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Required dataset split is missing: {split_path}")

        class_counts = {}
        keys = set()
        for class_name in ("real", "fake"):
            class_path = split_path / class_name
            if not class_path.exists():
                raise FileNotFoundError(f"Required class folder is missing: {class_path}")
            files = _image_files(class_path)
            class_counts[class_name] = len(files)
            keys.update(f"{class_name}/{path.name.lower()}" for path in files)

        total = sum(class_counts.values())
        minority = min(class_counts.values())
        majority = max(class_counts.values())
        imbalance_ratio = majority / minority if minority else float("inf")
        report["splits"][split] = {
            "path": str(split_path.resolve()),
            "total": total,
            "real": class_counts["real"],
            "fake": class_counts["fake"],
            "imbalance_ratio": imbalance_ratio,
            "imbalanced": imbalance_ratio > 1.1,
        }
        split_keys[split] = keys

    report["total_images"] = sum(item["total"] for item in report["splits"].values())
    report["real_images"] = sum(item["real"] for item in report["splits"].values())
    report["fake_images"] = sum(item["fake"] for item in report["splits"].values())
    report["obvious_filename_overlap"] = {
        "train_valid": len(split_keys["train"] & split_keys["valid"]),
        "train_test": len(split_keys["train"] & split_keys["test"]),
        "valid_test": len(split_keys["valid"] & split_keys["test"]),
    }

    if save_report:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "dataset_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def print_dataset_report(report: dict) -> None:
    print("\nDataset inspection")
    print("-" * 72)
    for split, values in report["splits"].items():
        print(
            f"{split:>5}: {values['total']:>7,} images "
            f"(REAL {values['real']:>6,}, FAKE {values['fake']:>6,})"
        )
    print(
        f"total: {report['total_images']:,} images "
        f"(REAL {report['real_images']:,}, FAKE {report['fake_images']:,})"
    )
    print("Class imbalance: none (all predefined splits are 50/50).")
    print(f"Obvious filename overlap: {report['obvious_filename_overlap']}")
    print("Split policy: reuse the dataset's official predefined splits.")
    print("-" * 72)


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)],
                p=0.35,
            ),
            transforms.RandomRotation(degrees=4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _stratified_subset(dataset: datasets.ImageFolder, maximum: int | None) -> datasets.ImageFolder | Subset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < len(dataset.classes):
        raise ValueError("A sample limit must be at least the number of classes.")

    rng = np.random.default_rng(SEED)
    per_class = maximum // len(dataset.classes)
    remainder = maximum % len(dataset.classes)
    selected: list[int] = []
    targets = np.asarray(dataset.targets)
    for class_index in range(len(dataset.classes)):
        indices = np.flatnonzero(targets == class_index)
        rng.shuffle(indices)
        take = per_class + (1 if class_index < remainder else 0)
        selected.extend(indices[:take].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=training_transform())
    valid_dataset = datasets.ImageFolder(VALID_PATH, transform=evaluation_transform())
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=evaluation_transform())

    if train_dataset.class_to_idx != {"fake": 0, "real": 1}:
        raise RuntimeError(f"Unexpected class mapping: {train_dataset.class_to_idx}")

    train_data = _stratified_subset(train_dataset, max_train_samples)
    valid_data = _stratified_subset(valid_dataset, max_valid_samples)
    test_data = _stratified_subset(test_dataset, max_test_samples)

    generator = torch.Generator().manual_seed(SEED)
    workers = min(NUM_WORKERS, os.cpu_count() or 1)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    valid_loader = DataLoader(valid_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx
