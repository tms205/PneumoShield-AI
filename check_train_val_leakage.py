"""Check whether validation images leak into the training split.

This script checks:

1. Reused patient IDs between train.csv and val.csv, when manifests exist.
2. Reused image file names between the train and validation folders.
3. Exact duplicate image contents using SHA-256, even if files were renamed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_EXAMPLES = 10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    return sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_patient_ids(manifest_path: Path) -> set[str] | None:
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "patient_id" not in reader.fieldnames:
            raise ValueError(f"Column 'patient_id' not found in: {manifest_path}")
        return {row["patient_id"] for row in reader if row["patient_id"]}


def duplicate_hash_pairs(
    train_images: list[Path], val_images: list[Path]
) -> list[tuple[Path, Path, str]]:
    train_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in train_images:
        train_by_hash[file_sha256(path)].append(path)

    duplicates = []
    for val_path in val_images:
        sha256 = file_sha256(val_path)
        for train_path in train_by_hash.get(sha256, []):
            duplicates.append((train_path, val_path, sha256))
    return duplicates


def print_examples(title: str, examples: list[str]) -> None:
    if not examples:
        return
    print(f"\n{title} (showing up to {MAX_EXAMPLES}):")
    for example in examples[:MAX_EXAMPLES]:
        print(f"  - {example}")


def check_train_val_leakage(dataset_dir: Path) -> bool:
    train_images = find_images(dataset_dir / "train")
    val_images = find_images(dataset_dir / "val")

    train_patient_ids = read_patient_ids(dataset_dir / "train.csv")
    val_patient_ids = read_patient_ids(dataset_dir / "val.csv")
    overlapping_patient_ids: set[str] = set()
    if train_patient_ids is not None and val_patient_ids is not None:
        overlapping_patient_ids = train_patient_ids & val_patient_ids

    train_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in train_images:
        train_by_name[path.name].append(path)
    overlapping_names = sorted(
        name for name in {path.name for path in val_images} if name in train_by_name
    )

    duplicate_contents = duplicate_hash_pairs(train_images, val_images)

    print(f"Dataset: {dataset_dir.resolve()}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")
    if train_patient_ids is not None and val_patient_ids is not None:
        print(f"Patient IDs appearing in both train and val: {len(overlapping_patient_ids)}")
    else:
        print("Patient ID check: skipped (train.csv or val.csv not found)")
    print(f"File names appearing in both train and val: {len(overlapping_names)}")
    print(f"Exact image duplicates across train and val: {len(duplicate_contents)}")

    print_examples("Overlapping patient IDs", sorted(overlapping_patient_ids))
    print_examples("Overlapping file names", overlapping_names)
    print_examples(
        "Exact duplicate files",
        [
            f"{train_path.relative_to(dataset_dir)} == "
            f"{val_path.relative_to(dataset_dir)} (sha256={sha256})"
            for train_path, val_path, sha256 in duplicate_contents
        ],
    )

    leaked = bool(overlapping_patient_ids or overlapping_names or duplicate_contents)
    if leaked:
        print("\nRESULT: LEAKAGE FOUND between train and val.")
    else:
        print("\nRESULT: No exact train/val leakage found.")
    return leaked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check patient IDs, filenames, and exact image duplicates across train/val."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_clean_70_15_15"),
        help="Dataset directory containing train/ and val/ folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    leakage_found = check_train_val_leakage(args.dataset_dir)
    raise SystemExit(1 if leakage_found else 0)
