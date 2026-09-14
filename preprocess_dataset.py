"""Prepare the RSNA DICOM dataset for binary NORMAL/PNEUMONIA training.

The raw RSNA training set contains three clinical categories. This pipeline
keeps only:

    Normal       -> NORMAL
    Lung Opacity -> PNEUMONIA

Images labelled "No Lung Opacity / Not Normal" are excluded because they are
not normal controls for a binary classification experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pydicom is required to read .dcm images. Install it with: "
        "python -m pip install pydicom"
    ) from exc


CLASS_MAP = {
    "Normal": "NORMAL",
    "Lung Opacity": "PNEUMONIA",
}
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
OUTPUT_CLASSES = ("NORMAL", "PNEUMONIA")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_binary_labels(labels_path: Path) -> list[dict[str, str]]:
    """Read unique labelled patient images and retain the two target classes."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    patients: dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            patient_id = row["patientId"]
            source_class = row["class"]
            previous = patients.setdefault(patient_id, source_class)
            if previous != source_class:
                raise ValueError(
                    f"Conflicting classes for patient {patient_id}: "
                    f"{previous!r} and {source_class!r}"
                )

    records = [
        {"patient_id": patient_id, "class": CLASS_MAP[source_class]}
        for patient_id, source_class in patients.items()
        if source_class in CLASS_MAP
    ]
    return records


def check_source_duplicates(
    records: list[dict[str, str]], images_dir: Path
) -> None:
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        source_path = images_dir / f"{record['patient_id']}.dcm"
        if not source_path.exists():
            raise FileNotFoundError(f"DICOM image not found: {source_path}")
        hashes[file_sha256(source_path)].append(record["patient_id"])

    duplicates = {
        sha256: patient_ids
        for sha256, patient_ids in hashes.items()
        if len(patient_ids) > 1
    }
    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} exact duplicate source images among "
            "retained labels. Resolve duplicates before splitting."
        )


def stratified_split(
    records: list[dict[str, str]], seed: int
) -> dict[str, list[dict[str, str]]]:
    """Split each class separately so all sets retain similar class ratios."""
    rng = random.Random(seed)
    by_class = {class_name: [] for class_name in OUTPUT_CLASSES}
    for record in records:
        by_class[record["class"]].append(record)

    splits = {split: [] for split in SPLIT_RATIOS}
    for class_name in OUTPUT_CLASSES:
        class_records = sorted(
            by_class[class_name], key=lambda record: record["patient_id"]
        )
        rng.shuffle(class_records)

        total = len(class_records)
        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])
        splits["train"].extend(class_records[:train_end])
        splits["val"].extend(class_records[train_end:val_end])
        splits["test"].extend(class_records[val_end:])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def dicom_to_grayscale_image(dicom_path: Path) -> Image.Image:
    """Convert one DICOM X-ray to a displayable 8-bit grayscale PIL image."""
    dataset = pydicom.dcmread(str(dicom_path))
    pixels = dataset.pixel_array.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if maximum <= minimum:
        normalized = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        normalized = ((pixels - minimum) / (maximum - minimum) * 255.0).astype(
            np.uint8
        )

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        normalized = 255 - normalized

    return Image.fromarray(normalized, mode="L")


def initialize_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    for split in SPLIT_RATIOS:
        for class_name in OUTPUT_CLASSES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir: Path, split: str, rows: list[dict[str, str]]) -> None:
    columns = ["patient_id", "class", "source_path", "image_path", "split", "sha256"]
    with (output_dir / f"{split}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_prepared_dataset(
    output_dir: Path, manifest_rows: list[dict[str, str]]
) -> None:
    expected_paths = {output_dir / Path(row["image_path"]) for row in manifest_rows}
    actual_paths = {
        path
        for split in SPLIT_RATIOS
        for class_name in OUTPUT_CLASSES
        for path in (output_dir / split / class_name).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if expected_paths != actual_paths:
        raise ValueError(
            "Prepared dataset images do not match the generated manifest."
        )

    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_classes: dict[str, set[str]] = defaultdict(set)
    image_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        image_path = output_dir / Path(row["image_path"])
        with Image.open(image_path) as image:
            image.verify()
        sha256 = file_sha256(image_path)
        if row["sha256"] != sha256:
            raise ValueError(f"SHA-256 validation failed for: {image_path}")
        patient_splits[row["patient_id"]].add(row["split"])
        patient_classes[row["patient_id"]].add(row["class"])
        image_hashes[sha256].append(row)

    crossing_patients = [
        patient_id
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    ]
    if crossing_patients:
        raise ValueError(
            f"Found {len(crossing_patients)} patient IDs across multiple splits."
        )
    class_conflicts = [
        patient_id
        for patient_id, classes in patient_classes.items()
        if len(classes) > 1
    ]
    if class_conflicts:
        raise ValueError(
            f"Found {len(class_conflicts)} patient IDs with conflicting classes."
        )
    duplicate_images = {
        sha256: rows for sha256, rows in image_hashes.items() if len(rows) > 1
    }
    if duplicate_images:
        raise ValueError(
            f"Found {len(duplicate_images)} exact duplicate output images."
        )

    print(
        "Validation passed: no missing images, patient leakage, class "
        "conflicts, or exact duplicate images."
    )


def preprocess_pipeline(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert DICOM files and write a binary 70/15/15 ImageFolder dataset."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    labels_path = raw_data_dir / "stage_2_detailed_class_info.csv"
    images_dir = raw_data_dir / "stage_2_train_images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {images_dir}")

    records = read_binary_labels(labels_path)
    check_source_duplicates(records, images_dir)
    splits = stratified_split(records, seed)
    initialize_output(output_dir, overwrite)

    summary: dict[str, dict[str, int]] = {}
    all_manifest_rows: list[dict[str, str]] = []
    total_images = sum(len(rows) for rows in splits.values())
    completed = 0

    for split, split_records in splits.items():
        manifest_rows = []
        for record in split_records:
            patient_id = record["patient_id"]
            class_name = record["class"]
            source_path = images_dir / f"{patient_id}.dcm"
            if not source_path.exists():
                raise FileNotFoundError(f"DICOM image not found: {source_path}")

            relative_image_path = Path(split) / class_name / f"{patient_id}.jpg"
            target_path = output_dir / relative_image_path
            image = dicom_to_grayscale_image(source_path)
            image.save(target_path, format="JPEG", quality=95)
            sha256 = file_sha256(target_path)

            manifest_rows.append(
                {
                    "patient_id": patient_id,
                    "class": class_name,
                    "source_path": str(source_path),
                    "image_path": str(relative_image_path),
                    "split": split,
                    "sha256": sha256,
                }
            )
            completed += 1
            if completed % 500 == 0 or completed == total_images:
                print(f"Converted {completed}/{total_images} images")

        write_manifest(output_dir, split, manifest_rows)
        all_manifest_rows.extend(manifest_rows)
        counts = Counter(row["class"] for row in manifest_rows)
        summary[split] = {
            class_name: counts[class_name] for class_name in OUTPUT_CLASSES
        }
        summary[split]["total"] = len(manifest_rows)

    write_manifest(output_dir, "split_manifest", all_manifest_rows)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
    validate_prepared_dataset(output_dir, all_manifest_rows)

    print(json.dumps(summary, indent=2))
    print(f"Prepared dataset written to: {output_dir.resolve()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RSNA DICOM images for NORMAL/PNEUMONIA training."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("dataset"),
        help="Raw dataset folder containing the class CSV and stage_2_train_images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_clean_70_15_15"),
        help="Output ImageFolder dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing non-empty output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    preprocess_pipeline(
        arguments.raw_dir,
        arguments.output_dir,
        arguments.seed,
        overwrite=arguments.overwrite,
    )
