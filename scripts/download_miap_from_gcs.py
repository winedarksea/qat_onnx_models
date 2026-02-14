#!/usr/bin/env python3
"""
Download MIAP data from GCS and build a COCO-style dataset for picodet_v5_qat.py.

Default output layout:
  D:/img_data/miap/
    train2017/
    val2017/
    annotations/
      instances_train2017.json
      instances_val2017.json

Example:
  python scripts/download_miap_from_gcs.py --output-root D:/img_data/miap
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

try:
    import gcsfs
except ImportError:  # pragma: no cover
    gcsfs = None

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover
    Image = None

    class UnidentifiedImageError(Exception):
        pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


DEFAULT_BUCKET = "colin-miap-madness"
DEFAULT_CSV_NAME = "vertex_miap_import.csv"
DEFAULT_OUTPUT_ROOT = "./miap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MIAP images + annotations from GCS and emit COCO train/val files."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket containing the MIAP manifest and images.")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="Manifest CSV filename in the bucket.")
    parser.add_argument(
        "--gcs-csv-path",
        default=None,
        help="Optional full gs:// path to manifest CSV. Overrides --bucket/--csv-name.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Dataset output root.")
    parser.add_argument("--val-split", type=float, default=0.07, help="Validation split fraction (0.0 to <1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on total images to process.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for GCS downloads.")
    parser.add_argument("--min-box-px", type=float, default=0.0, help="Drop boxes smaller than this in source image pixels.")
    parser.add_argument("--force-redownload", action="store_true", help="Re-download files even if already present locally.")
    return parser.parse_args()


def read_manifest(fs: gcsfs.GCSFileSystem, gcs_csv_path: str) -> dict[str, list[tuple[float, float, float, float]]]:
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    with fs.open(gcs_csv_path, "r") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if len(row) < 9:
                continue

            gcs_path = row[1].strip()
            if not gcs_path.startswith("gs://"):
                continue

            try:
                x_min = float(row[3])
                y_min = float(row[4])
                x_max = float(row[7])
                y_max = float(row[8])
            except ValueError:
                if row_idx <= 3:
                    print(f"[WARN] Invalid numeric row at line {row_idx}: {row}")
                continue

            grouped.setdefault(gcs_path, []).append((x_min, y_min, x_max, y_max))
    return grouped


def batch_download(
    fs: gcsfs.GCSFileSystem,
    split_name: str,
    gcs_paths: list[str],
    split_dir: Path,
    batch_size: int,
    force_redownload: bool,
) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    path_map: dict[str, Path] = {}
    to_download_remote: list[str] = []
    to_download_local: list[Path] = []

    for gcs_path in gcs_paths:
        image_name = gcs_path.rsplit("/", 1)[-1]
        local_path = split_dir / image_name
        path_map[gcs_path] = local_path
        if force_redownload or not local_path.exists():
            to_download_remote.append(gcs_path)
            to_download_local.append(local_path)

    if not to_download_remote:
        print(f"[INFO] {split_name}: no new files needed.")
        return path_map

    print(f"[INFO] {split_name}: downloading {len(to_download_remote)} images...")
    failures = 0
    for i in tqdm(range(0, len(to_download_remote), batch_size), desc=f"download-{split_name}", unit="batch"):
        batch_remote = to_download_remote[i:i + batch_size]
        batch_local = to_download_local[i:i + batch_size]
        batch_local_str = [str(p) for p in batch_local]

        try:
            fs.get(batch_remote, batch_local_str)
        except Exception:
            for remote, local_str in zip(batch_remote, batch_local_str):
                try:
                    fs.get(remote, local_str)
                except Exception:
                    failures += 1

    if failures:
        print(f"[WARN] {split_name}: failed to download {failures} images.")
    return path_map


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))


def build_coco_split(
    split_name: str,
    gcs_paths: list[str],
    grouped_boxes: dict[str, list[tuple[float, float, float, float]]],
    local_paths: dict[str, Path],
    min_box_px: float,
) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1
    skipped_images = 0

    for gcs_path in tqdm(gcs_paths, desc=f"annotate-{split_name}", unit="img"):
        local_path = local_paths[gcs_path]
        if not local_path.exists():
            skipped_images += 1
            continue

        try:
            with Image.open(local_path) as im:
                width, height = im.size
        except (UnidentifiedImageError, OSError):
            skipped_images += 1
            continue

        image_id = next_image_id
        next_image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": local_path.name,
                "width": int(width),
                "height": int(height),
            }
        )

        for x_min, y_min, x_max, y_max in grouped_boxes.get(gcs_path, []):
            x1 = _clamp01(x_min) * width
            y1 = _clamp01(y_min) * height
            x2 = _clamp01(x_max) * width
            y2 = _clamp01(y_max) * height

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0.0 or bh <= 0.0:
                continue
            if min_box_px > 0.0 and (bw < min_box_px or bh < min_box_px):
                continue

            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    now_iso = datetime.now(timezone.utc).isoformat()
    coco = {
        "info": {
            "description": "MIAP person subset converted from GCS manifest",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": now_iso,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    if skipped_images:
        print(f"[WARN] {split_name}: skipped {skipped_images} missing/corrupt images.")
    print(f"[INFO] {split_name}: images={len(images)} annotations={len(annotations)}")
    return coco


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    if gcsfs is None:
        raise SystemExit("Missing dependency: gcsfs. Install with: pip install gcsfs")
    if Image is None:
        raise SystemExit("Missing dependency: Pillow. Install with: pip install Pillow")

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0.0, 1.0).")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be > 0 when provided.")

    gcs_csv_path = args.gcs_csv_path or f"gs://{args.bucket}/{args.csv_name}"
    output_root = Path(args.output_root)
    train_dir = output_root / "train2017"
    val_dir = output_root / "val2017"
    ann_dir = output_root / "annotations"

    print(f"[INFO] Reading manifest: {gcs_csv_path}")
    fs = gcsfs.GCSFileSystem()
    grouped_boxes = read_manifest(fs, gcs_csv_path)
    if not grouped_boxes:
        raise RuntimeError(f"No usable rows found in manifest: {gcs_csv_path}")

    all_paths = list(grouped_boxes.keys())
    rnd = random.Random(args.seed)
    rnd.shuffle(all_paths)

    if args.max_images is not None:
        all_paths = all_paths[: args.max_images]

    if len(all_paths) < 2:
        raise RuntimeError("Need at least 2 images to create train/val split.")

    raw_val_count = int(round(len(all_paths) * args.val_split))
    val_count = max(1, raw_val_count) if args.val_split > 0.0 else 0
    val_count = min(val_count, len(all_paths) - 1)

    val_paths = all_paths[:val_count]
    train_paths = all_paths[val_count:]

    print(
        f"[INFO] Split sizes: train={len(train_paths)} val={len(val_paths)} "
        f"(total={len(all_paths)}, val_split={args.val_split})"
    )

    train_local_paths = batch_download(
        fs=fs,
        split_name="train",
        gcs_paths=train_paths,
        split_dir=train_dir,
        batch_size=args.batch_size,
        force_redownload=args.force_redownload,
    )
    val_local_paths = batch_download(
        fs=fs,
        split_name="val",
        gcs_paths=val_paths,
        split_dir=val_dir,
        batch_size=args.batch_size,
        force_redownload=args.force_redownload,
    )

    train_coco = build_coco_split(
        split_name="train",
        gcs_paths=train_paths,
        grouped_boxes=grouped_boxes,
        local_paths=train_local_paths,
        min_box_px=args.min_box_px,
    )
    val_coco = build_coco_split(
        split_name="val",
        gcs_paths=val_paths,
        grouped_boxes=grouped_boxes,
        local_paths=val_local_paths,
        min_box_px=args.min_box_px,
    )

    train_json = ann_dir / "instances_train2017.json"
    val_json = ann_dir / "instances_val2017.json"
    write_json(train_json, train_coco)
    write_json(val_json, val_coco)

    print("[INFO] Done.")
    print(f"[INFO] Dataset root: {output_root}")
    print(f"[INFO] Train images: {train_dir}")
    print(f"[INFO] Val images: {val_dir}")
    print(f"[INFO] Train ann: {train_json}")
    print(f"[INFO] Val ann: {val_json}")
    print(f"[INFO] Train with: python scripts/picodet_v5_qat.py --coco_root {output_root}")


if __name__ == "__main__":
    main()
