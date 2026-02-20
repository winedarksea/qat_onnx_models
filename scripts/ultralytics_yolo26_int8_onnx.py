#!/usr/bin/env python3
"""
Ultralytics YOLO26n → (optional) COCO→YOLO dataset w/ small-box filtering → train → ONNX → INT8 ONNX.

Goals (mirrors scripts/picodet_v5_qat.py intent):
  - Use COCO 2017 as the dataset source by default (same source as picodet_v5_qat.py when use_coco=True).
  - Filter out GT boxes that become too small at a small square training size (default 312 px).
  - Train using the `ultralytics` package (in your conda env, e.g. gpu311).
  - Export an ONNX model and quantize to INT8 with onnxruntime (static calibration).
  - Optionally embed simple preprocessing (uint8 → float, /255, resize-to-square) into the ONNX graph.

Notes:
  - Ultralytics models generally expect inputs already resized/letterboxed to `imgsz` and scaled to 0..1.
    Embedding preprocessing here uses a simple *stretch resize* to (imgsz, imgsz) (like ResizeNorm in picodet_lib_v2),
    which may differ from Ultralytics' default letterbox padding behavior. Keep `--embed_preprocess` optional.
  - "YOLO26" is assumed to be available to Ultralytics either as a local .pt file or via Ultralytics HUB.
    Do NOT hardcode API keys in the script; pass via env var or CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import onnx
import onnxruntime as ort  # type: ignore

import numpy as np


# COCO’s official 80-class list (order matters!)
CANONICAL_COCO80_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

CANONICAL_COCO80_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

CANONICAL_COCO80_MAP: dict[int, int] = {coco_id: i for i, coco_id in enumerate(CANONICAL_COCO80_IDS)}


log = logging.getLogger("ultralytics_yolo26_int8_onnx")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_label_map_and_names_from_ann(ann_path: Path):
    with open(ann_path, 'r') as f:
        d = json.load(f)
    cats = sorted(d['categories'], key=lambda c: c['id'])
    catid2contig = {c['id']: i for i, c in enumerate(cats)}
    id2name = {i: c['name'] for i, c in enumerate(cats)}
    return catid2contig, id2name


def _safe_symlink_dir(src: Path, dst: Path) -> None:
    """
    Create dst as a symlink to src. Falls back to copying the directory if symlinks are unavailable.
    """
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)
    except Exception as e:
        log.warning("Symlink failed (%s). Copying directory tree instead: %s -> %s", e, src, dst)
        shutil.copytree(src, dst)


@dataclass(frozen=True)
class FilterConfig:
    imgsz: int
    min_box_size: int
    max_cover: float = 0.90
    ar_min: float = 0.05
    ar_max: float = 20.0
    letterbox: bool = True


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _filter_and_convert_coco_box_to_yolo(
    coco_xywh: list[float],
    *,
    img_w: int,
    img_h: int,
    cfg: FilterConfig,
) -> tuple[float, float, float, float] | None:
    """
    Returns YOLO-normalized (cx, cy, w, h) in [0,1] if valid, else None.
    Filtering is based on expected post-resize size at cfg.imgsz, matching picodet_v5_qat.py's
    "remove small boxes" spirit for small training sizes.
    """
    x, y, w, h = [float(v) for v in coco_xywh]
    if w <= 0 or h <= 0:
        return None

    x1 = _clamp(x, 0.0, float(img_w))
    y1 = _clamp(y, 0.0, float(img_h))
    x2 = _clamp(x + w, 0.0, float(img_w))
    y2 = _clamp(y + h, 0.0, float(img_h))

    w2 = x2 - x1
    h2 = y2 - y1
    if w2 <= 0 or h2 <= 0:
        return None

    if cfg.letterbox:
        scale = min(cfg.imgsz / float(img_w), cfg.imgsz / float(img_h))
        w_r = w2 * scale
        h_r = h2 * scale
        H_r = float(img_h) * scale
        W_r = float(img_w) * scale
    else:
        sx = cfg.imgsz / float(img_w)
        sy = cfg.imgsz / float(img_h)
        w_r = w2 * sx
        h_r = h2 * sy
        W_r = float(cfg.imgsz)
        H_r = float(cfg.imgsz)

    ar = w_r / (h_r + 1e-6)
    tiny = (w_r < cfg.min_box_size) or (h_r < cfg.min_box_size)
    big = (w_r * h_r) > (cfg.max_cover * H_r * W_r)
    weird = (ar < cfg.ar_min) or (ar > cfg.ar_max)
    if tiny or big or weird:
        return None

    cx = (x1 + x2) / 2.0 / float(img_w)
    cy = (y1 + y2) / 2.0 / float(img_h)
    ww = w2 / float(img_w)
    hh = h2 / float(img_h)

    # final sanity
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < ww <= 1.0 and 0.0 < hh <= 1.0):
        return None

    return cx, cy, ww, hh


def prepare_coco2017_yolo_dataset(
    *,
    coco_root: Path,
    out_root: Path,
    imgsz: int,
    min_box_size: int,
    write_empty_labels: bool,
    drop_became_empty: bool,
    letterbox_filter: bool,
    seed: int,
    force: bool = False,
) -> Path:
    """
    Create an Ultralytics-friendly dataset directory:
      out_root/
        images/train -> coco_root/train2017 (symlink)
        images/val   -> coco_root/val2017   (symlink)
        labels/train/*.txt
        labels/val/*.txt
        data.yaml
    Returns the path to the generated data.yaml.
    """
    out_root = out_root.absolute()
    coco_root = coco_root.absolute()

    meta_path = out_root / "dataset_meta.json"
    meta = {
        "coco_root": coco_root.as_posix(),
        "imgsz": int(imgsz),
        "min_box_size": int(min_box_size),
        "write_empty_labels": bool(write_empty_labels),
        "drop_became_empty": bool(drop_became_empty),
        "letterbox_filter": bool(letterbox_filter),
        "canonical_coco80_ids": CANONICAL_COCO80_IDS,
        "canonical_coco80_names": CANONICAL_COCO80_NAMES,
    }

    data_yaml_existing = out_root / "data.yaml"
    if not force and data_yaml_existing.exists() and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
        except Exception:
            existing = None
        if existing == meta:
            log.info("Dataset already prepared (meta match). Reusing: %s", data_yaml_existing)
            return data_yaml_existing
        log.info("Dataset meta changed; rebuilding dataset at %s", out_root)

    ann_train = coco_root / "annotations" / "instances_train2017.json"
    ann_val = coco_root / "annotations" / "instances_val2017.json"
    img_train = coco_root / "train2017"
    img_val = coco_root / "val2017"

    for p in [ann_train, ann_val]:
        if not p.exists():
            raise FileNotFoundError(f"Missing COCO annotation file: {p}")
    for p in [img_train, img_val]:
        if not p.exists():
            raise FileNotFoundError(f"Missing COCO image directory: {p}")

    _ensure_dir(out_root)
    _ensure_dir(out_root / "images")
    _ensure_dir(out_root / "labels")
    _safe_symlink_dir(img_train, out_root / "images" / "train")
    _safe_symlink_dir(img_val, out_root / "images" / "val")
    _ensure_dir(out_root / "labels" / "train")
    _ensure_dir(out_root / "labels" / "val")

    filter_cfg = FilterConfig(imgsz=imgsz, min_box_size=min_box_size, letterbox=letterbox_filter)

    def convert_split(ann_path: Path, split: str) -> None:
        log.info("Parsing %s ...", ann_path)
        with ann_path.open("r") as f:
            d = json.load(f)

        images = d.get("images", [])
        annotations = d.get("annotations", [])

        img_by_id: dict[int, dict] = {int(im["id"]): im for im in images}
        ann_by_img: dict[int, list[dict]] = {}
        for a in annotations:
            if a.get("iscrowd", 0):
                continue
            img_id = int(a["image_id"])
            ann_by_img.setdefault(img_id, []).append(a)

        kept_boxes = 0
        dropped_boxes = 0
        empty_images = 0
        dropped_images_became_empty = 0

        list_path = out_root / f"{split}.txt"
        list_lines: list[str] = []

        for img_id in sorted(img_by_id.keys()):
            im = img_by_id[img_id]
            file_name = im["file_name"]
            w = int(im["width"])
            h = int(im["height"])

            yolo_lines: list[str] = []
            anns_in = ann_by_img.get(img_id, [])
            n_in = 0
            for a in anns_in:
                cid = int(a["category_id"])
                if cid not in CANONICAL_COCO80_MAP:
                    continue
                n_in += 1
                cls = CANONICAL_COCO80_MAP[cid]
                box = _filter_and_convert_coco_box_to_yolo(a["bbox"], img_w=w, img_h=h, cfg=filter_cfg)
                if box is None:
                    dropped_boxes += 1
                    continue
                cx, cy, bw, bh = box
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                kept_boxes += 1

            out_txt = out_root / "labels" / split / (Path(file_name).with_suffix(".txt").name)

            started_negative = (n_in == 0)
            became_empty = (n_in > 0) and (len(yolo_lines) == 0)
            keep_image = started_negative or (not became_empty) or (not drop_became_empty)

            if not keep_image:
                dropped_images_became_empty += 1
                if out_txt.exists():
                    out_txt.unlink()
                continue

            # Keep '/images/' in the path string (do not resolve symlinks), since Ultralytics derives
            # label paths by string replacement in many versions.
            img_path = (out_root / "images" / split / file_name).absolute().as_posix()
            list_lines.append(img_path)

            if yolo_lines:
                out_txt.write_text("\n".join(yolo_lines) + "\n")
            else:
                empty_images += 1
                if write_empty_labels:
                    out_txt.write_text("")
                elif out_txt.exists():
                    out_txt.unlink()

        list_path.write_text("\n".join(list_lines) + "\n")
        log.info("Wrote image list: %s (%d images)", list_path, len(list_lines))
        log.info(
            "Split=%s: kept_boxes=%d dropped_boxes=%d empty_images=%d dropped_images_became_empty=%d write_empty_labels=%s drop_became_empty=%s",
            split,
            kept_boxes,
            dropped_boxes,
            empty_images,
            dropped_images_became_empty,
            write_empty_labels,
            drop_became_empty,
        )

    convert_split(ann_train, "train")
    convert_split(ann_val, "val")

    data_yaml = out_root / "data.yaml"
    yaml_lines = [
        f"path: {out_root.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "names:",
    ]
    for i, n in enumerate(CANONICAL_COCO80_NAMES):
        yaml_lines.append(f"  {i}: {n}")
    yaml_txt = "\n".join(yaml_lines) + "\n"
    data_yaml.write_text(yaml_txt)
    log.info("Wrote dataset YAML: %s", data_yaml)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    log.info("Wrote dataset meta: %s", meta_path)
    return data_yaml


def prepare_custom_yolo_dataset(
    *,
    data_root: Path,
    ann_path: Path,
    out_root: Path,
    imgsz: int,
    min_box_size: int,
    val_pct: float,
    write_empty_labels: bool,
    drop_became_empty: bool,
    letterbox_filter: bool,
    seed: int,
    force: bool = False,
) -> Path:
    """
    Prepare a custom dataset (non-COCO) for Ultralytics YOLO.
    Performs random split and (optional) class-balanced oversampling if desired.
    """
    out_root = out_root.absolute()
    data_root = data_root.absolute()
    ann_path = ann_path.absolute()

    catid2contig, id2name = build_label_map_and_names_from_ann(ann_path)
    num_classes = len(id2name)

    meta = {
        "data_root": data_root.as_posix(),
        "ann_path": ann_path.as_posix(),
        "imgsz": int(imgsz),
        "min_box_size": int(min_box_size),
        "val_pct": float(val_pct),
        "num_classes": num_classes,
        "class_names": [id2name[i] for i in range(num_classes)],
    }

    data_yaml_existing = out_root / "data.yaml"
    if not force and data_yaml_existing.exists() and (out_root / "dataset_meta.json").exists():
        try:
            existing = json.loads((out_root / "dataset_meta.json").read_text())
        except Exception:
            existing = None
        if existing == meta:
            log.info("Custom dataset already prepared. Reusing: %s", data_yaml_existing)
            return data_yaml_existing

    log.info("Preparing custom dataset at %s ...", out_root)
    _ensure_dir(out_root)
    _ensure_dir(out_root / "images")
    _ensure_dir(out_root / "labels")
    _ensure_dir(out_root / "labels" / "train")
    _ensure_dir(out_root / "labels" / "val")

    with ann_path.open("r") as f:
        d = json.load(f)

    images = d.get("images", [])
    annotations = d.get("annotations", [])
    img_by_id = {int(im["id"]): im for im in images}
    ann_by_img = {}
    for a in annotations:
        if a.get("iscrowd", 0):
            continue
        img_id = int(a["image_id"])
        ann_by_img.setdefault(img_id, []).append(a)

    all_ids = sorted(img_by_id.keys())
    random.seed(seed)
    random.shuffle(all_ids)
    
    val_n = max(1, int(val_pct * len(all_ids)))
    val_ids = set(all_ids[:val_n])
    train_ids = all_ids[val_n:]

    filter_cfg = FilterConfig(imgsz=imgsz, min_box_size=min_box_size, letterbox=letterbox_filter)

    def process_split(ids: list[int] | set[int], split: str) -> list[str]:
        log.info("Processing split '%s' (%d images) ...", split, len(ids))
        list_lines = []
        
        # Track samples per class for possible oversampling
        sample_weights = []
        temp_list = []
        
        for img_id in sorted(list(ids)):
            im = img_by_id[img_id]
            file_name = im["file_name"]
            w, h = int(im["width"]), int(im["height"])
            
            yolo_lines = []
            anns = ann_by_img.get(img_id, [])
            primary_cls = -1
            if anns:
                # Use the first valid annotation's class as the primary class for balancing (matching picodet behavior)
                cid = int(anns[0]["category_id"])
                if cid in catid2contig:
                    primary_cls = catid2contig[cid]

            for a in anns:
                cid = int(a["category_id"])
                if cid not in catid2contig: continue
                cls = catid2contig[cid]
                box = _filter_and_convert_coco_box_to_yolo(a["bbox"], img_w=w, img_h=h, cfg=filter_cfg)
                if box is None: continue
                cx, cy, bw, bh = box
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            out_txt = out_root / "labels" / split / (Path(file_name).with_suffix(".txt").name)
            
            # Decide if we keep this image
            is_empty = (len(yolo_lines) == 0)
            originally_had_boxes = (len(anns) > 0)
            became_empty = originally_had_boxes and is_empty
            
            keep = True
            if became_empty and drop_became_empty:
                keep = False

            if keep:
                img_src = data_root / "images" / file_name
                img_dst = out_root / "images" / split / file_name
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                if not img_dst.exists():
                    try:
                        img_dst.symlink_to(img_src)
                    except:
                        shutil.copy2(img_src, img_dst)

                if yolo_lines:
                    out_txt.write_text("\n".join(yolo_lines) + "\n")
                elif write_empty_labels:
                    out_txt.write_text("")
                
                img_path = img_dst.absolute().as_posix()
                temp_list.append((img_path, primary_cls))

        if split == "train" and num_classes == 3:
            log.info("Applying class-balanced oversampling (target 3:1:1) ...")
            # Following picodet_v5_qat: target weights {0: 3.0, 1: 1.0, 2: 1.0}
            target_weights = {0: 3.0, 1: 1.0, 2: 1.0}
            class_counts = {c: 0 for c in range(3)}
            for _, cls in temp_list:
                if cls >= 0:
                    class_counts[cls] += 1
            
            # Calculate weight per class: weight = target / count
            weights_per_class = {}
            for c in range(3):
                if class_counts[c] > 0:
                    weights_per_class[c] = target_weights[c] / class_counts[c]
                else:
                    weights_per_class[c] = 0.0
            
            # Calculate image weights
            img_weights = []
            for _, cls in temp_list:
                if cls >= 0:
                    img_weights.append(weights_per_class[cls])
                else:
                    img_weights.append(min(weights_per_class.values()) if weights_per_class else 1.0)
            
            # Resample with replacement to keep the original training set size
            total_weight = sum(img_weights)
            if total_weight > 0:
                img_probs = [w / total_weight for w in img_weights]
                # Use random.choices for weighted sampling with replacement
                # This approximates the WeightedRandomSampler behavior
                indices = random.choices(range(len(temp_list)), weights=img_probs, k=len(temp_list))
                list_lines = [temp_list[i][0] for i in indices]
            else:
                list_lines = [x[0] for x in temp_list]
        else:
            list_lines = [x[0] for x in temp_list]
            
        return list_lines

    train_list = process_split(train_ids, "train")
    val_list = process_split(val_ids, "val")

    (out_root / "train.txt").write_text("\n".join(train_list) + "\n")
    (out_root / "val.txt").write_text("\n".join(val_list) + "\n")

    yaml_lines = [
        f"path: {out_root.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "names:",
    ]
    for i in range(num_classes):
        yaml_lines.append(f"  {i}: {id2name[i]}")
    
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n")
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    
    log.info("Custom dataset preparation complete: %s", data_yaml)
    return data_yaml


def _import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import `ultralytics`. Activate your conda env (e.g. gpu311) with ultralytics installed."
        ) from e
    return YOLO


def train_ultralytics_detector(
    *,
    data_yaml: Path,
    yolo_weights: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | None,
    workers: int,
    project: str,
    name: str,
    exist_ok: bool,
    seed: int,
) -> tuple[Path, Path]:
    """
    Trains with Ultralytics and returns (best weight path, run directory).
    """
    YOLO = _import_ultralytics()

    log.info("Loading pretrained model: %s", yolo_weights)
    model = YOLO(yolo_weights)

    log.info("Training (imgsz=%d epochs=%d batch=%d device=%s) ...", imgsz, epochs, batch, device)
    # Ultralytics handles seeding internally, but we also seed python/numpy for our own steps.
    random.seed(seed)
    np.random.seed(seed)

    _ = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        seed=seed,
        pretrained=True,
        val=True,
    )

    # Best-effort discovery of best.pt
    # 1. Try to get it directly from the trainer if available
    run_dir = None
    if hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
        run_dir = Path(model.trainer.save_dir)
        log.info("Discovered run_dir from trainer: %s", run_dir)

    if run_dir is None or not run_dir.exists():
        # 2. Fallback to specified project/name
        run_dir = Path(project) / name
        log.info("Checking run_dir: %s", run_dir)

    if not run_dir.exists():
        # 3. Fallback to what Ultralytics often does: prepending 'runs/detect' if project is relative
        alt_run_dir = Path("runs/detect") / project / name
        if alt_run_dir.exists():
            run_dir = alt_run_dir
            log.info("Discovered run_dir at alternative path: %s", run_dir)

    candidates = [
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
        run_dir / "weights" / "best.onnx",
    ]
    for c in candidates:
        if c.exists() and c.suffix == ".pt":
            log.info("Using checkpoint: %s", c)
            return c, run_dir

    # If Ultralytics auto-appended a suffix (name2, name3, ...)
    search_roots = [Path(project)]
    if Path("runs/detect").exists():
        search_roots.append(Path("runs/detect") / project)

    for root in search_roots:
        if root.exists():
            dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(name)]
            dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for d in dirs:
                for c in [d / "weights" / "best.pt", d / "weights" / "last.pt"]:
                    if c.exists():
                        log.info("Using checkpoint: %s", c)
                        return c, d

    # Last resort: search for a recent best.pt anywhere that might be related
    for root in search_roots:
        if root.exists():
            best_pts = [p for p in root.rglob("best.pt") if p.is_file()]
            best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if best_pts:
                log.info("Using checkpoint (fallback search): %s", best_pts[0])
                return best_pts[0], best_pts[0].parent.parent

    raise FileNotFoundError(f"Could not find best.pt/last.pt under {project}/{name} (or 'runs/detect/' variants).")


def export_onnx_with_ultralytics(
    *,
    weights_pt: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    nms: bool,
    max_det: int,
    device: str | None,
    out_dir: Path,
) -> Path:
    """
    Export ONNX using Ultralytics export pipeline and return the ONNX path.
    """
    YOLO = _import_ultralytics()
    model = YOLO(str(weights_pt))
    _ensure_dir(out_dir)

    log.info("Exporting ONNX (opset=%d simplify=%s nms=%s max_det=%d) ...", opset, simplify, nms, max_det)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        nms=nms,
        max_det=max_det,
        device=device,
    )

    # Ultralytics returns path-like in recent versions, but keep it robust.
    if isinstance(exported, (str, Path)) and Path(exported).exists():
        onnx_path = Path(exported)
    else:
        # Fallback: search next to weights or in CWD
        guess = weights_pt.with_suffix(".onnx")
        if guess.exists():
            onnx_path = guess
        else:
            found = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not found:
                raise FileNotFoundError("Ultralytics export did not produce an ONNX file we can locate.")
            onnx_path = found[0]

    # Move to requested directory if needed
    final = out_dir / onnx_path.name
    if onnx_path.resolve() != final.resolve():
        shutil.copy2(onnx_path, final)
        onnx_path = final

    log.info("Exported ONNX: %s", onnx_path)
    return onnx_path


def embed_uint8_preprocess_into_onnx(
    *,
    in_onnx: Path,
    out_onnx: Path,
    imgsz: int,
    input_scale: float = 1.0 / 255.0,
    resize: bool = True,
) -> Path:
    """
    Creates a new ONNX model that:
      uint8 NCHW -> Cast(float) -> Mul(input_scale) -> (optional) Resize -> feeds original model.

    The Resize is a *stretch resize* to (imgsz, imgsz) using ONNX Resize "sizes" input.
    """
    try:
        from onnx import TensorProto, helper, numpy_helper  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing `onnx` package. Install it in your env to embed preprocessing.") from e

    model = onnx.load(str(in_onnx))
    if len(model.graph.input) < 1:
        raise ValueError("Unexpected ONNX: no graph inputs.")

    orig_in = model.graph.input[0]
    orig_in_name = orig_in.name

    new_input_name = "images_uint8"
    new_input = helper.make_tensor_value_info(
        new_input_name,
        TensorProto.UINT8,
        ["BatchSize", 3, "Height", "Width"],
    )

    cast_out = f"{orig_in_name}__cast_f32"
    scaled_out = orig_in_name  # critical: feed existing graph by producing original input name if no resize

    nodes = []
    nodes.append(
        helper.make_node(
            "Cast",
            inputs=[new_input_name],
            outputs=[cast_out],
            name="Preprocess_CastUint8ToFloat",
            to=TensorProto.FLOAT,
        )
    )

    # Guidelines encourage Mul for speed (mean=0, std=255)
    scale_name = "Preprocess_InputScale"
    scale_tensor = numpy_helper.from_array(np.array([input_scale], dtype=np.float32), name=scale_name)
    model.graph.initializer.append(scale_tensor)

    if not resize:
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[cast_out, scale_name],
                outputs=[scaled_out],
                name="Preprocess_Scale",
            )
        )
    else:
        # If resizing, standard practice is Mul then Resize
        pre_resize_scaled = f"{orig_in_name}__pre_resize_scaled"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[cast_out, scale_name],
                outputs=[pre_resize_scaled],
                name="Preprocess_Scale",
            )
        )
        
        # Resize(x, roi, scales, sizes) in opset 13+.
        roi_name = "Preprocess_ResizeRoi"
        scales_name = "Preprocess_ResizeScales"
        sizes_name = "Preprocess_ResizeSizes"
        model.graph.initializer.append(numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name))
        model.graph.initializer.append(numpy_helper.from_array(np.array([], dtype=np.float32), name=scales_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([1, 3, imgsz, imgsz], dtype=np.int64), name=sizes_name)
        )
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=[pre_resize_scaled, roi_name, scales_name, sizes_name],
                outputs=[scaled_out],
                name="Preprocess_ResizeToSquare",
                mode="linear",
                nearest_mode="floor",
            )
        )

    # Replace original float input with uint8 input. Existing nodes continue to reference `orig_in_name`,
    # which is now produced by our preprocessing nodes.
    del model.graph.input[0]
    model.graph.input.insert(0, new_input)
    # ONNX graph.node is a protobuf repeated field; slice assignment is not
    # supported in newer protobuf versions.
    for node in reversed(nodes):
        model.graph.node.insert(0, node)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_onnx))
    log.info("Wrote preprocess-embedded ONNX: %s", out_onnx)
    return out_onnx


def _letterbox_uint8(
    img: np.ndarray,
    *,
    new_shape: int,
    color: int = 114,
) -> np.ndarray:
    """
    Minimal letterbox used for calibration images for Ultralytics-style inputs.
    img: HWC uint8 RGB
    returns: HWC uint8 RGB, square new_shape
    """
    from PIL import Image  # type: ignore

    h, w = img.shape[:2]
    scale = min(new_shape / float(h), new_shape / float(w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))

    pil = Image.fromarray(img, mode="RGB").resize((nw, nh), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (new_shape, new_shape), (color, color, color))
    left = (new_shape - nw) // 2
    top = (new_shape - nh) // 2
    canvas.paste(pil, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def _load_image_rgb_uint8(path: Path) -> np.ndarray:
    from PIL import Image  # type: ignore

    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


class NumpyCalibrationDataReader:
    """
    onnxruntime.quantization.CalibrationDataReader compatible.
    """

    def __init__(self, input_name: str, batches: list[np.ndarray]):
        self.input_name = input_name
        self._batches = batches
        self._iter: Iterator[np.ndarray] | None = None

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self._batches)
        try:
            x = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: x}


def quantize_onnx_int8_static(
    *,
    in_onnx: Path,
    out_onnx: Path,
    calib_images: list[Path],
    imgsz: int,
    expects_uint8_input: bool,
    use_letterbox_for_float_input: bool,
    per_channel: bool,
    reduce_range: bool,
) -> Path:
    """
    Static INT8 quantization using onnxruntime.quantization with a small calibration set.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as e:
        raise RuntimeError("Missing `onnxruntime` + quantization extras in this env.") from e

    sess = ort.InferenceSession(str(in_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    batches: list[np.ndarray] = []
    for p in calib_images:
        img = _load_image_rgb_uint8(p)  # HWC RGB
        from PIL import Image  # type: ignore
        if expects_uint8_input:
            # Keep calibration shapes consistent even if the model input is dynamic.
            img = np.asarray(
                Image.fromarray(img).resize((imgsz, imgsz), resample=Image.BILINEAR),
                dtype=np.uint8,
            )
            x = np.transpose(img, (2, 0, 1))[None, ...].astype(np.uint8)
        else:
            if use_letterbox_for_float_input:
                img = _letterbox_uint8(img, new_shape=imgsz)
            else:
                # Stretch resize for calibration
                img = np.asarray(
                    Image.fromarray(img).resize((imgsz, imgsz), resample=Image.BILINEAR),
                    dtype=np.uint8,
                )
            x = (np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)) * (1.0 / 255.0)
        batches.append(x)

    dr: CalibrationDataReader = NumpyCalibrationDataReader(input_name, batches)

    log.info(
        "Quantizing INT8 (static) calib=%d per_channel=%s reduce_range=%s input=%s",
        len(batches),
        per_channel,
        reduce_range,
        "uint8" if expects_uint8_input else "float32",
    )
    
    # Guidelines: keep post-processing path in float to preserve small score/box values.
    nodes_to_exclude: set[str] = set()
    temp_model = onnx.load(str(in_onnx))

    # Always exclude the guideline wrapper nodes.
    for node in temp_model.graph.node:
        if node.name.startswith("Guideline_"):
            nodes_to_exclude.add(node.name)

    # Also exclude the immediate postprocess path feeding Guideline_ReshapeN6's source tensor.
    # This avoids QDQ around score/box consolidation that can collapse scores to zero.
    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in temp_model.graph.node:
        for out_name in node.output:
            output_to_node[out_name] = node

    reshape_node = next((n for n in temp_model.graph.node if n.name == "Guideline_ReshapeN6"), None)
    if reshape_node is not None and reshape_node.input:
        source_tensor = reshape_node.input[0]
        frontier = [source_tensor]
        seen_tensors: set[str] = set()
        stop_ops = {"Conv", "ConvTranspose", "Gemm", "MatMul", "QLinearConv", "QLinearMatMul"}

        while frontier:
            tensor_name = frontier.pop()
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)

            producer = output_to_node.get(tensor_name)
            if producer is None or not producer.name:
                continue

            if producer.op_type in stop_ops:
                continue

            nodes_to_exclude.add(producer.name)
            for src in producer.input:
                if src:
                    frontier.append(src)

    nodes_to_exclude_list = sorted(nodes_to_exclude)
    log.info("Quantization nodes_to_exclude=%d", len(nodes_to_exclude_list))

    quantize_static(
        model_input=str(in_onnx),
        model_output=str(out_onnx),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        nodes_to_exclude=nodes_to_exclude_list,
    )


    log.info("Wrote INT8 ONNX: %s", out_onnx)
    return out_onnx


def ort_optimize_onnx(*, in_onnx: Path, out_onnx: Path) -> Path:
    """
    Save an ORT-optimized model (often helpful after quantization).
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(out_onnx)
    _ = ort.InferenceSession(str(in_onnx), so, providers=["CPUExecutionProvider"])
    if not out_onnx.exists():
        raise RuntimeError(f"ORT did not write optimized model to {out_onnx}")
    log.info("Wrote ORT-optimized ONNX: %s", out_onnx)
    return out_onnx


def _collect_calib_images(images_dir: Path, n: int, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    if not all_imgs:
        raise FileNotFoundError(f"No images found under {images_dir}")
    rng = random.Random(seed)
    rng.shuffle(all_imgs)
    return all_imgs[: min(n, len(all_imgs))]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Ultralytics YOLO26n on COCO (filtered small boxes) and export INT8 ONNX."
    )
    p.add_argument("--no_coco", action="store_false", dest="use_coco", default=True,
                   help="Use custom dataset source instead of COCO 2017.")
    p.add_argument("--data_root", type=Path, default=Path("test"), help="Root for custom dataset (non-COCO).")
    p.add_argument("--ann", type=Path, default=Path("test/annotations/instances.json"), help="Annotation JSON for custom dataset.")
    p.add_argument("--val_pct", type=float, default=0.05, help="Validation percentage for custom dataset split.")
    p.add_argument("--coco_root", type=Path, default=Path("coco"), help="COCO root with train2017/val2017/annotations/")
    p.add_argument("--imgsz", type=int, default=320, help="Square train/export size. Recommended multiple of 32.")
    p.add_argument("--min_box_size", type=int, default=8, help="Drop GT boxes smaller than this after resize to imgsz.")
    p.set_defaults(letterbox_filter=True)
    p.add_argument("--no_letterbox_filter", action="store_false", dest="letterbox_filter",
                   help="Use stretch scaling for small-box filtering.")

    p.add_argument("--dataset_out", type=Path, default=Path("datasets/coco2017_ultra_320"), help="Output dataset dir.")
    p.add_argument("--force_dataset_prep", action="store_true", default=False, help="Rebuild labels/lists even if cached.")
    p.set_defaults(write_empty_labels=True)
    p.add_argument("--no_write_empty_labels", action="store_false", dest="write_empty_labels",
                   help="Do not create empty label files for negatives.")
    p.set_defaults(drop_became_empty=True)
    p.add_argument("--keep_became_empty", action="store_false", dest="drop_became_empty",
                   help="Keep images that had boxes but became empty after filtering (not recommended).")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ultralytics_api_key", type=str, default=os.environ.get("ULTRALYTICS_API_KEY", "ul_4692c1115db32e76afc429ca608c38b6ec4b656f"),
                   help="Ultralytics API key for HUB downloads (stored in env var ULTRALYTICS_API_KEY).")
    p.add_argument("--yolo_weights", type=str, default="yolo26n.pt", help="Pretrained YOLO26 nano weights/model id.")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default='cuda', help="e.g. 'cuda', '0', 'mps', or 'cpu' (Ultralytics syntax).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/ultra_yolo26")
    p.add_argument("--name", type=str, default="coco312")
    p.add_argument("--exist_ok", action="store_true", default=False)
    p.add_argument("--skip_train", action="store_true", help="Skip training; export from --weights_pt.")
    p.add_argument("--weights_pt", type=Path, default=None, help="Path to a .pt checkpoint for export if --skip_train.")

    p.add_argument("--export_dir", type=Path, default=Path("exports/yolo26"))
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(simplify=True)
    p.add_argument("--no_simplify", action="store_false", dest="simplify")
    p.set_defaults(nms=True)
    p.add_argument("--nms", action="store_true", dest="nms", help="Export with NMS-like postprocessing enabled.")
    p.add_argument("--no_nms", action="store_false", dest="nms",
                   help="Disable NMS-like postprocessing in export (advanced/debug only).")
    p.add_argument("--max_det", type=int, default=100,
                   help="Maximum detections emitted by Ultralytics export when NMS/postprocess is enabled.")

    p.set_defaults(embed_preprocess=True)
    p.add_argument("--no_embed_preprocess", action="store_false", dest="embed_preprocess",
                   help="Do not embed preprocessing; keep the original float-input ONNX.")
    p.set_defaults(embed_resize=True)
    p.add_argument("--no_embed_resize", action="store_false", dest="embed_resize",
                   help="If embedding preprocessing, do not include resize-to-square.")

    p.set_defaults(quantize=True)
    p.add_argument("--no_quantize", action="store_false", dest="quantize")
    p.add_argument("--calib_images", type=int, default=2000)
    p.add_argument("--calib_split", choices=["train", "val"], default="train")
    p.set_defaults(per_channel=True)
    p.add_argument("--no_per_channel", action="store_false", dest="per_channel")
    p.add_argument("--reduce_range", action="store_true", default=False)
    p.set_defaults(ort_optimize=True)
    p.add_argument("--no_ort_optimize", action="store_false", dest="ort_optimize",
                   help="Skip ORT optimization (saves time; slightly larger models).")
    return p.parse_args(argv)


def _print_onnx_info(onnx_path: Path, prefix: str = ""):
    try:
        model = onnx.load(str(onnx_path))
        inputs = [f"{i.name}{[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}" for i in model.graph.input]
        outputs = [f"{o.name}{[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]}" for o in model.graph.output]
        log.info("%sONNX %s: inputs=%s, outputs=%s", prefix, onnx_path.name, inputs, outputs)
    except Exception as e:
        log.warning("Could not print ONNX info for %s: %s", onnx_path, e)


def fix_onnx_outputs_to_match_pico(*, onnx_path: Path, nms: bool, imgsz: int) -> None:
    """
    Renames/transforms ONNX outputs to match guidelines:
    Consolidated [N, 7] output tensor: [x1, y1, x2, y2, score, class_id, batch_idx]
    with coordinates rescaled to the original ONNX input resolution.
    """
    from onnx import helper, TensorProto, numpy_helper
    model = onnx.load(str(onnx_path))
    graph = model.graph

    # GUIDELINE: rescale coordinates to input tensor space.
    input_name = graph.input[0].name

    def _add_guideline_nodes(source_tensor: str) -> None:
        # source_tensor is [1, N, 6] where columns are [x1, y1, x2, y2, score, class]

        # 1) Reshape [1, N, 6] -> [N, 6]
        n6_name = "Guideline_Reshaped_N6"
        n6_shape = "Guideline_N6_Shape_Const"
        graph.initializer.append(numpy_helper.from_array(np.array([-1, 6], dtype=np.int64), name=n6_shape))
        graph.node.append(helper.make_node("Reshape", inputs=[source_tensor, n6_shape], outputs=[n6_name], name="Guideline_ReshapeN6"))

        # 2) Split [N, 6] -> boxes/scores/classes
        b_raw, s_raw, c_raw = "Guideline_B_Raw", "Guideline_S_Raw", "Guideline_C_Raw"
        split_const = "Guideline_Split_Const"
        graph.initializer.append(numpy_helper.from_array(np.array([4, 1, 1], dtype=np.int64), name=split_const))
        graph.node.append(helper.make_node("Split", inputs=[n6_name, split_const], outputs=[b_raw, s_raw, c_raw], axis=1, name="Guideline_SplitN6"))

        # 3) Generate batch_idx [N, 1] (all zeros, float32)
        s_shape = "Guideline_S_Shape"
        b_idx_2d = "Guideline_BatchIdx_2d"
        graph.node.append(helper.make_node("Shape", inputs=[s_raw], outputs=[s_shape], name="Guideline_GetSShape"))
        graph.node.append(
            helper.make_node(
                "ConstantOfShape",
                inputs=[s_shape],
                outputs=[b_idx_2d],
                value=helper.make_tensor("val", TensorProto.FLOAT, [1], [0.0]),
                name="Guideline_CreateBatchIdx",
            )
        )

        # 4) Rescale boxes from training imgsz space to ONNX input-space.
        b_scaled = "Guideline_B_Rescaled"
        in_shape = "Guideline_InputShape"
        graph.node.append(helper.make_node("Shape", inputs=[input_name], outputs=[in_shape], name="Guideline_GetInShape"))

        h_idx = "Guideline_HIdx"
        w_idx = "Guideline_WIdx"
        graph.initializer.extend([
            numpy_helper.from_array(np.array(2, dtype=np.int64), name=h_idx),
            numpy_helper.from_array(np.array(3, dtype=np.int64), name=w_idx),
        ])

        h_val, w_val = "Guideline_HVal", "Guideline_WVal"
        graph.node.append(helper.make_node("Gather", inputs=[in_shape, h_idx], outputs=[h_val], axis=0, name="Guideline_GatherH"))
        graph.node.append(helper.make_node("Gather", inputs=[in_shape, w_idx], outputs=[w_val], axis=0, name="Guideline_GatherW"))

        h_f32, w_f32 = "Guideline_Hf32", "Guideline_Wf32"
        graph.node.append(helper.make_node("Cast", inputs=[h_val], outputs=[h_f32], to=TensorProto.FLOAT, name="Guideline_CastH"))
        graph.node.append(helper.make_node("Cast", inputs=[w_val], outputs=[w_f32], to=TensorProto.FLOAT, name="Guideline_CastW"))

        train_imgsz = "Guideline_TrainImgsz"
        graph.initializer.append(numpy_helper.from_array(np.array([float(imgsz)], dtype=np.float32), name=train_imgsz))
        h_scale, w_scale = "Guideline_HScale", "Guideline_WScale"
        graph.node.append(helper.make_node("Div", inputs=[h_f32, train_imgsz], outputs=[h_scale], name="Guideline_DivH"))
        graph.node.append(helper.make_node("Div", inputs=[w_f32, train_imgsz], outputs=[w_scale], name="Guideline_DivW"))

        scales = "Guideline_BoxScales"
        graph.node.append(helper.make_node("Concat", inputs=[w_scale, h_scale, w_scale, h_scale], outputs=[scales], axis=0, name="Guideline_ConcatScales"))
        graph.node.append(helper.make_node("Mul", inputs=[b_raw, scales], outputs=[b_scaled], name="Guideline_RescaleBoxes"))

        # 5) Final consolidated output [N, 7]
        graph.node.append(
            helper.make_node(
                "Concat",
                inputs=[b_scaled, s_raw, c_raw, b_idx_2d],
                outputs=["detections"],
                axis=1,
                name="Guideline_FinalConcat",
            )
        )

        del graph.output[:]
        graph.output.append(helper.make_tensor_value_info("detections", TensorProto.FLOAT, ["N", 7]))

    # Identify primary output
    if len(graph.output) >= 1:
        out_name = graph.output[0].name
        try:
            shape = [d.dim_value if d.dim_value > 0 else 0 for d in graph.output[0].type.tensor_type.shape.dim]
        except:
            shape = []

        if len(shape) == 3 and shape[2] == 6:
            # Matches YOLOv10/v11-NMS or Consolidated output
            log.info("Aligning consolidated/NMS output with guidelines.")
            _add_guideline_nodes(out_name)
        else:
            # Fallback or Raw format (not yet fully aligned for [N, 7] guidelines)
            log.warning("Output format %s not recognized for automatic [N, 7] alignment; renaming.", shape)
            for out in graph.output:
                if "box" in out.name.lower(): out.name = "det_boxes"
                elif "score" in out.name.lower(): out.name = "det_scores"

    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    log.info("Guideline-aligned ONNX produced at %s", onnx_path)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.ultralytics_api_key:
        # Ultralytics uses this env var in recent versions for HUB access.
        os.environ["ULTRALYTICS_API_KEY"] = args.ultralytics_api_key

    # Ensure project is absolute to avoid Ultralytics prepending 'runs/detect' to relative paths.
    args.project = str(Path(args.project).resolve())
    log.info("Using project directory: %s", args.project)

    if args.use_coco:
        data_yaml = prepare_coco2017_yolo_dataset(
            coco_root=args.coco_root,
            out_root=args.dataset_out,
            imgsz=args.imgsz,
            min_box_size=args.min_box_size,
            write_empty_labels=args.write_empty_labels,
            drop_became_empty=args.drop_became_empty,
            letterbox_filter=args.letterbox_filter,
            seed=args.seed,
            force=args.force_dataset_prep,
        )
    else:
        # Use custom dataset source (matching picodet_v5_qat behavior)
        data_yaml = prepare_custom_yolo_dataset(
            data_root=args.data_root,
            ann_path=args.ann,
            out_root=args.dataset_out,
            imgsz=args.imgsz,
            min_box_size=args.min_box_size,
            val_pct=args.val_pct,
            write_empty_labels=args.write_empty_labels,
            drop_became_empty=args.drop_became_empty,
            letterbox_filter=args.letterbox_filter,
            seed=args.seed,
            force=args.force_dataset_prep,
        )

    if args.skip_train:
        if args.weights_pt is None:
            log.error("--skip_train requires --weights_pt")
            return 2
        weights_pt = args.weights_pt
    else:
        weights_pt, run_dir = train_ultralytics_detector(
            data_yaml=data_yaml,
            yolo_weights=args.yolo_weights,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            seed=args.seed,
        )
        # Log training history to epoch_history.csv to match picodet_v5_qat behavior.
        ultra_results = run_dir / "results.csv"
        if ultra_results.exists():
            try:
                import pandas as pd
                df = pd.read_csv(ultra_results)
                # Simple mapping: train/loss -> train_loss, metrics/mAP50(B) -> iou_at_50
                # Just keeping it compatible enough for the training plots if needed.
                df.columns = [c.strip() for c in df.columns]
                mapping = {
                    "train/box_loss": "train_loss", # Not perfect but gives a trend
                    "metrics/mAP50(B)": "iou_at_50",
                    "metrics/precision(B)": "pr_at_50",
                    "metrics/recall(B)": "rec_at_50",
                }
                for u_col, p_col in mapping.items():
                    if u_col in df.columns:
                        df[p_col] = df[u_col]
                df.to_csv("epoch_history.csv", index=False)
                log.info("Wrote training history to epoch_history.csv")
            except Exception as e:
                log.warning("Could not write epoch_history.csv: %s", e)

    onnx_fp32 = export_onnx_with_ultralytics(
        weights_pt=weights_pt,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        nms=args.nms,
        max_det=args.max_det,
        device=args.device,
        out_dir=args.export_dir,
    )
    _print_onnx_info(onnx_fp32, "[Diagnostics] Post-Export: ")

    onnx_for_quant = onnx_fp32
    if args.embed_preprocess:
        onnx_pre = args.export_dir / (onnx_fp32.stem + "_pre_u8.onnx")
        onnx_for_quant = embed_uint8_preprocess_into_onnx(
            in_onnx=onnx_fp32,
            out_onnx=onnx_pre,
            imgsz=args.imgsz,
            resize=args.embed_resize,
        )
        _print_onnx_info(onnx_for_quant, "[Diagnostics] Post-Preprocess: ")

    # Align with GUIDELINES (rename, consolidate [N, 7], rescale coordinates)
    fix_onnx_outputs_to_match_pico(onnx_path=onnx_for_quant, nms=args.nms, imgsz=args.imgsz)
    _print_onnx_info(onnx_for_quant, "[Diagnostics] Post-Guideline-Fix: ")

    if args.quantize:
        split_dir = args.dataset_out / "images" / args.calib_split
        calib_imgs = _collect_calib_images(split_dir, n=args.calib_images, seed=args.seed)
        onnx_int8 = args.export_dir / (onnx_for_quant.stem + "_int8.onnx")
        quantized = quantize_onnx_int8_static(
            in_onnx=onnx_for_quant,
            out_onnx=onnx_int8,
            calib_images=calib_imgs,
            imgsz=args.imgsz,
            expects_uint8_input=args.embed_preprocess,
            use_letterbox_for_float_input=True,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
        )
        if args.ort_optimize:
            ort_optimize_onnx(
                in_onnx=quantized,
                out_onnx=args.export_dir / (quantized.stem + "_opt.onnx"),
            )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    # raise SystemExit(main(sys.argv[1:]))
    main(sys.argv[1:])
