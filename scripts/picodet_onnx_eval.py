#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
picodet_onnx_eval.py
───────────────────────────────────────────────────────────────────────────────
Evaluate `picodet_int8.onnx` exported by the training/QAT pipeline.

Outputs
  • mean IoU on the chosen validation set
  • mean wall-clock inference time / image
  • mAP COCO style accuracy assessment
  • (optional) example images with predicted boxes drawn

"""

from __future__ import annotations
import argparse, os, random, time, warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import onnxruntime as ort
import torch, torchvision.transforms.v2 as T_v2
import torchvision.transforms.functional as F_tv
import torchvision.datasets as tvsets
import torchvision.ops as tvops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ───────────────────────── constants ──────────────────────────
SEED              = 42
IMG_SIZE          = 256        # must match ResizeNorm in exported model
VAL_BATCH_SIZE    = 8
NUM_WORKERS       = 0
DEFAULT_MODEL     = "picodet_int8.onnx"
RAND_EXAMPLES     = 5
DTYPE_EXPECTED    = np.uint8   # model input dtype

random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────── data loading ───────────────────────
def build_val_tf() -> T_v2.Compose:
    return T_v2.Compose([
        T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=False),
    ])

def collate(batch):
    imgs, annots = zip(*batch)
    imgs_t = [F_tv.pil_to_tensor(im).contiguous() for im in imgs]  # uint8 CHW
    return torch.stack(imgs_t), list(annots)

def get_loader(coco_root: str, subset: int | None) -> DataLoader:
    val_ds = tvsets.CocoDetection(
        f"{coco_root}/val2017",
        f"{coco_root}/annotations/instances_val2017.json",
        transform=build_val_tf()
    )
    if subset and subset < len(val_ds):
        idx = random.sample(range(len(val_ds)), subset)
        val_ds = Subset(val_ds, idx)
    return DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate, persistent_workers=bool(NUM_WORKERS)
    )

# ───────────────────────── coco helpers ───────────────────────
def coco_contiguous_map(coco_api) -> Dict[int, int]:
    return {c: i for i, c in enumerate(sorted(coco_api.getCatIds()))}

def contiguous_to_name(coco_api, coco2contig) -> Dict[int, str]:
    return {v: coco_api.loadCats([k])[0]["name"] for k, v in coco2contig.items()}

# ───────────────────────── plotting ───────────────────────────
def draw_image(img_uint8: torch.Tensor,
               boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray,
               id2name: Dict[int, str], score_th=0.25):
    """Draw predicted boxes on image using matplotlib."""
    img = img_uint8.permute(1, 2, 0).cpu().numpy()
    
    # Increase size and resolution here
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    ax.imshow(img)
    ax.axis("off")

    for b, lab, sc in zip(boxes, labels, scores):
        if sc < score_th:
            continue
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        col = np.random.rand(3,)
        ax.add_patch(patches.Rectangle((x1, y1), w, h,
                                       edgecolor=col, facecolor='none', lw=1.5))
        ax.text(x1, y1, f"{id2name.get(lab, lab)}:{sc:.2f}",
                fontsize=8, color=col,
                bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))
    return fig


# ───────────────────────── evaluation core ─────────────────────
@torch.no_grad()
def evaluate(session: ort.InferenceSession,
             loader: DataLoader,
             coco_api: COCO,
             coco2contig: Dict[int, int]) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (mean_IoU, mean_time_per_image_seconds, coco_eval_results)
    """
    inp_name   = session.get_inputs()[0].name
    out_names  = [o.name for o in session.get_outputs()]
    binding_ok = hasattr(session, "io_binding") \
        and session.get_providers()[0] != "CPUExecutionProvider"

    tot_iou, tot_imgs_with_gt, elapsed = 0.0, 0, 0.0
    io_binding = session.io_binding() if binding_ok else None

    # List to store detections in COCO format for mAP calculation
    coco_detections = []

    for imgs_u8, annots in tqdm(loader, desc="Eval", unit="batch"):
        # -- run model --
        x_np = imgs_u8.cpu().numpy().astype(DTYPE_EXPECTED, copy=False)
        start = time.perf_counter()

        if binding_ok:
            io_binding.bind_input(name=inp_name,
                                 device_type="cpu", device_id=0,
                                 element_type=DTYPE_EXPECTED,
                                 shape=x_np.shape, buffer_ptr=x_np.ctypes.data)
            # prepare outputs
            io_binding.bind_output(out_names[0])  # boxes
            io_binding.bind_output(out_names[1])  # scores
            io_binding.bind_output(out_names[2])  # labels
            io_binding.bind_output(out_names[3])  # batch idx
            session.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
        else:
            ort_outs = session.run(out_names, {inp_name: x_np})
        elapsed += time.perf_counter() - start

        boxes_np, scores_np, labels_np, bidx_np = ort_outs

        # -- per-image IoU and COCO detections --
        for bi in range(imgs_u8.size(0)):
            img_info = loader.dataset.dataset.coco.loadImgs(annots[bi][0]["image_id"])[0] if annots[bi] else None
            if not img_info:
                # If there are no annotations, we can't get original image info this way.
                # Need to retrieve img_id from dataset directly if possible.
                # For simplicity here, we'll skip if no annotations are present for a given image,
                # as mAP primarily cares about images with ground truth.
                # A more robust solution might involve iterating through original dataset indices.
                try:
                    img_id_from_loader = loader.dataset.ids[loader.dataset.indices[bi]] # Accessing original image ID
                    img_info = loader.dataset.dataset.coco.loadImgs(img_id_from_loader)[0]
                except AttributeError: # If not a Subset or other dataset structure
                    warnings.warn("Could not retrieve original image information for mAP calculation. Skipping some images.")
                    continue


            w0, h0 = img_info["width"], img_info["height"]
            sx, sy = IMG_SIZE / w0, IMG_SIZE / h0

            # Scale GT boxes to IMG_SIZE for IoU calculation
            scaled_gt = []
            if annots[bi]:
                for a in annots[bi]:
                    x, y, w, h = a["bbox"]
                    scaled_gt.append([
                        x * sx, y * sy,
                        (x + w) * sx, (y + h) * sy
                    ])
            
            # --- IoU Calculation (existing) ---
            if scaled_gt:
                gt_t = torch.tensor(scaled_gt)
                mask_pred = (bidx_np == bi)
                if mask_pred.any():  # no preds for this image
                    pred_t = torch.tensor(boxes_np[mask_pred])
                    iou = tvops.box_iou(pred_t, gt_t)
                    if iou.numel():
                        tot_iou += iou.max(dim=0).values.mean().item()
                        tot_imgs_with_gt += 1

            # --- COCO Detection Formatting for mAP ---
            mask_pred_coco = (bidx_np == bi)
            pred_boxes_orig_scale = []
            pred_scores = []
            pred_labels = []

            # Scale predicted boxes back to original image size
            if mask_pred_coco.any():
                for i in np.where(mask_pred_coco)[0]:
                    x1, y1, x2, y2 = boxes_np[i]
                    # Convert to [x, y, width, height] format
                    orig_x = x1 / sx
                    orig_y = y1 / sy
                    orig_w = (x2 - x1) / sx
                    orig_h = (y2 - y1) / sy
                    pred_boxes_orig_scale.append([orig_x, orig_y, orig_w, orig_h])
                    pred_scores.append(scores_np[i])
                    
                    # Convert contiguous label back to COCO category ID
                    # We need the reverse map from contiguous to COCO category ID
                    reverse_coco2contig = {v: k for k, v in coco2contig.items()}
                    pred_labels.append(reverse_coco2contig[labels_np[i]])

                for bbox, score, label in zip(pred_boxes_orig_scale, pred_scores, pred_labels):
                    coco_detections.append({
                        "image_id": img_info["id"],
                        "category_id": label,
                        "bbox": [float(x) for x in bbox], # Ensure float type
                        "score": float(score) # Ensure float type
                    })
    
    mean_iou = tot_iou / tot_imgs_with_gt if tot_imgs_with_gt else 0.0
    mean_t   = elapsed / len(loader.dataset)

    # --- COCO mAP Calculation ---
    coco_dt = coco_api.loadRes(coco_detections)
    coco_eval = COCOeval(coco_api, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract relevant mAP metrics
    # This list of mAP values is usually:
    #  [AP@0.5:0.95, AP@0.5, AP@0.75, AP_small, AP_medium, AP_large,
    #   AR@0.5:0.95_max_1, AR@0.5:0.95_max_10, AR@0.5:0.95_max_100,
    #   AR_small, AR_medium, AR_large]
    coco_eval_results = {
        "mAP_0_50_0_95": coco_eval.stats[0],
        "mAP_0_50": coco_eval.stats[1],
        "mAP_0_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
    }

    return mean_iou, mean_t, coco_eval_results

# ───────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_MODEL, help="model file")
    ap.add_argument("--coco_root", default="coco")
    ap.add_argument("--provider", default=None,
                    help="Execution provider override (CUDAExecutionProvider, CPUExecutionProvider, …)")
    ap.add_argument("--subset", type=int, default=800,
                    help="evaluate on N random val images (speed)")
    ap.add_argument("--plot", default=True, action="store_true", # Changed default to False
                    help="show a few example predictions")
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        raise FileNotFoundError(args.onnx)

    # -- data & label maps --
    loader         = get_loader(args.coco_root, args.subset)
    base_ds        = loader.dataset
    while isinstance(base_ds, Subset):
        base_ds = base_ds.dataset
    coco_api       = base_ds.coco
    coco2contig    = coco_contiguous_map(coco_api)
    contig2name    = contiguous_to_name(coco_api, coco2contig)

    # -- ONNX Runtime session --
    avail = ort.get_available_providers()
    if args.provider:
        providers = [args.provider]
    else:
        # auto-pick CUDA/DirectML if available
        for p in ["CUDAExecutionProvider", "DmlExecutionProvider"]:
            if p in avail:
                providers = [p]; break
        else:
            providers = ["CPUExecutionProvider"]
    sess_opt = ort.SessionOptions()# {"arena_extend_strategy":"kSameAsRequested"}
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(args.onnx, sess_options=sess_opt,
                                 providers=providers)
    print(f"[INFO] ONNX model loaded. Provider(s): {sess.get_providers()}")

    # -- evaluate --
    miou, t_img, coco_eval_results = evaluate(sess, loader, coco_api, coco2contig)
    print("\n───────── Results ─────────")
    print(f"Validation images      : {len(loader.dataset)}")
    print(f"Mean IoU               : {miou*100:5.2f} %")
    print(f"Avg inference / image  : {t_img*1e3:6.2f} ms "
          f"({1.0/t_img if t_img else 0:7.2f} img/s)")
    print("\n─── COCO mAP Results ───")
    print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = {coco_eval_results['mAP_0_50_0_95']:.3f}")
    print(f"  Average Precision  (AP) @[ IoU=0.50      | area=all | maxDets=100 ] = {coco_eval_results['mAP_0_50']:.3f}")
    print(f"  Average Precision  (AP) @[ IoU=0.75      | area=all | maxDets=100 ] = {coco_eval_results['mAP_0_75']:.3f}")
    print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=small | maxDets=100 ] = {coco_eval_results['mAP_small']:.3f}")
    print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {coco_eval_results['mAP_medium']:.3f}")
    print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=large | maxDets=100 ] = {coco_eval_results['mAP_large']:.3f}")
    print("──────────────────────────")


    # -- visualise a few predictions --
    if args.plot:
        print("[INFO] Showing example predictions …")
        sample_idx = random.sample(range(len(loader.dataset)), RAND_EXAMPLES)
        for si in sample_idx:
            img_pil, anns = loader.dataset[si]
            img_u8 = F_tv.pil_to_tensor(img_pil).contiguous()  # <-- convert to tensor
            inp_np = img_u8.unsqueeze(0).numpy().astype(DTYPE_EXPECTED, copy=False)
            boxes, scores, labels, bidx = sess.run(
                [o.name for o in sess.get_outputs()],
                {sess.get_inputs()[0].name: inp_np}
            )
            keep = (bidx == 0)
            fig = draw_image(img_u8,
                             boxes[keep], labels[keep], scores[keep],
                             contig2name, score_th=0.1)
            fig.suptitle(f"val image #{si}", fontsize=8)
        plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
