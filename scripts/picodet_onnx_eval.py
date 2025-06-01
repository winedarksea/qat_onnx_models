#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
picodet_onnx_eval.py
───────────────────────────────────────────────────────────────────────────────
Evaluate `picodet_int8.onnx` exported by the training/QAT pipeline.

Outputs
  • mean IoU on the chosen validation set
  • mean wall-clock inference time / image
  • (optional) three example images with predicted boxes drawn

Author: <you>
"""

from __future__ import annotations
import argparse, os, random, time, warnings
from pathlib import Path
from typing import Dict, Tuple

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

# ───────────────────────── constants ──────────────────────────
SEED              = 42
IMG_SIZE          = 256            # must match ResizeNorm in exported model
VAL_BATCH_SIZE    = 8
NUM_WORKERS       = 0
DEFAULT_MODEL     = "picodet_int8.onnx"
RAND_EXAMPLES     = 5
DTYPE_EXPECTED    = np.uint8       # model input dtype

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
             coco_api,
             coco2contig: Dict[int, int]) -> Tuple[float, float]:
    """
    Returns (mean_IoU, mean_time_per_image_seconds)
    """
    inp_name   = session.get_inputs()[0].name
    out_names  = [o.name for o in session.get_outputs()]
    binding_ok = hasattr(session, "io_binding") \
        and session.get_providers()[0] != "CPUExecutionProvider"

    tot_iou, tot_imgs_with_gt, elapsed = 0.0, 0, 0.0
    io_binding = session.io_binding() if binding_ok else None

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

        # -- per-image IoU --
        for bi in range(imgs_u8.size(0)):
            # scale GT boxes to 128×128
            scaled_gt = []
            if annots[bi]:
                img_id  = annots[bi][0]["image_id"]
                w0, h0  = coco_api.imgs[img_id]["width"], coco_api.imgs[img_id]["height"]
                sx, sy  = IMG_SIZE / w0, IMG_SIZE / h0
                for a in annots[bi]:
                    x, y, w, h = a["bbox"]
                    scaled_gt.append([
                        x * sx, y * sy,
                        (x + w) * sx, (y + h) * sy
                    ])
            if not scaled_gt:
                continue

            gt_t      = torch.tensor(scaled_gt)
            mask_pred = (bidx_np == bi)
            if not mask_pred.any():   # no preds for this image
                continue
            pred_t = torch.tensor(boxes_np[mask_pred])

            iou = tvops.box_iou(pred_t, gt_t)
            if iou.numel():
                tot_iou += iou.max(dim=0).values.mean().item()
                tot_imgs_with_gt += 1

    mean_iou = tot_iou / tot_imgs_with_gt if tot_imgs_with_gt else 0.0
    mean_t   = elapsed / len(loader.dataset)
    return mean_iou, mean_t

# ───────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_MODEL, help="model file")
    ap.add_argument("--coco_root", default="coco")
    ap.add_argument("--provider", default=None,
                    help="Execution provider override (CUDAExecutionProvider, CPUExecutionProvider, …)")
    ap.add_argument("--subset", type=int, default=800,
                    help="evaluate on N random val images (speed)")
    ap.add_argument("--plot", default=True, action="store_true",
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
    miou, t_img = evaluate(sess, loader, coco_api, coco2contig)
    print("\n───────── Results ─────────")
    print(f"Validation images      : {len(loader.dataset)}")
    print(f"Mean IoU               : {miou*100:5.2f} %")
    print(f"Avg inference / image  : {t_img*1e3:6.2f} ms "
          f"({1.0/t_img if t_img else 0:7.2f} img/s)")
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
                             contig2name, score_th=0.3)
            fig.suptitle(f"val image #{si}", fontsize=8)
        plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
