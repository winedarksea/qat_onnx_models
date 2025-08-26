# train_picodet_qat.py – minimal pipeline: COCO ➜ FP32 ➜ QAT ➜ INT8 ➜ ONNX (with NMS)
# built on pytorch version 2.7
from __future__ import annotations
import argparse, random, time, warnings, math, copy, os, json
import functools
from typing import List, Tuple
import traceback
import numpy as np

import torch, torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.datasets import CocoDetection
from torchvision.tv_tensors import BoundingBoxes
import torchvision.ops as tvops
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F

import onnx
from onnx import TensorProto as TP, helper as oh

from pycocotools.coco import COCO

if False:  # this is just here for a quirk in testing, imports are still available when run
    try: 
        from picodet_lib_v2 import (
            PicoDet, get_backbone, VarifocalLoss, dfl_loss,
            build_dfl_targets, PicoDetHead, ResizeNorm, QualityFocalLoss
        )
    except Exception:
        # when running manually in console, these are already loaded
        pass

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 256  # PicoDet’s anchors assume stride-divisible sizes. Divisible by 32

# ───────────────────── data & transforms ───────────────────────
# COCO’s official 80-class list (order matters!)
CANONICAL_COCO80_IDS: list[int] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

# { COCO-category-id  →  contiguous-id 0-79 }
CANONICAL_COCO80_MAP: dict[int, int] = {
    coco_id: i for i, coco_id in enumerate(CANONICAL_COCO80_IDS)
}


def contiguous_id_to_name(coco_api: COCO) -> dict[int, str]:
    """Return {0-79 → class-name} using the official 80-class list."""
    return {
        i: coco_api.loadCats([coco_id])[0]["name"]
        for i, coco_id in enumerate(CANONICAL_COCO80_IDS)
    }

# ---------- COCO → tv_tensors utility ----------
def _coco_to_tvt(annots, lb_map, canvas):
    boxes, labels = [], []
    W, H = canvas                                    # PIL size is (W,H)
    for a in annots:
        if a.get("iscrowd", 0):
            continue
        cid = a["category_id"]
        if cid not in lb_map:                        # skip classes >79
            continue
        x, y, w, h = a["bbox"]                       # COCO XYWH
        boxes.append([x, y, x + w, y + h])
        labels.append(lb_map[cid])

    if not boxes:                                    # keep training stable
        boxes  = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,),  dtype=torch.int64)

    bbx = BoundingBoxes(
        torch.as_tensor(boxes, dtype=torch.float32),
        format="XYXY", canvas_size=(H, W)
    )
    return {"boxes": bbx, "labels": torch.as_tensor(labels, dtype=torch.int64)}

# ---------- dataset wrapper ----------
class CocoDetectionV2(CocoDetection):
    """COCO dataset ready for torchvision-v2 transforms."""
    def __init__(self, img_dir, ann_file, lb_map, transforms=None):
        super().__init__(img_dir, ann_file)           # PIL, list[dict]
        self.lb_map = lb_map
        self._tf    = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        tgt = _coco_to_tvt(anns, self.lb_map, img.size)
        if self._tf is not None:                      # v2 ops work on (img,tgt)
            img, tgt = self._tf(img, tgt)
        return img, tgt

def build_label_map_and_names_from_ann(ann_path: str):
    with open(ann_path, 'r') as f:
        d = json.load(f)
    # Keep a stable order; your file uses ids 1:person, 2:snake, 3:spider
    cats = sorted(d['categories'], key=lambda c: c['id'])
    catid2contig = {c['id']: i for i, c in enumerate(cats)}   # {1:0, 2:1, 3:2}
    id2name      = {i: c['name'] for i, c in enumerate(cats)} # {0:'person',1:'snake',2:'spider'}
    return catid2contig, id2name

# ---------- transforms ----------
def build_transforms(size, train):
    aug = [
        T.ToImage(),
        T.RandomHorizontalFlip(0.25) if train else T.Identity(),
        T.RandomPhotometricDistort(p=0.2) if train else T.Identity(),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.05) if train else T.Identity(),
        T.RandomAutocontrast(p=0.1) if train else T.Identity(),
        T.RandomEqualize(p=0.1) if train else T.Identity(),
        # T.Resize(size, antialias=True),
        T.RandomResizedCrop(size, scale=(0.75, 1.05), antialias=True) if train else T.Resize(size, antialias=True),
        T.ToDtype(torch.uint8, scale=True),  # keep uint8, model normalises
    ]
    return T.Compose(aug)


def _make_valid_mask(xyxy, H, W,
                     min_wh      = 8,
                     max_cover   = 0.90,
                     ar_min      = 0.05,
                     ar_max      = 20.0,
                     allow_neg   = False):
    """
    Return a Bool mask identifying annotation boxes that pass all sanity checks.
    xyxy : (N,4) tensor, float
    H,W  : image height/width   (after v2 transforms _before_ ResizeNorm)
    """

    x1,y1,x2,y2 = xyxy.T
    w  = x2 - x1
    h  = y2 - y1
    ar = w / (h + 1e-6)

    # 1. positive dims
    pos_wh = (w > 0) & (h > 0) if allow_neg else torch.ones_like(w, dtype=torch.bool)

    # 2. reasonable size
    big = (w * h) > max_cover * H * W         # covers nearly full canvas
    tiny = (w < min_wh) | (h < min_wh)

    # 3. aspect ratio
    weird = (ar < ar_min) | (ar > ar_max)

    # 4. inside canvas
    inside = (x1 >= -1) & (y1 >= -1) & (x2 <= W+1) & (y2 <= H+1)

    return pos_wh & inside & ~tiny & ~big & ~weird


def collate_and_filter_more(
    batch: List[Tuple[torch.Tensor, dict]],
    *,
    min_box_size: int = 6,
    keep_original_negatives: bool = True,
    drop_became_empty: bool = True,
    # logging / debugging
    log_every: int = 2000,      # << fairly low verbosity by default
    max_stats_lines: int = 10,  # stop printing after this many messages
) -> Tuple[torch.Tensor, List[dict]] | Tuple[None, None]:
    """
    A collate_fn that:
      • sanitises boxes,
      • ALWAYS keeps images that started with 0 GT boxes (true negatives),
      • optionally drops images that had GT but lost all boxes after filtering,
      • returns (None, None) if literally nothing is left (very rare).

    Args
    ----
    min_box_size : int
        Minimum side (in *current* image pixels) for a GT box to be kept.
    keep_original_negatives : bool
        If True, images that *arrived* with 0 GTs are kept as negatives.
    drop_became_empty : bool
        If True, images that had boxes but all were filtered out are dropped.
        Set False if you also want to keep them as hard negatives.
    log_every : int
        Print running stats every N batches.
    max_stats_lines : int
        Stop printing stats after this many lines (keeps logs tidy).

    Returns
    -------
    (images, targets) or (None, None) if nothing survived.
    """

    imgs, tgts = zip(*batch)

    # persistent running stats
    if not hasattr(collate_and_filter, "stats"):
        collate_and_filter.stats = {
            "batches": 0,
            "boxes_in": 0,
            "boxes_out": 0,
            "imgs_in": 0,
            "imgs_out": 0,
            "printed": 0,
            "kept_neg": 0,
            "dropped_empty": 0,
        }
    S = collate_and_filter.stats

    good_imgs: List[torch.Tensor] = []
    good_tgts: List[dict] = []

    for img, tgt in zip(imgs, tgts):
        H, W = img.shape[-2:]
        boxes_in = tgt["boxes"]
        n_in = int(boxes_in.shape[0])
        started_negative = n_in == 0

        # Always count input stats
        S["imgs_in"] += 1
        S["boxes_in"] += n_in

        if n_in > 0:
            xyxy = boxes_in.clone().to(torch.float32)
            mask = _make_valid_mask(
                xyxy, H, W,
                min_wh=min_box_size,
                allow_neg=False
            )
            if mask.sum() > 0:
                tgt["boxes"] = boxes_in[mask]
                tgt["labels"] = tgt["labels"][mask]
            else:
                tgt["boxes"] = boxes_in.new_zeros((0, 4))
                tgt["labels"] = tgt["labels"].new_zeros((0,), dtype=torch.long)

        n_out = int(tgt["boxes"].shape[0])

        keep = True
        if started_negative and keep_original_negatives:
            # keep true negatives
            S["kept_neg"] += 1
            keep = True
        elif n_out == 0 and n_in > 0 and drop_became_empty:
            # had GT but became empty → drop (configurable)
            S["dropped_empty"] += 1
            keep = False

        if keep:
            good_imgs.append(img)
            good_tgts.append(tgt)
            S["imgs_out"] += 1
            S["boxes_out"] += n_out

    S["batches"] += 1
    if log_every > 0 and (S["batches"] % log_every == 0) and (S["printed"] < max_stats_lines):
        kept_pct = 100.0 * S["boxes_out"] / max(1, S["boxes_in"])
        print(
            f"[SanityFilter] b={S['batches']:>6} | "
            f"boxes kept {kept_pct:5.1f}% ({S['boxes_out']}/{S['boxes_in']}) | "
            f"imgs kept {S['imgs_out']}/{S['imgs_in']} | "
            f"true-neg kept {S['kept_neg']} | "
            f'dropped-empty {S["dropped_empty"]}'
        )
        S["printed"] += 1

    if not good_imgs:   # pathological but possible
        return None, None

    return torch.stack(good_imgs, 0), good_tgts


# ---------- collate ----------
def collate_v2(batch):
    imgs, tgts = zip(*batch)
    return torch.stack(imgs, 0), list(tgts)

def collate_and_filter(batch: list, min_box_size: int = 1):
    """
    A custom collate_fn that filters out small bounding boxes and then
    removes any images from the batch that no longer have annotations.

    Args:
        batch: A list of (image, target) tuples from the dataset.
        min_box_size: The minimum width and height a box must have to be kept.
                      A value of 1 or 0 effectively disables filtering.

    Returns:
        A tuple of (images_tensor, targets_list) or (None, None) if the
        entire batch is filtered out.
    """
    filtered_imgs = []
    filtered_tgts = []

    for img, tgt in batch:
        boxes = tgt.get("boxes")

        # If there are no boxes to begin with, it's a valid negative sample. Keep it.
        if boxes is None or boxes.numel() == 0:
            filtered_imgs.append(img)
            filtered_tgts.append(tgt)
            continue

        # Get box dimensions (works with torchvision BoundingBoxes object)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        # Create a mask to keep only boxes that meet the size criteria
        keep_mask = (widths >= min_box_size) & (heights >= min_box_size)

        # Apply the mask to boxes and their corresponding labels
        tgt["boxes"] = boxes[keep_mask]
        tgt["labels"] = tgt["labels"][keep_mask]

        # IMPORTANT: Only keep the image if it still has annotations after filtering
        if tgt["boxes"].numel() > 0:
            filtered_imgs.append(img)
            filtered_tgts.append(tgt)
        # If the image has no boxes left, it is implicitly dropped from the batch.

    # If the filtering process removed every single sample from the batch
    if not filtered_imgs:
        return None, None

    # Proceed with standard collation for the kept samples
    return torch.stack(filtered_imgs, 0), filtered_tgts

# ───────────────────── assigner (SimOTA, cached) ────────────────
class SimOTACache:
    def __init__(self,
                 nc: int,
                 ctr: float = 2.5,
                 topk: int = 10,
                 cls_cost_weight: float = 1.0,
                 debug_epochs: int = 0,         # 0 = silent
                 dynamic_k_min: int = 2,
                 ):
        self.nc = nc
        self.r = ctr
        self.k = topk
        self.cls_cost_weight = cls_cost_weight
        self.cache = {}               # anchor centres per (H,W,stride,device)
        self.simota_prefilter = False
        self.dynamic_k_min = dynamic_k_min  # 3, or 1-2
        self.power = 0.0 
        self.mean_fg_iou = 0.0

        # --- State for debugging system ---
        self._dbg_mod  = debug_epochs
        self.is_debug_mode = bool(debug_epochs)
        self._dbg_iter = 0
        self._debug_stats = {
            "ctr_hit_rate": [], # Stores the % of candidates that pass the center radius check
            "k_min_hits": [],   # Stores (num_hits, total_gt) for dynamic_k hitting the floor
            "k_dist": [],       # Stores the full distribution of dynamic_k values
        }
        self._classification_debug_stats = {
            "cls_cost_magnitude": [], # Avg. magnitude of the classification cost term
            "loc_cost_magnitude": [], # Avg. magnitude of the localization cost term
            "assignment_switches": [], # How often cls_cost changed the winning anchor
        }

    def print_classification_debug_report(self):
        if not self._classification_debug_stats["cls_cost_magnitude"]:
            return
    
        avg_cls_cost = np.mean(self._classification_debug_stats["cls_cost_magnitude"])
        avg_loc_cost = np.mean(self._classification_debug_stats["loc_cost_magnitude"])
        total_switches, total_anchors = map(sum, zip(*self._classification_debug_stats["assignment_switches"]))
        switch_rate = (total_switches / total_anchors) * 100 if total_anchors > 0 else 0
    
        print("\n" + "─" * 10 + " Classification Debug Report " + "─" * 10)
        print(f"  Cost Magnitudes: Avg. Cls Cost = {avg_cls_cost:.3f}, Avg. Loc Cost = {avg_loc_cost:.3f}")
        if avg_loc_cost > avg_cls_cost * 10:
            print(f"  [ACTION] Loc cost is >> Cls cost. Consider INCREASING `cls_cost_weight` (currently {self.cls_cost_weight}).")
        print(f"  Assignment Switch Rate: {switch_rate:.2f}% (Rate classification cost changed the assignment from IoU)")
        if switch_rate < 2:
            print("  May need to increase cls_cost_weight")
        elif switch_rate > 20:
            print("  May need to decrease cls_cost_weight")
        print("─" * 40 + "\n")
    
        # Reset stats
        for k in self._classification_debug_stats:
            self._classification_debug_stats[k].clear()
    
    def print_debug_report(self):
        """Calculates and prints the aggregated statistics, then resets.
        Tuning ctr: Look at the "Avg. Center Radius Hit Rate".
            If this value is very low (e.g., < 5%), your ctr might be too small, starving the assigner of good candidates before it even considers IoU. Consider increasing ctr.
            If it's very high (e.g., > 40-50%), ctr might be too large, leading to inefficient cost matrix calculations. You could consider decreasing it slightly. A value between 10-25% is often a healthy range.
        Tuning dynamic_k_min: Look at the "% of GTs using min_k".
            If this is high (> 80%), like in the example above, it means the model is struggling. It's not finding enough high-IoU anchors, so it's constantly being "forced" to assign the minimum number. During the bootstrap phase, this is okay, but it's a clear sign you should not reduce dynamic_k_min yet.
            If this is low (< 20%), it means the model is healthy and confident. It's finding many good anchors, and dynamic_k is naturally floating above the minimum. This is a strong signal that you can safely enter the "refinement" phase by decreasing dynamic_k_min (e.g., to 2) to get higher-quality matches.
        """
        if not self._debug_stats["ctr_hit_rate"]:
            return # Nothing to report

        # --- Metric 1: Center Radius Hit Rate ---
        # Helps tune `ctr`. If too low (<5%), `ctr` might be too small.
        # If too high (>40-50%), `ctr` might be too loose and inefficient.
        avg_ctr_hit_rate = np.mean(self._debug_stats["ctr_hit_rate"])

        # --- Metric 2: Percentage of GTs using the minimum k ---
        # The single most important metric for tuning `dynamic_k_min`.
        # If high (>75%), the model is struggling and you should consider INCREASING dynamic_k_min.
        # If low (<20%), the model is healthy and you can consider DECREASING dynamic_k_min for refinement.
        total_k_min_hits = sum(h for h, t in self._debug_stats["k_min_hits"])
        total_gt_boxes = sum(t for h, t in self._debug_stats["k_min_hits"])
        percent_gt_at_min_k = (total_k_min_hits / total_gt_boxes) * 100 if total_gt_boxes > 0 else 0

        # --- Metric 3: Full dynamic_k Distribution ---
        # Gives a much richer view than just the mean.
        all_k_vals = torch.cat(self._debug_stats["k_dist"]).float()
        k_mean = all_k_vals.mean().item()
        k_std = all_k_vals.std().item()
        k_median = all_k_vals.median().item()
        k_q25 = torch.quantile(all_k_vals, 0.25).item()
        k_q75 = torch.quantile(all_k_vals, 0.75).item()

        # --- Print the formatted report ---
        print("\n" + "─" * 10 + f" SimOTA Report (iter {self._dbg_iter}) " + "─" * 10)
        print(f"  [ctr={self.r:.1f}] Avg. Center Radius Hit Rate: {avg_ctr_hit_rate:.2%}")
        print(f"  [dynamic_k_min={self.dynamic_k_min}] % of GTs using min_k: {percent_gt_at_min_k:.1f}%")
        print(f"  Dynamic 'k' Stats: mean={k_mean:.2f}, std={k_std:.2f}, median={k_median:.1f}, q25-q75=[{k_q25:.1f}-{k_q75:.1f}]")
        print("─" * 40 + "\n")

        # --- Reset stats for the next interval ---
        for k in self._debug_stats:
            self._debug_stats[k].clear()

    @torch.no_grad()
    def __call__(self,
                 f_shapes,               # List[(H,W,stride)]
                 device: torch.device,  # Use a consistent device key string in order to avoid a cache miss
                 tgt: dict,              # {"boxes":(M,4), "labels":(M,)}
                 cls_logits: torch.Tensor, # Raw class logits (A, C) from model
                 # obj_logits: torch.Tensor | None = None,
                 model_head = None,
                 ):

        # 0. early-exit if no GT
        gt_boxes = tgt['boxes']
        gt_labels = tgt['labels']
        M = gt_boxes.size(0)
        if M == 0:
            A = sum(H * W for H, W, _ in f_shapes)
            empty_bool = torch.zeros(A, dtype=torch.bool,  device=device)
            return empty_bool,            \
                   torch.full((A,), -1,   dtype=torch.long,  device=device), \
                   torch.zeros((A, 4),    dtype=torch.float32, device=device), \
                   torch.zeros((A,),      dtype=torch.float32, device=device)

        # 1. collect anchor centres & strides (with cache)
        
        anchor_centers = torch.cat(
            [getattr(model_head, f'anchor_points_level_{i}') for i in range(model_head.nl)],
        0).to(device)
        
        anchor_strides = torch.cat([
            model_head.strides_buffer[i].expand(getattr(model_head, f'anchor_points_level_{i}').size(0))
            for i in range(model_head.nl)
        ], 0).to(device)

        A = anchor_centers.size(0)

        # 2. IoU and centre-radius mask
        s_div_2 = anchor_strides.unsqueeze(1) / 2.0
        anchor_candidate_boxes = torch.cat([anchor_centers - s_div_2,
                                            anchor_centers + s_div_2], 1)
        iou = tvops.box_iou(gt_boxes, anchor_candidate_boxes)     # (M,A)
        if iou.max() <= 0:
            print("SimOTA: No overlap found between any ground-truth box and candidate anchors")
            # Return empty/background tensors, matching the function's output signature
            A = anchor_centers.size(0)
            device = anchor_centers.device
            empty_bool = torch.zeros(A, dtype=torch.bool,  device=device)
            return empty_bool,            \
                   torch.full((A,), -1,   dtype=torch.long,  device=device), \
                   torch.zeros((A, 4),    dtype=torch.float32, device=device), \
                   torch.zeros((A,),      dtype=torch.float32, device=device)

        gt_centres = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
        dist = (anchor_centers.unsqueeze(1)
                - gt_centres.unsqueeze(0)).abs().max(dim=-1).values   # (A,M)
        centre_mask = dist < (self.r * anchor_strides.unsqueeze(1))   # (A,M)
        if self.simota_prefilter:
            # add inside mask filter (this needs further debugging as it removes too many boxes)
            inside_mask = (
                (anchor_centers[:, 0].unsqueeze(1) >= gt_boxes[:, 0]) &
                (anchor_centers[:, 0].unsqueeze(1) <= gt_boxes[:, 2]) &
                (anchor_centers[:, 1].unsqueeze(1) >= gt_boxes[:, 1]) &
                (anchor_centers[:, 1].unsqueeze(1) <= gt_boxes[:, 3])
            )                                                            # (A, M)
            
            centre_mask = centre_mask & inside_mask   # keep only anchors whose centre lies inside bbox

        # 3. dynamic-k candidates
        iou_sum_per_gt = iou.sum(1)
        dynamic_ks = torch.clamp(iou_sum_per_gt.ceil().int(),  # ceil not floor
                                 min=self.dynamic_k_min, max=self.k)
        fg_cand_mask = torch.zeros(A, dtype=torch.bool, device=device)

        for g in range(M):
            valid_idx = centre_mask[:, g].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel():
                ious_local = iou[g, valid_idx]
                k = min(dynamic_ks[g].item(), valid_idx.numel())
                topk_idx = torch.topk(ious_local, k).indices
                fg_cand_mask[valid_idx[topk_idx]] = True

        eps = 1e-8

        num_anchors_total = cls_logits.size(0)
        idx_anchor_dim = torch.arange(num_anchors_total, device=device).unsqueeze(0).repeat(M, 1) # Shape (M, A)
        idx_class_dim = gt_labels.unsqueeze(1).repeat(1, num_anchors_total) # Shape (M, A)
        logit_cls_gt = cls_logits[idx_anchor_dim, idx_class_dim] # Shape (M, A)

        giou = tvops.generalized_box_iou(gt_boxes, anchor_candidate_boxes) # (M,A)
        cost_loc = 6.0 * (1.0 - giou)
        # -- VFL-style cost --------------------------------------
        cls_prob = logit_cls_gt.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        # Use the square of the standard IoU as a powerful quality weight
        quality_weight = iou.pow(self.power)
        cost_cls = self.cls_cost_weight * quality_weight * F.binary_cross_entropy(
            cls_prob, torch.ones_like(cls_prob), reduction='none'
        )
        # Combine costs with a large penalty for invalid anchors
        large_pen = torch.tensor(1e4, device=device, dtype=cost_loc.dtype)
        cost = (
            cost_loc
          + cost_cls
          + (~centre_mask.T) * large_pen
          + (~fg_cand_mask.unsqueeze(0)) * large_pen
        )

        assign_gt = cost.argmin(0)                # (A,)
        
        if self.is_debug_mode:
            # Only calculate on the valid candidate anchors to get a meaningful comparison
            valid_mask = centre_mask.T & fg_cand_mask.unsqueeze(0) # (M, A)
            if valid_mask.any():
                self._classification_debug_stats["cls_cost_magnitude"].append(cost_cls[valid_mask].mean().item())
                self._classification_debug_stats["loc_cost_magnitude"].append(cost_loc[valid_mask].mean().item())
        
                # Check how many assignments were "switched" by the classification cost
                cost_only_loc = cost_loc + (~centre_mask.T) * large_pen + (~fg_cand_mask.unsqueeze(0)) * large_pen
                assign_gt_only_loc = cost_only_loc.argmin(0)
                num_switches = (assign_gt != assign_gt_only_loc).sum().item()
                self._classification_debug_stats["assignment_switches"].append((num_switches, A))

        # 5. final fg mask = candidate ∧ in-centre of its chosen GT
        fg_final = torch.zeros(A, dtype=torch.bool, device=device)
        cand_idx = fg_cand_mask.nonzero(as_tuple=False).squeeze(1)
        if cand_idx.numel():
            in_centre = centre_mask[cand_idx, assign_gt[cand_idx]]
            fg_final[cand_idx[in_centre]] = True

        # 6. build outputs
        gt_lbl_out = torch.full((A,), -1, dtype=torch.long,  device=device)
        gt_box_out = torch.zeros((A, 4), dtype=torch.float32, device=device)
        iou_out    = torch.zeros((A,),  dtype=torch.float32, device=device)

        if fg_final.any():
            fg_idx  = fg_final.nonzero(as_tuple=False).squeeze(1)
            gt_idx  = assign_gt[fg_idx]
            gt_lbl_out[fg_idx] = gt_labels[gt_idx]
            gt_box_out[fg_idx] = gt_boxes[gt_idx]
            iou_out[fg_idx]    = iou[gt_idx, fg_idx]

        # --- Aggregate stats if in debug mode ---
        if self.is_debug_mode:
            # Calculate and store stats for this specific image
            self._debug_stats["ctr_hit_rate"].append((centre_mask.sum() / centre_mask.numel()).item())
            self._debug_stats["k_dist"].append(dynamic_ks.cpu())
            k_min_hits_img = (dynamic_ks == self.dynamic_k_min).sum().item()
            self._debug_stats["k_min_hits"].append((k_min_hits_img, M))

        # 7. lightweight debug  -----------------------------------------
        if self.is_debug_mode and (self._dbg_iter % (2000 * self._dbg_mod) == 0):
            num_fg   = fg_final.sum().item()
            num_cand = fg_cand_mask.sum().item()
            mean_iou = iou_out[fg_final].mean().item() if num_fg else 0
            k_bar    = dynamic_ks.float().mean().item()
            print(f"[SimOTA] fg={num_fg:4d} cand={num_cand:4d} "
                  f"avgIoU={mean_iou:4.3f} k̄={k_bar:3.1f}")
            if num_fg and (iou_out[fg_final] < 0.05).float().mean() > 0.2:
                print("  ⚠️  >20 % FG IoU < 0.05 – verify anchor order / scales")

        self._dbg_iter += 1
        self.mean_fg_iou = iou_out[fg_final].mean().item() if fg_final.any() else 0.0
        # ----------------------------------------------------------------

        return fg_final, gt_lbl_out, gt_box_out, iou_out


# ───────────────────── train / val loops ────────────────────────
def train_epoch(
        model: nn.Module, loader, opt, scaler, assigner: SimOTACache,
        device: torch.device, epoch: int, coco_label_map: dict,
        head_nc_for_loss: int,
        head_reg_max_for_loss: int,
        dfl_project_buffer_for_decode: torch.Tensor,
        max_epochs: int = 300,
        quality_floor_vfl: float = 0.05,
        w_cls_loss: float = 3.0,
        w_obj_loss: float = 0.5,
        w_dfl_loss: float = 0.5,
        w_iou_loss: float = 4.0,
        iou_weight_change_epoch: int = None, # Epoch to change IoU weight
        use_focal_loss: bool = False,
        debug_prints: bool = True,
        gamma_loss = 2.0,
        alpha_loss = 0.75,
        q_gamma = 0.5,  # sqrt lift of low IoUs
):
    model.train()
    # ─── containers for epoch‑wide diagnostics ──────────────────────────
    total_samples_contributing_to_loss_epoch = 0
    fg_total, img_total = 0, 0
    skips = 0
    _, tot_loss_accum = time.time(), 0.

    for i, (imgs, tgts_batch) in enumerate(loader):
        # Handle the case where the entire batch was filtered out
        if imgs is None or not imgs.numel():
            if debug_prints:
                print(f"Skipping batch {i} because it became empty after filtering.")
            continue
        H, W = imgs.shape[-2:]
        assert H % 32 == 0 and W % 32 == 0, "IMG_SIZE must be multiple of 32"

        imgs = imgs.to(device)
        model_outputs = model(imgs)

        if not model.training:
             raise RuntimeError("train_epoch called with model not in training mode.")
        if len(model_outputs) != 3:  # 4 for objectness
            raise ValueError(
                f"Expected 4 outputs from model in training mode (preds_cls, #preds_obj, preds_reg, strides), got {len(model_outputs)}"
            )
        # cls_preds_levels, obj_preds_levels, reg_preds_levels, strides_per_level_tensors = model_outputs
        cls_preds_levels, reg_preds_levels, strides_per_level_tensors = model_outputs

        bs = imgs.size(0)
        fmap_shapes = []
        for lv in range(len(cls_preds_levels)):
            H, W = cls_preds_levels[lv].shape[2:]
            s = strides_per_level_tensors[lv].item()
            fmap_shapes.append((H, W, s))
        if debug_prints and epoch == 0 and i == 0:
            for lv, (_, _, actual_s) in enumerate(fmap_shapes):
                expected_s = model.head.strides_buffer[lv].item()
                print(f"[DEBUG STRIDE CHECK] Level {lv}: actual stride = {actual_s}, expected (head buffer) = {expected_s}")

        batch_total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_samples_with_loss_in_batch = 0

        loss_cls_print, loss_obj_print, loss_dfl_print, loss_iou_print = [torch.tensor(0.0) for _ in range(4)]

        for b_idx in range(bs):
            tgt_i = tgts_batch[b_idx]
            gt_boxes_unfiltered = tgt_i["boxes"].to(dtype=torch.float32, device=device, non_blocking=True)
            gt_labels_unfiltered = tgt_i["labels"].to(device=device, non_blocking=True)

            gt_boxes_img = gt_boxes_unfiltered
            gt_labels_img = gt_labels_unfiltered

            if gt_boxes_unfiltered.numel() > 0:
                # assert (gt_labels_img < model.head.nc).all(), "Found an invalid class label index!"
                widths = gt_boxes_unfiltered[:, 2] - gt_boxes_unfiltered[:, 0]
                heights = gt_boxes_unfiltered[:, 3] - gt_boxes_unfiltered[:, 1]
                epsilon_wh = 1e-3
                valid_gt_mask = (widths > epsilon_wh) & (heights > epsilon_wh)
                gt_boxes_img = gt_boxes_unfiltered[valid_gt_mask]
                gt_labels_img = gt_labels_unfiltered[valid_gt_mask]

            target_dict_for_assigner = {'boxes': gt_boxes_img, 'labels': gt_labels_img}

            if assigner.nc != head_nc_for_loss and i == 1:
                 warnings.warn(f"Warning: assigner.nc ({assigner.nc}) != head_nc_for_loss ({head_nc_for_loss}).")

            if debug_prints and epoch == 0 and i == 0:
                H_img_shape, W_img_shape = imgs[b_idx].shape[-2:]
                bad_coords_after_filter = False
                if gt_boxes_img.numel() > 0:
                    widths_filtered = gt_boxes_img[:, 2] - gt_boxes_img[:, 0]
                    heights_filtered = gt_boxes_img[:, 3] - gt_boxes_img[:, 1]
                    bad_coords_after_filter = (
                        (gt_boxes_img[:, 0] < -epsilon_wh) | (gt_boxes_img[:, 1] < -epsilon_wh) |
                        (gt_boxes_img[:, 2] > W_img_shape + epsilon_wh) | (gt_boxes_img[:, 3] > H_img_shape + epsilon_wh) |
                        (widths_filtered <= 0) | (heights_filtered <= 0)
                    ).any()
                first_tgt_original_boxes = tgts_batch[b_idx]["boxes"]
                first_box_original_coords = first_tgt_original_boxes[:1].tolist() if first_tgt_original_boxes.numel() else None
                if bad_coords_after_filter:
                    raise ValueError(f"bad coords: [CHECK] Img shape {imgs[b_idx].shape}, sample {b_idx} first original box: {first_box_original_coords}")

            # flatten predictions
            cls_p_img = torch.cat([
                lvl[b_idx].permute(1, 2, 0).reshape(-1, head_nc_for_loss)
                for lvl in cls_preds_levels
            ], dim=0)                                     # (A, C)
            # obj_p_img = torch.cat([
            #     lvl[b_idx].permute(1, 2, 0).reshape(-1)
            #     for lvl in obj_preds_levels
            # ], dim=0)                                     # (A,)
            reg_p_img = torch.cat([
                lvl[b_idx].permute(1,2,0).reshape(-1, 4*(head_reg_max_for_loss+1))
                for lvl in reg_preds_levels
            ], dim=0)
            if i == 0 and b_idx in [0, 1]   :
                print("\n--- COORDINATE SYSTEM SANITY CHECK ---")
                gt_sample = gt_boxes_img[0] if gt_boxes_img.numel() > 0 else "N/A no GT"
                gt_max = gt_boxes_img.max().item() if gt_boxes_img.numel() > 0 else "N/A no GT"
                
                # Recreate the anchor list used by the assigner
                anchors = torch.cat([getattr(model.head, f'anchor_points_level_{i}') for i in range(model.head.nl)], 0)
                
                print(f"Image Size: {list(imgs.shape[-2:])}")
                print(f"Ground-Truth Box Sample: {gt_sample}")
                print(f"Ground-Truth Box Max Value: {gt_max}")
                print(f"Anchor Center Max Value: {anchors.max().item()}")
                print("--- END CHECK ---\n")

            fg_mask_img, gt_labels, gt_boxes, gt_ious = assigner(
                    fmap_shapes, device, target_dict_for_assigner,
                    cls_logits=cls_p_img, # Raw class logits
                    # obj_logits=obj_p_img,
                    model_head=model.head,
            )
            reg_p_fg_img = reg_p_img[fg_mask_img]
            box_targets_fg_img = gt_boxes[fg_mask_img]

            num_fg_img = fg_mask_img.sum().item()
            fg_total += num_fg_img
            img_total += 1
            loss_norm_factor = float(num_fg_img)
            if num_fg_img == 0:
                skips += 1
                if skips == 400:
                    print("Large number of skipped batches, Possibly there is a bug")
                    
                    print("\n--- COORDINATE SYSTEM SANITY CHECK ---")
                    gt_sample = gt_boxes_img[0] if gt_boxes_img.numel() > 0 else "N/A no GT"
                    gt_max = gt_boxes_img.max().item() if gt_boxes_img.numel() > 0 else "N/A no GT"
                    
                    # Recreate the anchor list used by the assigner
                    anchors = torch.cat([getattr(model.head, f'anchor_points_level_{i}') for i in range(model.head.nl)], 0)
                    
                    print(f"Image Size: {list(imgs.shape[-2:])}")
                    print(f"Ground-Truth Box Sample: {gt_sample}")
                    print(f"Ground-Truth Box Max Value: {gt_max}")
                    print(f"Anchor Center Max Value: {anchors.max().item()}")
                    print("--- END CHECK ---\n")
                continue

            ##########
            # --- Distribution Focal Loss (DFL) & IoU Loss for BBox Regression ---
            # (These are independent of the classification/objectness choice above and operate on reg_p_img)

            pred_ltrb_offsets_fg_img = PicoDetHead.dfl_decode_for_training(
                reg_p_fg_img,
                dfl_project_buffer_for_decode.to(reg_p_fg_img.device),
                head_reg_max_for_loss
            )

            # build anchors & strides from the head buffers (always consistent)
            head_anchors_list  = []
            head_strides_list  = []
            for lvl_idx in range(model.head.nl):
                anchor_level = getattr(model.head, f'anchor_points_level_{lvl_idx}').to(device)
                stride_val   = model.head.strides_buffer[lvl_idx].to(device)          # scalar tensor
                head_anchors_list.append(anchor_level)
                head_strides_list.append(stride_val.expand(anchor_level.size(0)))
            
            anchor_centers_all_img = torch.cat(head_anchors_list, 0)     # (A , 2)
            anchor_centers_fg_img = anchor_centers_all_img[fg_mask_img]
            strides_all_anchors_img = torch.cat(head_strides_list, 0)    # (A ,)
            strides_fg_img = strides_all_anchors_img[fg_mask_img].unsqueeze(-1)

            if debug_prints and epoch == 0 and i == 0 and b_idx == 0:
                head_anchors_flat_list = []
                for lvl_idx_head in range(model.head.nl):
                    # Ensure model.head is accessible; if qat_model is passed, might need model_fp32.head
                    anchor_points_level = getattr(model.head, f'anchor_points_level_{lvl_idx_head}')
                    head_anchors_flat_list.append(anchor_points_level.to(device))
                head_anchors_concatenated = torch.cat(head_anchors_flat_list, dim=0)
            
                if not torch.allclose(anchor_centers_all_img, head_anchors_concatenated, atol=1e-5):
                    print("[CRITICAL WARNING] Anchor order mismatch between SimOTA generated anchors and PicoDetHead internal anchors!")
                    print("SimOTA anchors (first 5):", anchor_centers_all_img[:5])
                    print("Head anchors (first 5):", head_anchors_concatenated[:5])
                    # Consider raising an error or adding more detailed comparison
                else:
                    print("[INFO] Anchor order check passed for first sample.")

            ltrb_offsets = torch.stack([
                anchor_centers_fg_img[:, 0] - box_targets_fg_img[:, 0],
                anchor_centers_fg_img[:, 1] - box_targets_fg_img[:, 1],
                box_targets_fg_img[:, 2] - anchor_centers_fg_img[:, 0],
                box_targets_fg_img[:, 3] - anchor_centers_fg_img[:, 1],
            ], dim=1) / strides_fg_img
            ltrb_offsets = ltrb_offsets.clamp_(0, head_reg_max_for_loss)
            dfl_target_dist = build_dfl_targets(ltrb_offsets, head_reg_max_for_loss)
            loss_dfl = dfl_loss(reg_p_fg_img, dfl_target_dist)  #  / num_fg_img

            pred_ltrb_pixels_fg_img = pred_ltrb_offsets_fg_img * strides_fg_img
            pred_boxes_fg_img = torch.stack((
                anchor_centers_fg_img[:, 0] - pred_ltrb_pixels_fg_img[:, 0],
                anchor_centers_fg_img[:, 1] - pred_ltrb_pixels_fg_img[:, 1],
                anchor_centers_fg_img[:, 0] + pred_ltrb_pixels_fg_img[:, 2],
                anchor_centers_fg_img[:, 1] + pred_ltrb_pixels_fg_img[:, 3]
            ), dim=1)
            loss_iou = tvops.complete_box_iou_loss(pred_boxes_fg_img, box_targets_fg_img, reduction='sum') / num_fg_img

            ########
            # Joint logits: always the same expression
            # joint_logits = cls_p_img + model.head.logit_scale * obj_p_img.unsqueeze(1)
            joint_logits = cls_p_img
            
            # foreground indices for this image
            pos_indices = fg_mask_img.nonzero(as_tuple=False).squeeze(1)  # (N_fg,)
            
            if pos_indices.numel() == 0:          # nothing to learn from this image
                continue

            gt_labels_pos    = gt_labels[pos_indices]             # (N_fg ,)
            
            # ------------------------------------------------------------------
            # 2. Classification loss
            # ------------------------------------------------------------------
            loss_obj = torch.tensor(0.0)
            if use_focal_loss:          # warm-up path (plain BCE is fine here)
                cls_logits_pos = cls_p_img[pos_indices]
                gt_labels_pos = gt_labels[pos_indices]
                # Create one-hot targets for the foreground predictions
                cls_targets_pos = torch.zeros_like(cls_logits_pos)
                cls_targets_pos[torch.arange(num_fg_img, device=device), gt_labels_pos] = 1.0
                # Use torchvision's numerically stable and reliable focal loss.
                # It's normalized by the number of positive anchors, as is standard.
                loss_cls = tvops.sigmoid_focal_loss(
                    cls_logits_pos,
                    cls_targets_pos,
                    alpha=0.25,  # Standard value to balance positive/negative examples
                    gamma=gamma_loss,   # Standard value to focus on hard examples
                    reduction='sum'
                ) / loss_norm_factor
                # 2. Objectness Loss (Binary Cross-Entropy)
                # Applied to ALL anchors (foreground and background).
                # obj_targets = torch.zeros_like(obj_p_img)
                # For foreground anchors, the target is their IoU with the matched GT box.
                # This teaches the model that higher IoU boxes are "more object".
                # obj_targets[fg_mask_img] = gt_ious[fg_mask_img]
                # Use standard BCE with logits, averaged over all anchors.
                # loss_obj = F.binary_cross_entropy_with_logits(obj_p_img, obj_targets, reduction='mean')

                """
                alpha_focal, gamma = 0.25, 2.0
                loss_cls = tvops.sigmoid_focal_loss(
                    cls_p_img, cls_targets, alpha=alpha_focal, gamma=gamma, reduction='sum'
                ) / loss_norm_factor    
                """
            
            else:  # Varifocal Loss proper
                targets = torch.zeros_like(joint_logits)
                
                if pos_indices.numel() > 0:
                    gt_labels_pos = gt_labels[pos_indices]
                
                    with torch.no_grad():
                        # predicted boxes & matched GT for the *foreground* anchors
                        iou_pred = tvops.box_iou(pred_boxes_fg_img, box_targets_fg_img).diag()
                
                        # shape quality
                        quality = iou_pred.clamp_min(quality_floor_vfl).pow(q_gamma)  # VFL
                        # quality = iou_pred.clamp_min(quality_floor_vfl)  # QFL
                
                    targets[pos_indices, gt_labels_pos] = quality
                
                vfl_calculator = VarifocalLoss(alpha=alpha_loss, gamma=gamma_loss, reduction='sum')
                # vfl_calculator = QualityFocalLoss(beta=gamma_loss, reduction='sum')
                loss_cls = vfl_calculator(joint_logits, targets) / max(1, num_fg_img)

                w_obj_loss = 0.01  # small placeholder
            
            # Final sample loss (obj already inside joint_logits)
            current_sample_total_loss = (
                  w_cls_loss * loss_cls
                + w_dfl_loss * loss_dfl
                + w_iou_loss * loss_iou
                + w_obj_loss * loss_obj
            )
            if not torch.isfinite(current_sample_total_loss):
                print(f"current_sample_total_loss is not finite: {current_sample_total_loss}")
                current_sample_total_loss = 10.0  # larger than normal error
            
            # for the first sample in the epoch – diagnostics only
            if num_samples_with_loss_in_batch == 0 and total_samples_contributing_to_loss_epoch == 0:
                loss_cls_print = loss_cls.detach()
                loss_obj_print = loss_obj.detach()
                loss_dfl_print = loss_dfl.detach()
                loss_iou_print = loss_iou.detach()

            batch_total_loss += current_sample_total_loss
            num_samples_with_loss_in_batch += 1

        if num_samples_with_loss_in_batch > 0:
            averaged_batch_loss = batch_total_loss / num_samples_with_loss_in_batch
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(averaged_batch_loss) or averaged_batch_loss.isnan():
                continue
            scaler.scale(averaged_batch_loss).backward()
            scaler.unscale_(opt)

            if i % 3000 == 0:
                ### Norm sizing debug
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Batch {i}, Total Grad Norm: {total_norm:.4f}")
            gradient_norm = 6.0  # 1.0 to 10.0 common, max gradient observed in debug was around 6
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
            scaler.step(opt); scaler.update()
            # with torch.no_grad():
            #     model.head.logit_scale.clamp_(-1.0, 3.0)
            tot_loss_accum += averaged_batch_loss.item() * num_samples_with_loss_in_batch
            total_samples_contributing_to_loss_epoch += num_samples_with_loss_in_batch

        if debug_prints and i == 0 and total_samples_contributing_to_loss_epoch > 0 : # Print first batch loss breakdown
             print(f"E{epoch} B0 loss components: "
                  f"cls {loss_cls_print.item() * w_cls_loss:.3f} (w={w_cls_loss})  obj {loss_obj_print.item() * w_obj_loss:.3f} (w={w_obj_loss})  "
                  f"dfl {loss_dfl_print.item() * w_dfl_loss:.3f} (w={w_dfl_loss})  iou {loss_iou_print.item() * w_iou_loss:.3f} (w={w_iou_loss})  "
                  f"tot_batch_avg {averaged_batch_loss.item():.3f}")
             with torch.no_grad():
                    fg_indices = fg_mask_img.nonzero(as_tuple=False).squeeze(1)
                    cls_p_fg_img = cls_p_img[fg_indices] # (N_fg, C)
                    gt_labels_fg = gt_labels[fg_indices] # (N_fg,)
            
                    # Get the logit value for the correct class for each FG prediction
                    correct_class_logits = cls_p_fg_img[torch.arange(num_fg_img), gt_labels_fg]
            
                    # Find the highest logit among all *incorrect* classes for each FG prediction
                    cls_p_fg_img[torch.arange(num_fg_img), gt_labels_fg] = -float('inf') # Mask out correct class
                    max_incorrect_class_logits, _ = cls_p_fg_img.max(dim=1)
            
                    # The "margin" is the difference. We want this to be large and positive.
                    logit_margin = correct_class_logits - max_incorrect_class_logits
            
                    print(f"\n--- [DBG E{epoch} B{i}] Logit Analysis (FG Samples) ---")
                    print(f"  Avg Logit for Correct Class: {correct_class_logits.mean().item():.3f}")
                    print(f"  Avg Logit for Top Incorrect Class: {max_incorrect_class_logits.mean().item():.3f}")
                    print(f"  Avg Margin (Correct - Incorrect): {logit_margin.mean().item():.3f} (Stdev: {logit_margin.std().item():.3f})")
                    print(f"  % of FG samples with negative margin: {(logit_margin < 0).float().mean().item():.2%}\n")

        elif debug_prints and i % 500 == 0 and i > 0 and num_samples_with_loss_in_batch > 0:
            print(f"E{epoch} {i:04d}/{len(loader)} loss {averaged_batch_loss.item():.3f} (batch avg)")
        if debug_prints and (epoch % 10 == 0) and i == 0:
            try:
                with torch.no_grad():
                    # ─── concatenate logits from all FPN levels ───────────────────
                    cls_logits_all = torch.cat(
                        [lvl.detach().permute(0, 2, 3, 1).reshape(-1, head_nc_for_loss)
                         for lvl in cls_preds_levels], dim=0)                # [A_total , C]
                    # obj_logits_all = torch.cat(
                    #     [lvl.detach().permute(0, 2, 3, 1).reshape(-1, 1)
                    #      for lvl in obj_preds_levels], dim=0)                # [A_total , 1]
            
                    # Classification confidence analysis
                    cls_probs = cls_logits_all.sigmoid()
                    # obj_probs = obj_logits_all.sigmoid()
                    # joint_conf = (cls_logits_all + obj_logits_all).sigmoid().flatten()
                    joint_conf = cls_logits_all.sigmoid().flatten()
                    
                    # Per-class prediction distribution
                    max_cls_probs, max_cls_indices = cls_probs.max(dim=1)
                    print(f"\n[DBG] Epoch {epoch} Classification Analysis:")
                    print(f"  Cls logits range: [{cls_logits_all.min().item():.3f}, {cls_logits_all.max().item():.3f}]")
                    # print(f"  Obj logits range: [{obj_logits_all.min().item():.3f}, {obj_logits_all.max().item():.3f}]")
                    print(f"  Joint conf range: [{joint_conf.min().item():.3f}, {joint_conf.max().item():.3f}]")
                    
                    # Class distribution
                    print("  Top predicted classes:")
                    for class_id in range(min(5, head_nc_for_loss)):
                        count = (max_cls_indices == class_id).sum().item()
                        avg_conf = cls_probs[:, class_id].mean().item()
                        print(f"    Class {class_id}: {count:6d} predictions, avg_conf: {avg_conf:.4f}")
            
                    # sample at most 100k values to keep it fast
                    if joint_conf.numel() > 100000:
                        idx = torch.randperm(joint_conf.numel(), device=joint_conf.device)[:100_000]
                        joint_conf = joint_conf[idx]
            
                    q50, q90, q99 = torch.quantile(
                        joint_conf.cpu(), torch.tensor([0.5, 0.9, 0.99]))
                    print(f"  Conf quantiles 50/90/99%: {q50:.3f} {q90:.3f} {q99:.3f}")
                    
                    # Confidence thresholds
                    for thresh in [0.01, 0.05, 0.1, 0.25]:
                        above_thresh = (joint_conf > thresh).sum().item()
                        percentage = 100 * above_thresh / joint_conf.numel()
                        print(f"  >{thresh:.2f}: {percentage:.1f}%")
                        
            except Exception as e:
                print(f"[DEBUG ERROR] {repr(e)}")

    if total_samples_contributing_to_loss_epoch > 0:
        avg_epoch_loss_per_sample = tot_loss_accum / total_samples_contributing_to_loss_epoch
        epoch_diag = {
            "fg_per_img": fg_total / max(img_total, 1),
            "centre_hit":  np.mean(assigner._debug_stats["ctr_hit_rate"])
        }
        return avg_epoch_loss_per_sample, epoch_diag
    else:
        print(f"E{epoch} No samples contributed to loss this epoch.")
        return 0.0, {}


# ───────────────────── QAT helpers ───────────────────────────────
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping, QConfig,
    MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver # Add MovingAverageMinMaxObserver
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx


def qat_prepare(model: nn.Module, example_input: torch.Tensor) -> torch.fx.GraphModule:
    """
    Prepares the model for Quantization-Aware Training (QAT) using FX graph mode.

    Args:
        model: The PyTorch nn.Module to prepare for QAT.
        example_input: A representative example input tensor for tracing the model.

    Returns:
        A torch.fx.GraphModule prepared for QAT.
    """
    # 1. Define the desired QConfig for most layers
    #    Activations: per-tensor, affine, quint8
    #    Weights: per-channel, symmetric, qint8 (common for conv/linear layers)
    global_qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
    )

    # 2. Get the default QAT qconfig mapping for the chosen backend
    #    'x86' or 'qnnpack' (for ARM) are common backends.
    backend_config = "x86"  # Or "qnnpack" explicitly
    qconfig_mapping = get_default_qat_qconfig_mapping(backend_config)

    # 3. Apply the global QConfig to the mapping
    #    This will apply `global_qconfig` to all quantizable module types
    #    unless overridden by more specific settings (e.g., per module type/instance).
    qconfig_mapping = qconfig_mapping.set_global(global_qconfig)

    # 4. Specify modules to skip quantization
    #    The 'pre' module (ResizeNorm) handles uint8 input and normalization;
    qconfig_mapping = qconfig_mapping.set_module_name('pre', None)
    qconfig_mapping = qconfig_mapping.set_object_type(ResizeNorm, None)

    # 5. Prepare the model for QAT using FX graph mode
    model.cpu().train() # Ensure model is on CPU and in train mode
    
    prepared_model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_input # Must also be on CPU
    )
    
    return prepared_model


def unwrap_dataset(ds):
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds


def apply_nms_and_padding_to_raw_outputs_with_debug( # Renamed
    raw_boxes_batch: torch.Tensor, # (B, Total_Anchors, 4)
    raw_scores_batch: torch.Tensor, # (B, Total_Anchors, NC)
    score_thresh: float,
    iou_thresh: float,
    max_detections: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    
    device = raw_boxes_batch.device
    batch_size = raw_boxes_batch.shape[0]

    final_boxes_list = []
    final_scores_list = []
    final_labels_list = []
    
    num_preds_before_nms_list = [] # Debug
    num_preds_after_nms_list = []  # Debug

    for b_idx in range(batch_size):
        boxes_img = raw_boxes_batch[b_idx]     # (Total_Anchors, 4)
        scores_img = raw_scores_batch[b_idx]   # (Total_Anchors, NC)

        # For each anchor, find the max score across classes and the corresponding class label
        conf_per_anchor, labels_per_anchor = torch.max(scores_img, dim=1) # (Total_Anchors,)
        
        # Filter by score_thresh BEFORE NMS
        keep_by_score_mask = conf_per_anchor >= score_thresh
        
        boxes_above_thresh = boxes_img[keep_by_score_mask]
        scores_above_thresh = conf_per_anchor[keep_by_score_mask]
        labels_above_thresh = labels_per_anchor[keep_by_score_mask]

        num_preds_before_nms_this_img = boxes_above_thresh.shape[0]
        num_preds_before_nms_list.append(num_preds_before_nms_this_img)

        num_preds_after_nms_this_img = 0
        if boxes_above_thresh.numel() == 0:
            # Pad and append if no boxes pass threshold
            padded_boxes = torch.zeros((max_detections, 4), dtype=boxes_img.dtype, device=device)
            padded_scores = torch.zeros((max_detections,), dtype=scores_img.dtype, device=device)
            padded_labels = torch.full((max_detections,), -1, dtype=torch.long, device=device) # Use -1 for padding label
        else:
            # Perform NMS
            nms_keep_indices = tvops.batched_nms(
                boxes_above_thresh,    # boxes
                scores_above_thresh,   # scores
                labels_above_thresh,   # class indexes
                iou_thresh             # iou_threshold
            )

            # Keep only up to max_detections
            boxes_after_nms = boxes_above_thresh[nms_keep_indices[:max_detections]]
            scores_after_nms = scores_above_thresh[nms_keep_indices[:max_detections]]
            labels_after_nms = labels_above_thresh[nms_keep_indices[:max_detections]]
            
            num_preds_after_nms_this_img = boxes_after_nms.shape[0]
            
            # Pad to max_detections
            num_current_dets = boxes_after_nms.shape[0]
            pad_size = max_detections - num_current_dets
            
            padded_boxes = F.pad(boxes_after_nms, (0, 0, 0, pad_size), mode='constant', value=0.0)
            padded_scores = F.pad(scores_after_nms, (0, pad_size), mode='constant', value=0.0)
            padded_labels = F.pad(labels_after_nms, (0, pad_size), mode='constant', value=-1) # Use -1 for padding label

        final_boxes_list.append(padded_boxes)
        final_scores_list.append(padded_scores)
        final_labels_list.append(padded_labels)
        num_preds_after_nms_list.append(num_preds_after_nms_this_img)


    return (torch.stack(final_boxes_list),
            torch.stack(final_scores_list),
            torch.stack(final_labels_list),
            num_preds_before_nms_list, # Return debug info
            num_preds_after_nms_list)  # Return debug info


@torch.no_grad()
def quick_val_iou(
    model: nn.Module, # Use generic nn.Module to indicate it handles different model types
    loader, device,
    score_thresh: float,
    iou_thresh: float,
    max_detections: int,
    iou_match_thresh: float = 0.5, # Threshold to consider a prediction "correctly localized"
    epoch_num: int = -1,
    run_name: str = "N/A",
    debug_prints: bool = True,
):
    model.eval()
    total_iou_sum = 0.
    num_images_with_gt = 0
    num_images_processed = 0
    total_gt_boxes_across_images = 0
    total_preds_before_nms_filter_across_images = 0
    total_preds_after_nms_filter_across_images = 0

    # Add counters for classification accuracy
    total_matched_preds = 0
    correctly_classified_preds = 0

    if debug_prints:
        print(f"\n--- quick_val_iou Start (Epoch: {epoch_num}, Run: {run_name}) ---")
        print(f"Params: score_thresh={score_thresh}, iou_thresh={iou_thresh}, max_detections={max_detections}")

    for batch_idx, (imgs_batch, tgts_batch) in enumerate(loader):
        raw_pred_boxes_batch, raw_pred_scores_batch = model(imgs_batch.to(device))

        if debug_prints and batch_idx == 0:
            print(f"[Debug Eval Batch 0] raw_pred_boxes_batch shape: {raw_pred_boxes_batch.shape}")
            print(f"[Debug Eval Batch 0] raw_pred_scores_batch shape: {raw_pred_scores_batch.shape}")
            bb = tgts_batch[0]["boxes"]
            print(f"  [DEBUG] batch {batch_idx} images: {imgs_batch.shape}, boxes: {bb.shape}, sample: {bb[:2]}")

        (pred_boxes_batch_padded,
         pred_scores_batch_padded,
         pred_labels_batch_padded,
         debug_num_preds_before_nms_batch,
         debug_num_preds_after_nms_batch
         ) = apply_nms_and_padding_to_raw_outputs_with_debug(
                raw_pred_boxes_batch, raw_pred_scores_batch,
                score_thresh, iou_thresh, max_detections
            )

        total_preds_before_nms_filter_across_images += sum(debug_num_preds_before_nms_batch)

        for i in range(imgs_batch.size(0)):
            num_images_processed += 1
            current_img_annots_raw = tgts_batch[i]

            # Make sure to get both boxes and labels for the ground truth
            gt_boxes_xyxy = current_img_annots_raw["boxes"]
            gt_labels = current_img_annots_raw["labels"]

            if gt_boxes_xyxy.numel():
                gt_boxes_tensor = gt_boxes_xyxy.to(device=device, dtype=torch.float32)
                gt_labels_tensor = gt_labels.to(device=device)
            else:
                gt_boxes_tensor = torch.empty((0, 4), device=device)
                gt_labels_tensor = torch.empty((0,), dtype=torch.long, device=device)

            if gt_boxes_tensor.numel() == 0:
                continue

            num_images_with_gt += 1
            total_gt_boxes_across_images += gt_boxes_tensor.shape[0]

            num_actual_dets_this_img = debug_num_preds_after_nms_batch[i]
            total_preds_after_nms_filter_across_images += num_actual_dets_this_img

            if debug_prints and batch_idx == 0 and i < 2:
                print(f"[Debug Eval Img {num_images_processed-1}] GTs: {gt_boxes_tensor.shape[0]}.")
                print(f"  Num Preds BEFORE NMS (passed score_thresh): {debug_num_preds_before_nms_batch[i]}")
                print(f"  Num Preds AFTER NMS (final): {num_actual_dets_this_img}")
                # The problematic debug line that accessed model.head.cls_pred was here and has been removed.

            if num_actual_dets_this_img == 0:
                continue

            # Get the actual (unpadded) predictions for this image
            actual_predicted_boxes = pred_boxes_batch_padded[i, :num_actual_dets_this_img]
            actual_predicted_labels = pred_labels_batch_padded[i, :num_actual_dets_this_img]

            iou_matrix = tvops.box_iou(actual_predicted_boxes, gt_boxes_tensor)

            if iou_matrix.numel() > 0:
                max_iou_per_gt, _ = iou_matrix.max(dim=0)
                total_iou_sum += max_iou_per_gt.sum().item()

                # --- Logic for Classification Accuracy ---
                max_iou_per_pred, best_gt_indices_for_pred = iou_matrix.max(dim=1)
                matched_mask = max_iou_per_pred > iou_match_thresh
                matched_pred_labels = actual_predicted_labels[matched_mask]
                matched_gt_labels = gt_labels_tensor[best_gt_indices_for_pred[matched_mask]]
                matches_are_correctly_classified = (matched_pred_labels == matched_gt_labels).sum().item()
                total_matched_preds += matched_mask.sum().item()
                correctly_classified_preds += matches_are_correctly_classified

    if debug_prints:
        print(f"--- quick_val_iou End (Epoch: {epoch_num}, Run: {run_name}) ---")
        print(f"Images processed: {num_images_processed}, Images with GT: {num_images_with_gt}")
        print(f"Total GT boxes: {total_gt_boxes_across_images}")
        avg_preds_before_nms = total_preds_before_nms_filter_across_images / num_images_processed if num_images_processed > 0 else 0
        avg_preds_after_nms = total_preds_after_nms_filter_across_images / num_images_processed if num_images_processed > 0 else 0
        print(f"Avg preds/img (passed score_thresh, before NMS): {avg_preds_before_nms:.2f}")
        print(f"Avg preds/img (after NMS & final filter): {avg_preds_after_nms:.2f}")

    if total_gt_boxes_across_images > 0:
        final_mean_iou = total_iou_sum / total_gt_boxes_across_images
    else:
        final_mean_iou = 0.0

    if total_matched_preds > 0:
        final_accuracy = correctly_classified_preds / total_matched_preds
    else:
        final_accuracy = 0.0

    return final_mean_iou, final_accuracy

@torch.no_grad()
def run_epoch_validation(model: nn.Module, loader, device, epoch_num: int, head_ref: PicoDetHead):
    """
    Runs a comprehensive validation for an epoch, testing at multiple score thresholds.

    Args:
        model: The model to evaluate.
        loader: The validation DataLoader.
        device: The device to run on.
        epoch_num: The current epoch number for logging.
        head_ref: A reference to the model's head to get NMS parameters.

    Returns:
        A dictionary containing key metrics for the epoch.
    """
    model.eval()
    
    # Define the score thresholds you want to track
    score_thresholds_to_track = [0.05, 0.25, 0.50]
    
    epoch_summary = {}

    print(f"\n--- Running Validation for Epoch {epoch_num} ---")
    for score_th in score_thresholds_to_track:
        run_name = f"val_ep{epoch_num}_score{score_th}"
        
        # We set debug_prints=False to keep the console clean during this automated run
        iou, acc = quick_val_iou(
            model, loader, device,
            score_thresh=score_th,
            iou_thresh=head_ref.iou_th,
            max_detections=head_ref.max_det,
            epoch_num=epoch_num, 
            run_name=run_name,
            debug_prints=False, # Keep logs clean
        )
        
        # Use a descriptive key for the summary dictionary
        key_iou = f"iou_at_{int(score_th*100)}".replace('.', '')
        key_acc = f"acc_at_{int(score_th*100)}".replace('.', '')
        
        epoch_summary[key_iou] = iou
        epoch_summary[key_acc] = acc
        print(f"  [Score > {score_th:.2f}] --> Mean IoU: {iou:.4f}, Accuracy: {acc:.4f}")
        
    return epoch_summary


class PostprocessorForONNX(nn.Module):
    def __init__(self, head_ref: PicoDetHead): # Pass original PicoDetHead instance
        super().__init__()
        self.nc = head_ref.nc
        self.reg_max = head_ref.reg_max
        self.nl = head_ref.nl # Number of FPN levels (e.g., 3)

        # Register buffers needed for decoding, cloned from the original head.
        # These will be part of the ONNX graph as constants if not inputs.
        self.register_buffer('strides_buffer', head_ref.strides_buffer.clone().detach(), persistent=False)
        self.register_buffer('dfl_project_buffer', head_ref.dfl_project_buffer.clone().detach(), persistent=False)
        # self.register_buffer('logit_scale', head_ref.logit_scale.clone().detach(), persistent=False)
        for i in range(self.nl):
            anchor_points = getattr(head_ref, f'anchor_points_level_{i}')
            self.register_buffer(f'anchor_points_level_{i}', anchor_points.clone().detach(), persistent=False)

    def _dfl_to_ltrb_inference_onnx(self, x_reg_logits_3d: torch.Tensor) -> torch.Tensor:
        # x_reg_logits_3d shape: (B, N_anchors_img_level, 4 * (reg_max + 1))
        # Using .size() for ONNX compatibility with dynamic shapes
        b = x_reg_logits_3d.size(0)
        n_anchors_img_level = x_reg_logits_3d.size(1)

        # Reshape for softmax and projection. self.reg_max + 1 is constant.
        # Target shape: (B, N_anchors_img_level, 4, self.reg_max + 1)
        x_reg_logits_reshaped = x_reg_logits_3d.view(b, n_anchors_img_level, 4, self.reg_max + 1)
        x_softmax = F.softmax(x_reg_logits_reshaped, dim=3) # Apply softmax over reg_max+1 dimension

        # self.dfl_project_buffer has shape (reg_max+1)
        # Unsqueeze for broadcasting: (1, 1, 1, reg_max+1)
        proj = self.dfl_project_buffer.view(1, 1, 1, self.reg_max + 1) # Use self.reg_max + 1 for shape
        ltrb_offsets = (x_softmax * proj).sum(dim=3) # Sum over reg_max+1 dim
        return ltrb_offsets

    def _decode_predictions_for_level_onnx(
            self,
            cls_logit: torch.Tensor,  # (B, NC, H_feat, W_feat)
            # obj_logit: torch.Tensor,  # (B, 1,  H_feat, W_feat)
            reg_logit: torch.Tensor,  # (B, 4*(reg_max+1), H_feat, W_feat)
            level_idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # B can be dynamic, H_feat, W_feat might be if input image size is dynamic
        B = cls_logit.size(0)
        H_feat = cls_logit.size(2)
        W_feat = cls_logit.size(3)

        stride_val = self.strides_buffer[level_idx] # Scalar tensor for this level
        num_anchors_level = H_feat * W_feat

        # anchor_points_center shape: (Max_H_feat_for_level * Max_W_feat_for_level, 2)
        # If H_feat, W_feat are dynamic and smaller than max, slicing/selection might be needed.
        # For fixed IMG_SIZE during QAT, H_feat, W_feat will be fixed for each level.
        anchor_points_center = getattr(self, f'anchor_points_level_{level_idx}')

        # Permute and reshape logits
        # (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
        # For ONNX, ensure reshape uses -1 carefully if B or H*W is dynamic.
        cls_logit_perm = cls_logit.permute(0, 2, 3, 1).contiguous().view(B, num_anchors_level, self.nc)
        # obj_logit_perm = obj_logit.permute(0, 2, 3, 1).contiguous().view(B, num_anchors_level, 1)
        reg_logit_perm = reg_logit.permute(0, 2, 3, 1).contiguous().view(B, num_anchors_level, 4 * (self.reg_max + 1))

        ltrb_offsets = self._dfl_to_ltrb_inference_onnx(reg_logit_perm)
        ltrb_offsets_scaled = ltrb_offsets * stride_val # Element-wise with broadcasting (stride_val is scalar)

        # ap_expanded shape: (1, num_anchors_level, 2) for broadcasting with B
        ap_expanded = anchor_points_center.unsqueeze(0)

        # Calculate box coordinates
        x1 = ap_expanded[..., 0] - ltrb_offsets_scaled[..., 0]
        y1 = ap_expanded[..., 1] - ltrb_offsets_scaled[..., 1]
        x2 = ap_expanded[..., 0] + ltrb_offsets_scaled[..., 2]
        y2 = ap_expanded[..., 1] + ltrb_offsets_scaled[..., 3]
        boxes_xyxy_level = torch.stack([x1, y1, x2, y2], dim=-1) # Shape (B, num_anchors_level, 4)

        # scores_level = torch.sigmoid(cls_logit_perm + self.logit_scale * obj_logit_perm)
        # scores_level = torch.sigmoid((cls_logit_perm + 0.5) / 0.8)
        scores_level = torch.sigmoid(cls_logit_perm)

        return boxes_xyxy_level, scores_level

    def forward(self, raw_model_outputs_nested_tuple: Tuple[Tuple[torch.Tensor, ...], ...]):
        if not isinstance(raw_model_outputs_nested_tuple, tuple) or len(raw_model_outputs_nested_tuple) < 2:
            raise ValueError(
                f"PostprocessorForONNX expected a nested tuple with at least 3 inner tuples, "
                f"got type {type(raw_model_outputs_nested_tuple)} with length {len(raw_model_outputs_nested_tuple) if isinstance(raw_model_outputs_nested_tuple, tuple) else 'N/A'}"
            )

        raw_cls_logits_levels_tuple = raw_model_outputs_nested_tuple[0]
        # raw_obj_logits_levels_tuple = raw_model_outputs_nested_tuple[1]  # below as [2]
        raw_reg_logits_levels_tuple = raw_model_outputs_nested_tuple[1]

        if not (isinstance(raw_cls_logits_levels_tuple, tuple) and
                  # isinstance(raw_obj_logits_levels_tuple, tuple) and
                  isinstance(raw_reg_logits_levels_tuple, tuple)):
            raise ValueError("Inner elements of the input to PostprocessorForONNX are not tuples as expected.")

        if not (len(raw_cls_logits_levels_tuple) == self.nl and
                # len(raw_obj_logits_levels_tuple) == self.nl and
                len(raw_reg_logits_levels_tuple) == self.nl):
            raise ValueError(
                f"PostprocessorForONNX: Inner tuples do not have the expected length ({self.nl}). "
                f"Got lengths: cls={len(raw_cls_logits_levels_tuple)}, "
                # f"obj={len(raw_obj_logits_levels_tuple)}, "
                f"reg={len(raw_reg_logits_levels_tuple)}"
            )

        decoded_boxes_all_levels_list: List[torch.Tensor] = []
        decoded_scores_all_levels_list: List[torch.Tensor] = []

        for i in range(self.nl):
            cls_l = raw_cls_logits_levels_tuple[i]
            # obj_l = raw_obj_logits_levels_tuple[i]
            reg_l = raw_reg_logits_levels_tuple[i]

            # boxes_level, scores_level = self._decode_predictions_for_level_onnx(cls_l, obj_l, reg_l, i)
            boxes_level, scores_level = self._decode_predictions_for_level_onnx(cls_l, reg_l, i)
            decoded_boxes_all_levels_list.append(boxes_level)
            decoded_scores_all_levels_list.append(scores_level)

        batched_all_boxes = torch.cat(decoded_boxes_all_levels_list, dim=1)
        batched_all_scores = torch.cat(decoded_scores_all_levels_list, dim=1)

        return batched_all_boxes, batched_all_scores


class ONNXExportablePicoDet(nn.Module):
    def __init__(self,
                 quantized_core_model: nn.Module,
                 head_postprocessor: PostprocessorForONNX):
        super().__init__()
        self.core_model = quantized_core_model
        self.postprocessor = head_postprocessor

    def forward(self, x: torch.Tensor):
        is_training_before = self.core_model.training
        self.core_model.train()
        raw_feature_outputs_tuple = self.core_model(x)
        self.core_model.training = is_training_before

        # The postprocessor takes this tuple and decodes it to the inference outputs
        return self.postprocessor(raw_feature_outputs_tuple)

# ────────────────── append_nms_to_onnx ────────────────────
def append_nms_to_onnx(
        in_path: str,
        out_path: str,
        score_thresh: float,
        iou_thresh: float,
        max_det: int,
        *,
        raw_boxes: str = "raw_boxes",     # [B , A , 4]
        raw_scores: str = "raw_scores",   # [B , A , C]
        top_k_before_nms: bool = True,
        k_value: int = 600,
):
    m = onnx.load(in_path)
    g = m.graph

    # ───────── constants ─────────
    g.initializer.extend([
        oh.make_tensor("nms_iou_th",   TP.FLOAT, [], [iou_thresh]),
        oh.make_tensor("nms_score_th", TP.FLOAT, [], [score_thresh]),
        oh.make_tensor("nms_max_det",  TP.INT64, [], [max_det]),
        # oh.make_tensor("nms_axis0", TP.INT64, [1], [0]),
        oh.make_tensor("nms_axis1", TP.INT64, [1], [1]),
        oh.make_tensor("nms_axis2", TP.INT64, [1], [2]),
        oh.make_tensor("nms_shape_boxes3d",  TP.INT64, [3], [0, -1, 4]),
        oh.make_tensor("nms_shape_scores3d", TP.INT64, [3], [0, 0, -1]),
    ])
    if top_k_before_nms:
        g.initializer.extend([
            oh.make_tensor("nms_k_topk", TP.INT64, [1], [k_value]),
        ])

    # ───────── reshape / transpose raw outputs ─────────
    boxes3d = "nms_boxes3d"
    g.node.append(oh.make_node(
        "Reshape", [raw_boxes, "nms_shape_boxes3d"], [boxes3d],
        name="nms_Reshape_Boxes3D"))

    scores3d = "nms_scores3d"
    g.node.append(oh.make_node(
        "Reshape", [raw_scores, "nms_shape_scores3d"], [scores3d],
        name="nms_Reshape_Scores3D"))

    scores_bca = "nms_scores_bca"
    g.node.append(oh.make_node(
        "Transpose", [scores3d], [scores_bca],
        perm=[0, 2, 1], name="nms_Transpose_BCA"))  # [B , C , A]

    # ───────── optional Top-K filter ─────────
    if top_k_before_nms:
        max_conf = "nms_max_conf"
        g.node.append(oh.make_node(
            "ReduceMax", [scores_bca, "nms_axis1"], [max_conf],
            keepdims=0, name="nms_ReduceMax"))

        topk_vals = "nms_topk_vals"
        topk_idx  = "nms_topk_idx"                 # [B , K]
        g.node.append(oh.make_node(
            "TopK",
            [max_conf, "nms_k_topk"],
            [topk_vals, topk_idx],
            axis=1, largest=1, sorted=0,
            name="nms_TopK"))

        topk_idx_unsq = "nms_topk_idx_unsq"        # [B , K , 1]
        g.node.append(oh.make_node(
            "Unsqueeze", [topk_idx, "nms_axis2"], [topk_idx_unsq],
            name="nms_UnsqTopKIdx"))

        boxes_topk = "nms_boxes_topk"              # [B , K , 4]
        g.node.append(oh.make_node(
            "GatherND", [boxes3d, topk_idx_unsq], [boxes_topk],
            batch_dims=1, name="nms_GatherBoxesTopK"))

        scores_bac = "nms_scores_bac"
        g.node.append(oh.make_node(
            "Transpose", [scores_bca], [scores_bac],
            perm=[0, 2, 1], name="nms_Transpose_BAC"))

        scores_bkc = "nms_scores_bkc"              # [B , K , C]
        g.node.append(oh.make_node(
            "GatherND", [scores_bac, topk_idx_unsq], [scores_bkc],
            batch_dims=1, name="nms_GatherScoresTopK"))

        scores_bck = "nms_scores_bck"              # [B , C , K]
        g.node.append(oh.make_node(
            "Transpose", [scores_bkc], [scores_bck],
            perm=[0, 2, 1], name="nms_Transpose_BCK"))

        nms_boxes  = boxes_topk
        nms_scores = scores_bck
    else:
        nms_boxes  = boxes3d
        nms_scores = scores_bca

    # ───────── Non-Max Suppression ─────────
    sel = "nms_selected"                          # [N , 3]
    g.node.append(oh.make_node(
        "NonMaxSuppression",
        [nms_boxes, nms_scores,
         "nms_max_det", "nms_iou_th", "nms_score_th"],
        [sel], name="nms_NMS"))

    # split indices (batch , class , anchor)
    g.initializer.extend([
        oh.make_tensor("nms_split111", TP.INT64, [3], [1, 1, 1]),
    ])
    b_col, c_col, a_col = "nms_b", "nms_c", "nms_a"
    g.node.append(oh.make_node(
        "Split", [sel, "nms_split111"], [b_col, c_col, a_col],
        axis=1, name="nms_SplitSel"))

    # squeeze to 1-D
    b_idx, class_idx, anc_idx = "batch_idx", "class_idx", "anchor_idx"
    for src, dst in [(b_col, b_idx), (c_col, class_idx), (a_col, anc_idx)]:
        g.node.append(oh.make_node(
            "Squeeze", [src, "nms_axis1"], [dst],
            name=f"nms_Squeeze_{dst}"))

    # ───── gather det_boxes  (batch_dims=1, indices=[anchor]) ─────
    a_unsq = "nms_a_unsq"
    g.node.append(oh.make_node(
        "Unsqueeze", [anc_idx, "nms_axis1"], [a_unsq],
        name="nms_UnsqAnchor"))

    det_boxes = "det_boxes"
    g.node.append(oh.make_node(
        "GatherND", [nms_boxes, a_unsq], [det_boxes],
        batch_dims=1, name="nms_GatherDetBoxes"))

    # ───── gather det_scores  (batch_dims=1, indices=[class,anchor]) ─────
    cls_unsq = "nms_cls_unsq"
    g.node.append(oh.make_node(
        "Unsqueeze", [class_idx, "nms_axis1"], [cls_unsq],
        name="nms_UnsqClass"))

    idx_scores = "nms_idx_scores"                 # [N , 2]
    g.node.append(oh.make_node(
        "Concat", [cls_unsq, a_unsq], [idx_scores],
        axis=1, name="nms_CatClassAnchor"))

    det_scores = "det_scores"
    g.node.append(oh.make_node(
        "GatherND", [nms_scores, idx_scores], [det_scores],
        batch_dims=1, name="nms_GatherDetScores"))

    # ───────── declare final outputs ─────────
    del g.output[:]   # remove existing outputs
    g.output.extend([
        oh.make_tensor_value_info(det_boxes,  TP.FLOAT, ['N', 4]),
        oh.make_tensor_value_info(det_scores, TP.FLOAT, ['N']),
        oh.make_tensor_value_info(class_idx,    TP.INT64, ['N']),
        oh.make_tensor_value_info(b_idx,      TP.INT64, ['N']),
    ])

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"[SAVE] Final ONNX with NMS → {out_path}")

def save_fp32_onnx_reference(model, cfg):
    """
    Export a 'lower risk' FP32 ONNX model for comparison.
    This uses the model directly in eval mode without complex wrappers.
    """
    print("[INFO] Exporting FP32 reference ONNX model...")
    model.cpu().eval()
    
    # Create a simple wrapper that matches the expected ONNX output format
    class FP32ReferenceWrapper(nn.Module):
        def __init__(self, fp32_model):
            super().__init__()
            self.model = fp32_model
            
        def forward(self, x):
            # Model in eval mode returns (boxes, scores) directly
            return self.model(x)
    
    fp32_wrapper = FP32ReferenceWrapper(model)
    dummy_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    
    fp32_onnx_path = cfg.out.replace(".onnx", "_fp32_reference.onnx")
    
    torch.onnx.export(
        fp32_wrapper,
        dummy_input,
        fp32_onnx_path,
        input_names=['images_uint8'],
        output_names=['raw_boxes', 'raw_scores'],
        dynamic_axes={
            'images_uint8': {0: 'batch', 2: 'h', 3: 'w'},
            'raw_boxes':    {0: 'batch', 1: 'anchors'},
            'raw_scores':   {0: 'batch', 1: 'anchors'}
        },
        opset_version=19,  # 14 for hard-swish
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
    )
    print(f'[SAVE] FP32 reference ONNX → {fp32_onnx_path}')
    
    # Also create NMS version of FP32 model
    fp32_nms_path = cfg.out.replace(".onnx", "_fp32_reference_with_nms.onnx")
    append_nms_to_onnx(
        in_path=fp32_onnx_path,
        out_path=fp32_nms_path,
        score_thresh=float(model.head.score_th),
        iou_thresh=float(model.head.iou_th),
        max_det=int(model.head.max_det),
    )
    print(f'[SAVE] FP32 reference ONNX with NMS → {fp32_nms_path}')
    return fp32_onnx_path, fp32_nms_path


def save_intermediate_onnx(qat_model, cfg, model):
    # --- Convert QAT model to INT8 ---
    qat_copy = copy.deepcopy(qat_model).cpu().eval()

    int8_model_with_preprocessor = convert_fx(qat_copy) # This model still contains 'pre'
    print("[INFO] QAT model converted to INT8.")

    # ---------------- Create the wrapper for ONNX export ----------------
    if not hasattr(model, 'head') or not isinstance(model.head, PicoDetHead):
        raise RuntimeError("Original FP32 model or its head is not available for PostprocessorForONNX initialization.")

    # It's crucial that model.head's buffers (strides, dfl_project, anchors) are correctly initialized
    # for the IMG_SIZE used during training. This should already be the case.
    onnx_postprocessor = PostprocessorForONNX(model.head) # Pass the head of the original FP32 model
    
    # Create the final model that combines the quantized core and the new postprocessor
    final_exportable_int8_model = ONNXExportablePicoDet(
        int8_model_with_preprocessor, # The FX GraphModule from convert_fx
        onnx_postprocessor
    )
    final_exportable_int8_model.cpu().eval()

    # ---------------- ONNX export (model WITH preprocessor AND wrapped postprocessor, without NMS) ----------------
    temp_onnx_path = cfg.out.replace(".onnx", "_temp_no_nms.onnx")

    dummy_uint8_input_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8, device='cpu')
    actual_onnx_input_example = dummy_uint8_input_cpu.cpu() # uint8 tensor

    print("[INFO] Exporting final INT8 model with wrapped postprocessing to ONNX...")
    torch.onnx.export(
        final_exportable_int8_model,
        actual_onnx_input_example,
        temp_onnx_path,
        input_names=['images_uint8'],
        output_names=['raw_boxes', 'raw_scores'],
        dynamic_axes={
            'images_uint8': {0: 'batch', 2: 'h', 3: 'w'},
            'raw_boxes':    {0: 'batch', 1: 'anchors'}, # Shape [batch, total_anchors, 4]
            'raw_scores':   {0: 'batch', 1: 'anchors'}  # Shape [batch, total_anchors, num_classes]
                                                        # The 3rd dim (num_classes) is static
        },
        opset_version=18,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
    )
    print(f'[SAVE] Intermediate ONNX (no NMS, with wrapped postprocessor) → {temp_onnx_path}')
    return final_exportable_int8_model, int8_model_with_preprocessor, actual_onnx_input_example, temp_onnx_path


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: argparse.Namespace, out_path: str):
    """
    Saves a comprehensive checkpoint.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer state to save.
        epoch: The current epoch number.
        cfg: The argparse configuration namespace.
        out_path: The path to save the checkpoint file.
    """
    # 1. Create a configuration dictionary from the model's parameters
    # This metadata is the key to flexible loading.
    model_config = {
        'img_size': model.head.img_size, # Or get from model.pre.size
        'arch': cfg.arch, # Assumes arch is in your config
        'feat_chs': model.neck.in_chs,
        'num_classes': model.head.nc,
        'neck_out_ch': model.neck.out[0].cv3.out_channels,
        'head_reg_max': model.head.reg_max,
        'cls_conv_depth': model.head.cls_conv_depth,
        'lat_k': model.neck.lat[0].dw.kernel_size[0],
    }

    # 2. Bundle everything into a checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # 3. Save the checkpoint
    torch.save(checkpoint, out_path)
    print(f"[SAVE] Checkpoint saved to {out_path} (Epoch: {epoch})")


def load_checkpoint(new_model: nn.Module, ckpt_path: str, optimizer: torch.optim.Optimizer | None = None, device: torch.device = 'cpu'):
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint file not found at {ckpt_path}. Starting from scratch.")
        return 0

    print(f"[LOAD] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_config = ckpt.get('model_config', {})
    model_sd = ckpt['model_state_dict']

    new_model_config = {
        'img_size': new_model.head.img_size,
        'head_reg_max': new_model.head.reg_max,
        'neck_out_ch': new_model.neck.out[0].cv3.out_channels,
        'num_classes': new_model.head.nc,
    }

    is_finetuning = (
        saved_config.get('img_size')     != new_model_config['img_size'] or
        saved_config.get('head_reg_max') != new_model_config['head_reg_max'] or
        saved_config.get('neck_out_ch')  != new_model_config['neck_out_ch']  or
        saved_config.get('num_classes')  != new_model_config['num_classes']
    )

    start_epoch = ckpt.get('epoch', 0) + 1

    if is_finetuning:
        print("[INFO] Mismatch detected between saved and new model configurations.")
        print("         Mode: Finetuning. Loading weights with strict=False.")
        print(f"         Saved config:   img_size={saved_config.get('img_size')}, reg_max={saved_config.get('head_reg_max')}")
        print(f"         Current config: img_size={new_model_config['img_size']}, reg_max={new_model_config['head_reg_max']}")

        # 1) Remove incompatible class head tensors (80→3)
        pruned_sd = {}
        for k, v in model_sd.items():
            if k.startswith('head.cls_pred.'):
                # drop these so shape mismatch doesn't error
                continue
            pruned_sd[k] = v

        # If your old checkpoint had an objectness branch you removed, ignore it too:
        # if k.startswith('head.obj_pred.'): continue

        missing_keys, unexpected_keys = new_model.load_state_dict(pruned_sd, strict=False)

        print("\n--- Weight Loading Report (finetune) ---")
        if missing_keys:
            print(f"Missing (expected): {len(missing_keys)} keys (e.g. class head)")
        if unexpected_keys:
            print(f"Unexpected (ignored): {len(unexpected_keys)} keys")
        print("----------------------------------------\n")

        # 2) Re-init the new class head (optional but recommended)
        with torch.no_grad():
            for m in new_model.head.cls_pred:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    # modestly negative prior works well for one-stage detectors
                    m.bias.fill_(-2.0)

        print("[INFO] Optimizer state not loaded. Starting finetuning from epoch 0.")
        return 0

    else:
        print("[INFO] Configurations match. Resuming training.")
        new_model.load_state_dict(model_sd, strict=True)
        if optimizer:
            print("[INFO] Loading optimizer state.")
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("[WARN] No optimizer provided. Optimizer state not loaded.")
        print(f"[INFO] Resuming from epoch {start_epoch}.")
        return start_epoch


def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', default='mnv4c', choices=['mnv3', 'mnv4s', 'mnv4', 'mnv4c'])
    pa.add_argument('--epochs', type=int, default=25) 
    pa.add_argument('--batch', type=int, default=64)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_v5_int8.onnx')
    pa.add_argument('--no_inplace_head_neck', default=True, action='store_true', help="Disable inplace activations in head/neck")
    pa.add_argument('--min_box_size', type=int, default=11, help="Minimum pixel width/height for a GT box to be used in training.")
    pa.add_argument('--load_from', type=str, default="picodet_50coco.pt", help="Path to a checkpoint to resume or finetune from (e.g., picodet_50coco.pt)")
    pa.add_argument('--data_root', default='webextension')
    pa.add_argument('--ann', default='webextension/annotations/instances.json')
    pa.add_argument('--val_pct', type=float, default=0.05)
    cfg = pa.parse_args(argv)

    TRAIN_SUBSET = None
    VAL_SUBSET   = None
    debug_prints = True
    BACKBONE_FREEZE_EPOCHS = 2 if cfg.epochs < 12 else 3  # 0 to disable
    use_focal_loss = False
    FOCAL_LOSS_WARMUP_EPOCHS = 10

    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(cfg.device)
    print(f'[INFO] device = {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, ckpt=None, img_size=IMG_SIZE) # Pass img_size
    reg_conv_depth = 2
    if IMG_SIZE < 320:
        out_ch = 96
        lat_k = 5
        cls_conv_depth = 3
    elif IMG_SIZE < 512:
        out_ch = 96
        lat_k = 5
        cls_conv_depth = 3
    else:
        # for out_ch choose a number divisible by 2, 8
        # large out_ch can often take a smaller conv_depth
        out_ch = 120  # 144
        lat_k = 5
        cls_conv_depth = 3
        # assigner ctr often needs to be larger with larger images as well
    gamma_loss = 2.0
    alpha_loss = 0.75
    quality_floor_vfl = 0.04
    q_gamma = 0.5
    CLS_WEIGHT = 2.0
    IOU_WEIGHT = 2.0

    # Load data
    root = cfg.coco_root
    coco_train_raw = CocoDetection(
        f"{root}/train2017",
        f"{root}/annotations/instances_train2017.json",
    )
    coco_label_map = CANONICAL_COCO80_MAP

    use_coco = True
    if not use_coco:
        NUM_CLASS = 3
        root = cfg.data_root
        ann_path = cfg.ann
        label_map, id2name = build_label_map_and_names_from_ann(ann_path)
        coco_label_map = label_map   # just a name reuse
        # Build two dataset objects so they can have different transforms
        full_train_tf = build_transforms((IMG_SIZE, IMG_SIZE), train=True)
        full_val_tf   = build_transforms((IMG_SIZE, IMG_SIZE), train=False)
        
        full_trainval_ds = CocoDetectionV2(
            img_dir=f"{root}/images",
            ann_file=ann_path,
            lb_map=label_map,
            transforms=None  # transform applied after subset selection
        )
        
        # Random split (deterministic)
        g = torch.Generator().manual_seed(SEED)
        n = len(full_trainval_ds)
        val_n = max(1, int(cfg.val_pct * n))
        perm = torch.randperm(n, generator=g)
        val_idx   = perm[:val_n].tolist()

        train_ds = CocoDetectionV2(f"{root}/images", ann_path, label_map, transforms=full_train_tf)
        val_ds = torch.utils.data.Subset(
            CocoDetectionV2(f"{root}/images", ann_path, label_map, transforms=full_val_tf),
            val_idx
        )
    else:
        NUM_CLASS = 80
        train_ds = CocoDetectionV2(
            f"{root}/train2017",
            f"{root}/annotations/instances_train2017.json",
            coco_label_map,
            transforms=build_transforms((IMG_SIZE, IMG_SIZE), train=True),
        )
        val_ds   = CocoDetectionV2(
            f"{root}/val2017",
            f"{root}/annotations/instances_val2017.json",
            coco_label_map,
            transforms=build_transforms((IMG_SIZE, IMG_SIZE), train=False),
        )

    model = PicoDet(
        backbone, 
        feat_chs,
        num_classes=NUM_CLASS, 
        neck_out_ch=out_ch,  # 96
        img_size=IMG_SIZE,
        head_reg_max=9 if IMG_SIZE < 320 else int((2 * math.ceil(IMG_SIZE / 128) + 3)),
        reg_conv_depth=reg_conv_depth,
        cls_conv_depth=cls_conv_depth,
        lat_k=lat_k,
        inplace_act_for_head_neck=not cfg.no_inplace_head_neck # Control from arg
    ).to(dev)
    
    # ------- (optional) random subset for quick runs -------
    train_sampler = None
    if TRAIN_SUBSET is not None:
        train_sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=False,
            num_samples=min(TRAIN_SUBSET, len(train_ds))
        )
    val_sampler = None
    if VAL_SUBSET is not None:
        g = torch.Generator().manual_seed(42)
        val_idx  = torch.randperm(len(val_ds), generator=g)[:VAL_SUBSET].tolist()
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    if cfg.min_box_size is not None and cfg.min_box_size > 1:
        print(f"[INFO] Filtering enabled: Dropping GT boxes smaller than {cfg.min_box_size}x{cfg.min_box_size} pixels.")
        # Use functools.partial to create a collate_fn with our argument "baked in"
        custom_collate_fn = functools.partial(collate_and_filter_more, min_box_size=cfg.min_box_size)
    else:
        print("[INFO] GT box filtering disabled.")
        custom_collate_fn = collate_v2

    # Curriculum learning, gradually smaller boxes included
    curriculum_schedule = [
        (18, 24),  # Stage 1: For the first N epochs, only use boxes >= 24x24 pixels.
        (2, 22),
        (2, 20),
        (2, 18),
        (20, 16),   # Stage 2: For the next 20 epochs
        (0, cfg.min_box_size)     # Stage 3: For the rest of training, use boxes >= cfg.min_box_size.
                   # (Epoch count of 0 or a large number signifies 'the rest')
    ]
    current_stage_idx = 0
    epochs_in_current_stage = 0
    # Create a dictionary of DataLoaders, one for each stage of the curriculum
    train_loaders = {}
    total_scheduled_epochs = 0
    for i, (num_epochs, min_size) in enumerate(curriculum_schedule):
        collate_fn = functools.partial(collate_and_filter_more, min_box_size=min_size)
        
        # Use the existing train_sampler logic
        train_loaders[i] = DataLoader(
            train_ds, batch_size=cfg.batch, sampler=train_sampler,
            num_workers=cfg.workers,
            collate_fn=collate_fn,
            pin_memory=True, persistent_workers=bool(cfg.workers)
        )
        if i < len(curriculum_schedule) - 1:
            total_scheduled_epochs += num_epochs
    """
    tr_loader = DataLoader(
        train_ds, batch_size=cfg.batch, sampler=train_sampler,
        num_workers=cfg.workers, collate_fn=custom_collate_fn,
        pin_memory=True, persistent_workers=bool(cfg.workers)
    )
    """
    vl_loader = DataLoader(
        val_ds,   batch_size=cfg.batch, sampler=val_sampler,
        num_workers=cfg.workers, collate_fn=custom_collate_fn,
        pin_memory=True, persistent_workers=bool(cfg.workers)
    )
    
    print(f"[INFO] COCO contiguous label map built – {len(coco_label_map)} classes")
    if len(coco_label_map) != model.head.nc:
        print(f"[WARN] num classes in map ({len(coco_label_map)}) ≠ model.head.nc ({model.head.nc})")

    id2name = {v: coco_train_raw.coco.loadCats([k])[0]["name"] for k, v in coco_label_map.items()}  # noqa

    if True:
        peak_lr_adamw = 1e-4
        weight_decay_adamw = 0.02
        beta1_adamw = 0.9
        beta2_adamw = 0.999
        eps_adamw = 1e-8

        if cfg.epochs < 5:
            warmup_epochs_adamw = 1
        elif cfg.epochs <= 30:
            warmup_epochs_adamw = max(1, cfg.epochs // 4)
        else:
            warmup_epochs_adamw = 6
        cosine_decay_to_fraction_adamw = 0.01 # Decay to 1% of peak_lr_adamw

        # Parameter groups for differential weight decay
        param_groups = [
            {'params': [], 'weight_decay': weight_decay_adamw, 'name': 'decay'},
            {'params': [], 'weight_decay': 0.0, 'name': 'no_decay'}
        ]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Norm layers, biases, and sometimes embedding layers don't use weight decay
            if ('cls_pred' in name) or ('obj_pred' in name):
                param_groups[1]['params'].append(param)   # ← goes to no-decay bucket
            elif len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name.lower():  #  or "bn" in name.lower()
                param_groups[1]['params'].append(param)
            else:
                param_groups[0]['params'].append(param)
        
        print(f"[INFO] AdamW: {len(param_groups[0]['params'])} params with WD, {len(param_groups[1]['params'])} params without WD.")
        
        opt = AdamW(
            param_groups,
            lr=peak_lr_adamw, # Optimizer's lr is the peak LR
            betas=(beta1_adamw, beta2_adamw),
            eps=eps_adamw
        )

        # --- Schedulers (Epoch-based for simpler integration) ---
        warmup_scheduler = LinearLR(
            opt,
            start_factor=0.01, # Start at 1% of peak_lr_adamw
            end_factor=1.0,    # End at 100% of peak_lr_adamw
            total_iters=warmup_epochs_adamw # Number of *epochs* for warmup
        )
        cosine_t_max_epochs = max(1, cfg.epochs - warmup_epochs_adamw)
        cosine_scheduler = CosineAnnealingLR(
            opt,
            T_max=cosine_t_max_epochs, # Number of *epochs* for the cosine decay phase
            eta_min=peak_lr_adamw * cosine_decay_to_fraction_adamw # Absolute minimum LR
        )
        sch = SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs_adamw]
        )
        
        print(f"[INFO] Using AdamW with peak_lr={peak_lr_adamw:.1e}, warmup_epochs={warmup_epochs_adamw}.")
        print(f"       Cosine T_max_epochs={cosine_t_max_epochs}, eta_min={peak_lr_adamw * cosine_decay_to_fraction_adamw:.1e}")
    else:
        base_lr = 0.006
        warmup_epochs = 3
        cosine_decay_alpha = 0.04
        opt = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        warmup_scheduler = LinearLR(
            opt,
            start_factor=1e-5,
            end_factor=1.0,    # End at the peak_lr
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            opt,
            T_max=cfg.epochs - warmup_epochs, # Remaining epochs after warm-up
            eta_min=base_lr * cosine_decay_alpha # This will be 0 if cosine_decay_alpha is 0.0
        )
        sch = SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')

    assigner = SimOTACache(
        nc=model.head.nc,
        ctr=4.0,  # 2.5,   1.5 might be better later for strict bounding
        topk=10,  # 10
        cls_cost_weight=3.0,  # Increased to emphasize classification
        debug_epochs=5 if debug_prints else 0,
        dynamic_k_min=2,
    )
    # ── Resume / Finetune from checkpoint (optional) ─────────────────
    start_epoch = 0
    if cfg.load_from:
        start_epoch = load_checkpoint(model, cfg.load_from, opt, dev)  # returns next epoch index
        # Align the LR scheduler with the starting epoch
        # (SequentialLR respects last_epoch; set and do NOT step here)
        sch.last_epoch = start_epoch - 1
        # Align the curriculum stage so the right DataLoader is used
        def _curriculum_stage_for_epoch(ep:int):
            passed = 0
            for i, (num_epochs, _min_size) in enumerate(curriculum_schedule):
                if i == len(curriculum_schedule) - 1:   # last stage "the rest"
                    return i, max(0, ep - passed)
                if ep < passed + num_epochs:
                    return i, ep - passed
                passed += num_epochs
            return len(curriculum_schedule) - 1, 0

        current_stage_idx, epochs_in_current_stage = _curriculum_stage_for_epoch(start_epoch)
        print(f"[RESUME] start_epoch={start_epoch}, curriculum stage={current_stage_idx}, "
              f"epoch-in-stage={epochs_in_current_stage}")

    original_model_head_nc = model.head.nc
    original_model_head_reg_max = model.head.reg_max
    original_dfl_project_buffer = model.head.dfl_project_buffer
    training_history = {}

    # ... (FP32 training loop) ...
    fp32_epochs = cfg.epochs
    if start_epoch > fp32_epochs:
        fp32_epochs = start_epoch + 2
    for ep in range(start_epoch, fp32_epochs):
        if ep < BACKBONE_FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = False
            if ep == 0:
                print(f"[INFO] Backbone frozen for {BACKBONE_FREEZE_EPOCHS} epochs…")
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
            if ep == BACKBONE_FREEZE_EPOCHS:
                print("[INFO] Backbone unfrozen – full network now training")

        # Check if we need to advance to the next stage in the curriculum
        if current_stage_idx < len(curriculum_schedule) - 1:
            stage_duration, _ = curriculum_schedule[current_stage_idx]
            if epochs_in_current_stage >= stage_duration:
                current_stage_idx += 1
                epochs_in_current_stage = 0
                print("-" * 50)
                print(f"[INFO] Curriculum Change: Entering Stage {current_stage_idx + 1} at Epoch {ep + 1}")
                print(f"[INFO] New min_box_size: {curriculum_schedule[current_stage_idx][1]}")
                print("-" * 50)
        
        # Select the correct loader for the current stage
        tr_loader = train_loaders[current_stage_idx]
        epochs_in_current_stage += 1
        ##########
        if FOCAL_LOSS_WARMUP_EPOCHS is not None:
            use_focal_loss_for_epoch = ep < FOCAL_LOSS_WARMUP_EPOCHS
            if ep == FOCAL_LOSS_WARMUP_EPOCHS:
                print("[INFO] Switching from Focal Loss warmup to Varifocal Loss for subsequent epochs.")
                quality_floor_vfl = quality_floor_vfl * 2
            elif ep == (FOCAL_LOSS_WARMUP_EPOCHS + 1):
                quality_floor_vfl = quality_floor_vfl / 2
        else:
            use_focal_loss_for_epoch = use_focal_loss
        if ep < 2:
            assigner.k = 12
            assigner.dynamic_k_min = 5
            assigner.cls_cost_weight = 2.0
            assigner.r = 5.25
            CLS_WEIGHT = 0.4
            IOU_WEIGHT = 4.0
        elif ep < 4:
            assigner.cls_cost_weight = 2.2
            CLS_WEIGHT = 0.5
        elif ep < 6:
            assigner.dynamic_k_min = 4
            assigner.r = 4.6
            assigner.k = 10
            assigner.cls_cost_weight = 3.0
            CLS_WEIGHT = 1.0
        elif ep < 8:
            assigner.r = 4.0
            assigner.dynamic_k_min = 2
            assigner.cls_cost_weight = 3.5
            CLS_WEIGHT = 1.8
        elif ep < 10:
            CLS_WEIGHT = 2.5
        elif ep == 14:
            IOU_WEIGHT = 2.5
        elif ep == 15:
            assigner.cls_cost_weight = 4.0
            CLS_WEIGHT = 3.0
        elif ep == 17:
            quality_floor_vfl = 0.02
        elif ep == 22:
            assigner.dynamic_k_min = 1
            quality_floor_vfl = 0.005
        elif ep == 55:
            q_gamma = 0.4
        elif ep == 60:
            assigner.r = 3.0
        elif ep == 65 and assigner.mean_fg_iou > 0.45:
            q_gamma = 0.3
        elif ep == 70:
            gamma_loss = 2.25
        elif ep == 80:
            assigner.cls_cost_weight = 4.0
        elif ep > 100:
            assigner.dynamic_k_min = 1
            assigner.r = 2.5
            CLS_WEIGHT = 4.0

        if assigner.mean_fg_iou < 0.35 or ep < 5:
            assigner.power = 0.0
        elif assigner.mean_fg_iou < 0.45:
            assigner.power = min(1.0, assigner.power + 0.20)
            print(f"epoch {ep} and assigner IoU weighting power is {assigner.power}")
        else:
            assigner.power = min(2.0, assigner.power + 0.25)
            print(f"epoch {ep} and assigner IoU weighting power is {assigner.power}")

        model.train()
        l, diag = train_epoch(
            model, tr_loader, opt, scaler, assigner, dev, ep, coco_label_map,
            head_nc_for_loss=original_model_head_nc,
            head_reg_max_for_loss=original_model_head_reg_max,
            dfl_project_buffer_for_decode=original_dfl_project_buffer,
            max_epochs=fp32_epochs, # Pass total epochs for VFL alpha scheduling
            quality_floor_vfl=quality_floor_vfl,  # try sometime 0.15
            debug_prints=debug_prints,
            use_focal_loss=use_focal_loss_for_epoch,
            w_cls_loss=CLS_WEIGHT,
            w_iou_loss=IOU_WEIGHT,
            gamma_loss=gamma_loss,
            alpha_loss=alpha_loss,
            q_gamma=q_gamma,
        )
        model.eval()
        try:
            epoch_metrics = run_epoch_validation(model, vl_loader, dev, ep + 1, model.head)
            epoch_metrics['train_loss'] = l if l is not None else -1.0
            epoch_metrics.update(diag)
            training_history[ep + 1] = epoch_metrics
            current_lr = opt.param_groups[0]['lr']
            iou_05 = epoch_metrics.get('iou_at_5', 0.0)
            iou_25 = epoch_metrics.get('iou_at_25', 0.0)
            print(f"Epoch {ep + 1}/{fp32_epochs} | Loss: {l:.4f} | IoU@.05: {iou_05:.3f} | IoU@.25: {iou_25:.3f} | LR: {current_lr:.6f}\n")
            print("─" * 15 + f" SimOTA Report for Epoch {ep + 1} " + "─" * 15)
            assigner.print_debug_report()
            assigner.print_classification_debug_report()
            print("─" * (75 + len(str(ep + 1))) + "\n")
            if ep == 10:
                plot_training_history(training_history, title="PicoDet Training Progress EP10")
        except Exception as e:
            print(repr(e))

        sch.step()
        # print(f"[INFO] Logit scaler value: {model.head.logit_scale.item():.4f}")
        assigner._dbg_iter = 0   
        
        ckpt_path = "picodet.pt"
        save_checkpoint(model, opt, ep + 1, cfg, ckpt_path)

    print("[INFO] Evaluating FP32 model...")
    model.eval()
    try:
        iou_05, acc = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.05,
                               iou_thresh=model.head.iou_th,
                               max_detections=model.head.max_det,
                               epoch_num=ep,
                               run_name="score_thresh_0.05",
                               debug_prints=debug_prints,
                               )
        print(f"[INFO] Validation IoU (score_th=0.05): {iou_05:.4f}")
        
        # Run for score_thresh = 0.25
        iou_25, acc = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.25,
                               iou_thresh=model.head.iou_th,
                               max_detections=model.head.max_det,
                               epoch_num=ep,
                               run_name="score_thresh_0.25",
                               debug_prints=debug_prints,
                               )
        print(f"[INFO] Validation IoU (score_th=0.25): {iou_25:.4f}")
    except Exception as e:
        print(repr(e))
    
    # --- Export FP32 reference ONNX for comparison ---
    print("[INFO] Exporting FP32 reference ONNX models...")
    fp32_onnx_path, fp32_nms_path = save_fp32_onnx_reference(model, cfg)
    
    model.train()

    # --- QAT Preparation ---
    print("[INFO] Preparing model for QAT...")
    dummy_uint8_input_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8, device='cpu')

    example_input_for_qat_entire_model = dummy_uint8_input_cpu.cpu()

    model.train()
    model.cpu()

    print("[INFO] Running qat_prepare...")
    # qat_prepare will trace the 'model', including 'model.pre'.
    # 'model.pre' will be skipped for quantization inserts due to set_module_name('pre', None)
    # but it will be part of the traced graph.
    qat_model = qat_prepare(model, example_input_for_qat_entire_model)
    qat_head = qat_model.head      # ObservedGraphModule
    orig_head = model.head         # FP32
    
    qat_head.nl = orig_head.nl
    qat_head.register_buffer('strides_buffer', orig_head.strides_buffer, persistent=False)
    for i in range(orig_head.nl):
        buf = getattr(orig_head, f'anchor_points_level_{i}')
        qat_head.register_buffer(f'anchor_points_level_{i}', buf, persistent=False)
    qat_head.register_buffer('dfl_project_buffer', orig_head.dfl_project_buffer, persistent=False)
    # qat_head.register_parameter(
    #     'logit_scale',
    #     nn.Parameter(orig_head.logit_scale.detach().clone(), requires_grad=True)
    # )
    qat_model = qat_model.to(dev)
    print("[INFO] QAT model prepared and moved to device.")

    # --- QAT Finetuning ---
    qat_model.train()

    qat_epochs = int(cfg.epochs * 0.2)
    qat_epochs = 3 if qat_epochs < 3 else qat_epochs

    qat_initial_lr = 0.0001
    
    # Filter parameters for QAT optimizer
    opt_q_params = filter(lambda p: p.requires_grad, qat_model.parameters())
    opt_q = SGD(opt_q_params, lr=qat_initial_lr, momentum=0.9, weight_decay=1e-5)
    scheduler_q = CosineAnnealingLR(opt_q, T_max=qat_epochs, eta_min=qat_initial_lr * 0.01)
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))

    print(f"[INFO] Starting QAT finetuning for {qat_epochs} epochs with initial LR {qat_initial_lr:.6f}...")
    
    best_qat_iou = -1.0 # To save the best QAT model
    final_exportable_int8_model = None

    for qep in range(qat_epochs):
        qat_model.train() # Ensure model is in train mode for each epoch
        current_lr_qat = opt_q.param_groups[0]['lr']
        print(f"[QAT] Starting Epoch {qep + 1}/{qat_epochs} with LR {current_lr_qat:.7f}")

        lq, diag = train_epoch(
            qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, coco_label_map,
            head_nc_for_loss=original_model_head_nc,
            head_reg_max_for_loss=original_model_head_reg_max,
            dfl_project_buffer_for_decode=original_dfl_project_buffer,
            max_epochs=qat_epochs, # For VFL alpha scheduling (relative to QAT duration)
            quality_floor_vfl=quality_floor_vfl,
            iou_weight_change_epoch=2,
            debug_prints=False,
            w_cls_loss=CLS_WEIGHT,
            w_iou_loss=IOU_WEIGHT,
            use_focal_loss=use_focal_loss,
            alpha_loss=alpha_loss,
            q_gamma=q_gamma,
        )
        scheduler_q.step() # Step the QAT LR scheduler

        if lq is not None:
            print(f'[QAT] Epoch {qep + 1}/{qat_epochs} Train Loss {lq:.3f}')
        else:
            print(f'[QAT] Epoch {qep + 1}/{qat_epochs} Train Loss N/A (no samples contributed)')

        # --- QAT Validation after each epoch ---
        print(f"[QAT] Evaluating after Epoch {qep + 1}...")
        qat_model.eval() # Switch qat_model to eval for validation

        try:
            eval_compatible_qat_model = ONNXExportablePicoDet(qat_model, PostprocessorForONNX(model.head))
            eval_compatible_qat_model.to(dev).eval() # Ensure it's on device and in eval mode

            # Using a consistent score_thresh for tracking QAT progress, e.g., 0.05 or 0.25
            # The one used for final ONNX NMS params can be different.
            current_qat_val_iou, acc = quick_val_iou(
                                   eval_compatible_qat_model, vl_loader, dev,
                                   score_thresh=0.05, # Consistent validation score_thresh
                                   iou_thresh=model.head.iou_th,
                                   max_detections=model.head.max_det,
                                   epoch_num=qep, # Pass QAT epoch number
                                   run_name=f"QAT_ep{qep+1}_score0.05",
                                   debug_prints=debug_prints,
                                )
            print(f"[QAT Eval] Epoch {qep + 1}/{qat_epochs} Val IoU (score_th=0.05): {current_qat_val_iou:.4f}  Acc {acc:.3f}")

            if current_qat_val_iou > best_qat_iou:
                best_qat_iou = current_qat_val_iou
                final_exportable_int8_model, int8_model_with_preprocessor, actual_onnx_input_example, temp_onnx_path = save_intermediate_onnx(
                    qat_model, cfg, model
                )
                print(f"[QAT] New best QAT validation IoU: {best_qat_iou:.4f}. Model saved.")

        except Exception as e:
            print(f"Error during QAT model validation (Epoch {qep + 1}): {e}")
            traceback.print_exc()

    print(f"[INFO] QAT finetuning completed. Best QAT Val IoU (score_th=0.05): {best_qat_iou:.4f}")


    if final_exportable_int8_model is None:
        final_exportable_int8_model, int8_model_with_preprocessor, actual_onnx_input_example, temp_onnx_path = save_intermediate_onnx(qat_model, cfg, model)

    if debug_prints:
        # DEBUG: Inspect intermediate ONNX model outputs
        intermediate_model_check = onnx.load(temp_onnx_path)
        print("[DEBUG] Intermediate ONNX model input ValueInfo:")
        for input_vi in intermediate_model_check.graph.input:
            if hasattr(input_vi.type, 'tensor_type'):
                tensor_type = input_vi.type.tensor_type
                elem_type = tensor_type.elem_type
                shape_dims = [str(d.dim_value) if d.dim_value else d.dim_param for d in tensor_type.shape.dim]
                print(f"  Name: {input_vi.name}, Type: tensor({onnx.TensorProto.DataType.Name(elem_type)}), Shape: {shape_dims}")
            else:
                print(f"  Name: {input_vi.name}, Type (raw): {input_vi.type.WhichOneof('value')}")
    
    
        print("[DEBUG] Intermediate ONNX model output ValueInfo:")
        for output_vi in intermediate_model_check.graph.output:
            if hasattr(output_vi.type, 'tensor_type'):
                tensor_type = output_vi.type.tensor_type
                elem_type = tensor_type.elem_type
                shape_dims = [str(d.dim_value) if d.dim_value else d.dim_param for d in tensor_type.shape.dim]
                print(f"  Name: {output_vi.name}, Type: tensor({onnx.TensorProto.DataType.Name(elem_type)}), Shape: {shape_dims}")
            elif hasattr(output_vi.type, 'sequence_type'):
                seq_type = output_vi.type.sequence_type
                if hasattr(seq_type.elem_type, 'tensor_type'):
                    tensor_type = seq_type.elem_type.tensor_type
                    elem_type = tensor_type.elem_type
                    shape_dims = [str(d.dim_value) if d.dim_value else d.dim_param for d in tensor_type.shape.dim]
                    print(f"  Name: {output_vi.name}, Type: seq(tensor({onnx.TensorProto.DataType.Name(elem_type)})), Shape_of_elem: {shape_dims}")
                else:
                    print(f"  Name: {output_vi.name}, Type: seq(<unknown_elem_type>)")
            else:
                print(f"  Name: {output_vi.name}, Type (raw int): {output_vi.type.WhichOneof('value')}")
    
        print("[DEBUG] Running PyTorch forward pass on int8_model_with_preprocessor...")
        try:
            int8_model_with_preprocessor.cpu().eval()
            actual_onnx_input_example_cpu = actual_onnx_input_example.cpu()
    
            py_outputs = int8_model_with_preprocessor(actual_onnx_input_example_cpu)
    
            print(f"  [PyTorch Output] Type of py_outputs: {type(py_outputs)}")
            if isinstance(py_outputs, (tuple, list)):
                print(f"  [PyTorch Output] Number of elements: {len(py_outputs)}")
                for i, item in enumerate(py_outputs):
                    print(f"    Element {i}: Type={type(item)}")
                    if isinstance(item, torch.Tensor):
                        print(f"      Shape={item.shape}, Dtype={item.dtype}")
                    elif isinstance(item, (list, tuple)):
                         print(f"      (Nested list/tuple) Length={len(item)}")
    
            elif isinstance(py_outputs, torch.Tensor):
                 print(f"  [PyTorch Output] (Single Tensor) Shape={py_outputs.shape}, Dtype={py_outputs.dtype}")
            else:
                print(f"  [PyTorch Output] Unexpected output type: {type(py_outputs)}")
    
        except Exception as e:
            print(f"  [PyTorch Output] Error during PyTorch forward: {e}")
            traceback.print_exc()
        
        print("[DEBUG] Inspecting output of int8_model_with_preprocessor directly...")
        int8_model_with_preprocessor.cpu().eval()
        dummy_input_for_inspection = actual_onnx_input_example.cpu()
        
        try:
            core_model_outputs = int8_model_with_preprocessor(dummy_input_for_inspection)
            print(f"  [Core Model Output] Type: {type(core_model_outputs)}")
            if isinstance(core_model_outputs, tuple):
                print(f"  [Core Model Output] Number of elements: {len(core_model_outputs)}")
                for i, item in enumerate(core_model_outputs):
                    if isinstance(item, torch.Tensor):
                        print(f"    Element {i}: Shape={item.shape}, Dtype={item.dtype}")
                    else:
                        print(f"    Element {i}: Type={type(item)}")
            elif isinstance(core_model_outputs, torch.Tensor):
                print(f"  [Core Model Output] (Single Tensor) Shape={core_model_outputs.shape}, Dtype={core_model_outputs.dtype}")
            else:
                print(f"  [Core Model Output] Unexpected output type: {type(core_model_outputs)}")
        except Exception as e:
            print(f"  [Core Model Output] Error during direct call: {e}")
            traceback.print_exc()
    
    
        print("[DEBUG] Running PyTorch forward pass on final_exportable_int8_model...")
        try:
            py_outputs_final_exportable = final_exportable_int8_model(actual_onnx_input_example.cpu())
            print(f"  [PyTorch Final Exportable Output] Type: {type(py_outputs_final_exportable)}")
            if isinstance(py_outputs_final_exportable, tuple) and len(py_outputs_final_exportable) == 2:
                print(f"    raw_boxes shape: {py_outputs_final_exportable[0].shape}, dtype: {py_outputs_final_exportable[0].dtype}")
                print(f"    raw_scores shape: {py_outputs_final_exportable[1].shape}, dtype: {py_outputs_final_exportable[1].dtype}")
            else:
                print("    Unexpected output structure from final_exportable_int8_model.")
        except Exception as e:
            print(f"  [PyTorch Final Exportable Output] Error during PyTorch forward: {e}")
            traceback.print_exc()

    # ---------------- Append NMS to the ONNX model ------------------------------
    out_dest = cfg.out
    # out_dest = "picodet_int8.onnx"
    # temp_onnx_path = "picodet_int8_temp_no_nms.onnx"
    append_nms_to_onnx(
        in_path=temp_onnx_path,
        out_path=out_dest,
        score_thresh=float(model.head.score_th), # 0.05
        iou_thresh=float(model.head.iou_th),  # 0.6
        max_det=int(model.head.max_det),  # 100
    )
    return training_history


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_training_history(history: dict, title: str = 'Training Progress'):
    """
    Plots the key metrics from a training‑history dictionary.

    Each epoch entry is expected to contain (at minimum)
    ------------------
      • train_loss
      • iou_at_05, iou_at_25
      • acc_at_05, acc_at_25
      • fg_per_img               (number of FG anchors per image)
      • centre_hit               (ratio 0‑1: how many GT boxes got ≥1 centre‑radius hit)
    """
    # ---------- gather ----------
    epochs = sorted(history)
    if not epochs:
        print("History is empty, cannot plot."); return

    as_list = lambda k: [history[e].get(k, float('nan')) for e in epochs]

    train_loss   = as_list('train_loss')
    iou_05, iou_25 = as_list('iou_at_5'), as_list('iou_at_25')
    acc_05, acc_25 = as_list('acc_at_5'), as_list('acc_at_25')
    fg_per_img   = as_list('fg_per_img')
    centre_hit   = as_list('centre_hit')

    # ---------- layout ----------
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 12), sharex=True,
        gridspec_kw={'height_ratios': [1.1, 1.2, 1.0]}
    )
    fig.suptitle(title, fontsize=16)

    # ── 1 · Loss ──────────────────────────────────────────────
    ax1.plot(epochs, train_loss, 'o-', color='tomato', label='Training Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(ls='--', alpha=.6)
    ax1.set_ylim(bottom=0 if all(l >= 0 for l in train_loss) else None)
    ax1.legend()

    # ── 2 · Val IoU / Acc ────────────────────────────────────
    ax2.plot(epochs, iou_05, 's--',  color='dodgerblue', label='IoU@0.05')
    ax2.plot(epochs, iou_25, '^-',   color='navy',       label='IoU@0.25')
    ax2.plot(epochs, acc_05, 's--',  color='limegreen',  label='Acc@0.05')
    ax2.plot(epochs, acc_25, '^-',   color='darkgreen',  label='Acc@0.25')
    ax2.set_ylabel('Metric (0‑1)')
    ax2.set_title('Validation IoU & Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(ls='--', alpha=.6)
    ax2.legend(loc='lower right')

    # ── 3 · Assigner diagnostics ─────────────────────────────
    #   left‑axis : FG / img   (bar)
    #   right‑axis: centre‑hit ratio (line)
    bars = ax3.bar(epochs, fg_per_img, width=.6, color='slategray',
                   alpha=.35, label='FG / img')
    ax3.set_ylabel('FG per image')
    ax3.set_title('SimOTA diagnostics')

    ax3_t = ax3.twinx()
    ax3_t.plot(epochs, centre_hit, 'o-', color='orange', label='Centre‑hit ratio')
    ax3_t.set_ylabel('Centre‑hit')
    ax3_t.set_ylim(0, 1)

    # combine legends from both axes
    handles, labels = ax3.get_legend_handles_labels()
    h2, l2 = ax3_t.get_legend_handles_labels()
    ax3_t.legend(handles + h2, labels + l2, loc='upper right')

    # ── global tweaks ────────────────────────────────────────
    ax3.set_xlabel('Epoch')
    for ax in (ax2, ax3): ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.minorticks_on()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    final_history = main()
    plot_training_history(final_history, title="PicoDet Training Progress")

    temp_onnx_path = "picodet_v5_int8_temp_no_nms.onnx"

    score_th = 0.5
    iou_th = 0.3
    max_det = 100
    out_dest = f'picodet_v5_int8_{str(score_th).replace(".", "_")}_{str(iou_th).replace(".", "_")}_{max_det}.onnx'
    append_nms_to_onnx(
        in_path=temp_onnx_path,
        out_path=out_dest,
        score_thresh=float(score_th), # 0.05
        iou_thresh=float(iou_th),  # 0.6
        max_det=max_det,  # 100
    )
