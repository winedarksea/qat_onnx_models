# train_picodet_qat.py – minimal pipeline: COCO ➜ FP32 ➜ QAT ➜ INT8 ➜ ONNX (with NMS)
# built on pytorch version 2.7
"""
Things to try adding:
1. Alternative Optimizer (AdamW)
2. Alternative Loss Function (Focal Loss / Quality Focal Loss)
3. Alpha Blending for Joint Logits
4. More Complex ONNX NMS Post-processing
"""
from __future__ import annotations
import argparse, random, time, warnings
from typing import List, Tuple
import traceback

import torch, torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.datasets import CocoDetection
from torchvision.tv_tensors import BoundingBoxes
import torchvision.ops as tvops
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F

from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfig, MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

import onnx
from onnx import TensorProto as TP, helper as oh

if False:
    try: 
        from picodet_lib import (
            PicoDet, get_backbone, VarifocalLoss, dfl_loss,
            build_dfl_targets
        )
    except Exception:
        pass

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 256

# ───────────────────── data & transforms ───────────────────────
CANONICAL_COCO80_IDS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
CANONICAL_COCO80_MAP: dict[int, int] = {coco_id: i for i, coco_id in enumerate(CANONICAL_COCO80_IDS)}

def _coco_to_tvt(annots, lb_map, canvas):
    boxes, labels = [], []
    W, H = canvas
    for a in annots:
        if a.get("iscrowd", 0): continue
        cid = a["category_id"]
        if cid not in lb_map: continue
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(lb_map[cid])
    if not boxes:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    bbx = BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(H, W))
    return {"boxes": bbx, "labels": torch.as_tensor(labels, dtype=torch.int64)}

class CocoDetectionV2(CocoDetection):
    def __init__(self, img_dir, ann_file, lb_map, transforms=None):
        super().__init__(img_dir, ann_file)
        self.lb_map = lb_map
        self._tf = transforms
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        tgt = _coco_to_tvt(anns, self.lb_map, img.size)
        if self._tf is not None: img, tgt = self._tf(img, tgt)
        return img, tgt

def build_transforms(size, train):
    aug = [
        T.ToImage(),
        T.RandomHorizontalFlip(0.25) if train else T.Identity(),
        T.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True) if train else T.Resize(size, antialias=True),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1) if train else T.Identity(),
        T.ToDtype(torch.uint8, scale=True),
    ]
    return T.Compose(aug)

def collate_v2(batch):
    return torch.stack(list(zip(*batch))[0], 0), list(zip(*batch))[1]

# ───────────────────── Unified Score Logic ────────────────
def logit_of_product(cls_raw: torch.Tensor, obj_raw: torch.Tensor) -> torch.Tensor:
    """Calculates the logit of the product of sigmoids, a.k.a. log(sigmoid(a)*sigmoid(b))"""
    p = torch.sigmoid(cls_raw) * torch.sigmoid(obj_raw)
    # Clamp to avoid log(0)
    eps = torch.finfo(p.dtype).eps
    return torch.log(p.clamp_min(eps)) - torch.log1p(-p.clamp_max(1 - eps))

# ───────────────────── assigner (SimOTA, REFACTORED) ────────────────
class SimOTAAssigner:
    def __init__(self,
                 nc: int,
                 ctr: float = 1.5, # Reduced radius, since we now require anchors to be inside the GT box
                 topk: int = 10,
                 cls_cost_weight: float = 1.0,
                 iou_cost_weight: float = 3.0,
                 debug_epochs: int = 0):
        self.nc = nc
        self.r = ctr
        self.k = topk
        self.cls_w = cls_cost_weight
        self.iou_w = iou_cost_weight
        self._dbg_mod = debug_epochs
        self._dbg_iter = 0

    @torch.no_grad()
    def __call__(self,
                 anchor_points: torch.Tensor,   # (A, 2)
                 anchor_strides: torch.Tensor,  # (A, 1)
                 tgt: dict,                     # {"boxes":(M,4), "labels":(M,)}
                 cls_logits: torch.Tensor,      # (A, C)
                 obj_logits: torch.Tensor,      # (A, 1)
                 ):
        gt_boxes = tgt['boxes']
        gt_labels = tgt['labels']
        M, A = gt_boxes.size(0), anchor_points.size(0)
        device = anchor_points.device
        
        if M == 0:
            return (torch.zeros(A, dtype=torch.bool, device=device),
                    torch.full((A,), -1, dtype=torch.long, device=device),
                    torch.zeros((A, 4), dtype=torch.float32, device=device),
                    torch.zeros((A,), dtype=torch.float32, device=device))

        # --- Strict Geometric Pre-filtering ---
        # 1. Anchor centers must be inside a GT box.
        ap_x, ap_y = anchor_points[:, 0], anchor_points[:, 1]
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
        
        # is_in_gt_matrix[i, j] is True if anchor i is inside GT j
        is_in_gt_matrix = (
            (ap_x[:, None] >= gt_x1[None, :]) & (ap_x[:, None] < gt_x2[None, :]) &
            (ap_y[:, None] >= gt_y1[None, :]) & (ap_y[:, None] < gt_y2[None, :])
        ) # Shape (A, M)

        # 2. Anchor center must be close to GT center (center sampling radius).
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
        dist = (anchor_points.unsqueeze(1) - gt_centers.unsqueeze(0)).abs().max(dim=-1).values
        is_in_radius_matrix = dist < (self.r * anchor_strides)

        # A candidate anchor must satisfy both conditions.
        candidate_mask = is_in_gt_matrix & is_in_radius_matrix # (A, M)
        
        # --- Dynamic-k Selection ---
        # Calculate IoU only for candidate anchors to save computation
        candidate_ious = torch.zeros_like(candidate_mask, dtype=torch.float32) # (A, M)
        # Flatten for efficient calculation
        gt_idx_flat, anchor_idx_flat = torch.where(candidate_mask.T) # (M, A) -> (A, M)
        
        if len(anchor_idx_flat) > 0:
            candidate_boxes_xyxy = torch.cat([
                anchor_points[anchor_idx_flat] - 0.5 * anchor_strides[anchor_idx_flat],
                anchor_points[anchor_idx_flat] + 0.5 * anchor_strides[anchor_idx_flat]
            ], dim=1)
            ious = tvops.box_iou(gt_boxes[gt_idx_flat], candidate_boxes_xyxy).diag()
            candidate_ious[anchor_idx_flat, gt_idx_flat] = ious

        # Dynamic k is the number of top IoU candidates for each GT
        topk_ious_per_gt, _ = torch.topk(candidate_ious.T, self.k, dim=1) # (M, k)
        dynamic_ks = torch.clamp(topk_ious_per_gt.sum(1).int(), min=1)

        # --- Cost Calculation ---
        # Cost is computed only for candidate (anchor, GT) pairs
        eps = 1e-9
        
        # Classification cost: -log(P(pred_class=gt_class))
        cls_prob = cls_logits.sigmoid()
        obj_prob = obj_logits.sigmoid()
        joint_prob = cls_prob[:, gt_labels] * obj_prob # (A, M)
        cls_cost = -torch.log(joint_prob + eps)
        
        # IoU cost
        iou_cost = -torch.log(candidate_ious + eps)

        # Final cost
        cost_matrix = self.cls_w * cls_cost + self.iou_w * iou_cost + (~candidate_mask) * 1e4

        # --- Assignment ---
        # For each GT, find the `dynamic_k` anchors with the lowest cost
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        for g_idx in range(M):
            k = dynamic_ks[g_idx]
            # Consider only candidates for this GT
            _, topk_anchor_indices = torch.topk(cost_matrix[:, g_idx], k, largest=False)
            matching_matrix[topk_anchor_indices, g_idx] = 1
        
        # If an anchor is matched to multiple GTs, assign it to the one with the lowest cost
        anchor_cost_per_gt = matching_matrix * cost_matrix
        anchor_cost_per_gt[anchor_cost_per_gt == 0] = 1e4 # Penalize non-matches
        
        assign_gt_indices = anchor_cost_per_gt.argmin(dim=1)
        
        # Final foreground mask: anchor must be matched to at least one GT
        fg_mask = matching_matrix.sum(dim=1) > 0
        
        # --- Build Outputs ---
        assigned_labels = torch.full((A,), -1, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)
        assigned_ious = torch.zeros((A,), dtype=torch.float32, device=device)

        if fg_mask.any():
            assigned_gt_idx_fg = assign_gt_indices[fg_mask]
            assigned_labels[fg_mask] = gt_labels[assigned_gt_idx_fg]
            assigned_boxes[fg_mask] = gt_boxes[assigned_gt_idx_fg]
            assigned_ious[fg_mask] = candidate_ious[fg_mask, assigned_gt_idx_fg]

        if self._dbg_mod and (self._dbg_iter % (1000 * self._dbg_mod) == 0):
            num_fg = fg_mask.sum().item()
            num_cand = candidate_mask.sum().item()
            mean_iou = assigned_ious[fg_mask].mean().item() if num_fg else 0
            k_bar = dynamic_ks.float().mean().item()
            print(f"[SimOTA] fg={num_fg:4d} cand={num_cand:4d} avgIoU={mean_iou:4.3f} k̄={k_bar:3.1f}")
        self._dbg_iter += 1
        
        return fg_mask, assigned_labels, assigned_boxes, assigned_ious


# ───────────────────── train / val loops ────────────────────────
def train_epoch(
        model: PicoDet, loader, opt, scaler, assigner: SimOTAAssigner,
        device: torch.device, epoch: int, max_epochs: int,
        w_vfl: float = 2.0, w_dfl: float = 1.0, w_iou: float = 5.0,
        quality_floor_vfl: float = 0.05, debug_prints: bool = True,
):
    model.train()
    _, tot_loss_accum = time.time(), 0.
    total_samples_contributing_to_loss_epoch = 0

    # Get anchors once, as image size is fixed
    anchor_points, anchor_strides_flat = model.anchor_manager()
    
    for i, (imgs, tgts_batch) in enumerate(loader):
        imgs = imgs.to(device)
        
        # Model in train mode returns (cls_logits, obj_logits, reg_logits, strides_tuple)
        cls_preds_levels, obj_preds_levels, reg_preds_levels, _ = model(imgs)

        # Flatten all predictions for the batch for efficient processing
        bs, nc = imgs.size(0), model.head.nc
        cls_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, nc) for lvl in cls_preds_levels], 1)
        obj_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 1) for lvl in obj_preds_levels], 1)
        reg_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 4 * (model.head.reg_max + 1)) for lvl in reg_preds_levels], 1)
        
        batch_total_loss = torch.tensor(0.0, device=device)
        num_samples_with_loss_in_batch = 0
        loss_vfl_print, loss_dfl_print, loss_iou_print = [torch.tensor(0.0) for _ in range(3)]

        for b_idx in range(bs):
            tgt_i = tgts_batch[b_idx]
            gt_boxes = tgt_i["boxes"].to(device, non_blocking=True)
            gt_labels = tgt_i["labels"].to(device, non_blocking=True)
            
            # Filter tiny GT boxes
            if gt_boxes.numel() > 0:
                valid_gt_mask = (gt_boxes[:, 2] - gt_boxes[:, 0] > 1e-3) & (gt_boxes[:, 3] - gt_boxes[:, 1] > 1e-3)
                gt_boxes = gt_boxes[valid_gt_mask]
                gt_labels = gt_labels[valid_gt_mask]

            target_dict = {'boxes': gt_boxes, 'labels': gt_labels}
            
            fg_mask, gt_lbls, gt_bxs, gt_ious = assigner(
                anchor_points, anchor_strides_flat, target_dict,
                cls_p_flat[b_idx], obj_p_flat[b_idx]
            )

            num_fg = fg_mask.sum()
            if num_fg == 0: continue
            
            # --- VFL (Classification) Loss ---
            vfl = VarifocalLoss(alpha=0.75, gamma=2.0, reduction='sum')
            joint_logits = logit_of_product(cls_p_flat[b_idx], obj_p_flat[b_idx])
            quality = torch.zeros_like(gt_ious, device=device)
            quality[fg_mask] = torch.maximum(gt_ious[fg_mask], torch.tensor(quality_floor_vfl, device=device))
            vfl_targets = torch.zeros_like(joint_logits)
            vfl_targets[fg_mask, gt_lbls[fg_mask]] = quality[fg_mask]
            loss_vfl = vfl(joint_logits, vfl_targets) / num_fg

            # --- DFL + IoU (Regression) Loss ---
            reg_p_fg = reg_p_flat[b_idx][fg_mask]
            box_targets_fg = gt_bxs[fg_mask]
            anchor_points_fg = anchor_points[fg_mask]
            anchor_strides_fg = anchor_strides_flat[fg_mask]
            
            ltrb_offsets = torch.cat([
                anchor_points_fg - box_targets_fg[:, :2],
                box_targets_fg[:, 2:] - anchor_points_fg
            ], 1) / anchor_strides_fg
            
            dfl_target_dist = build_dfl_targets(ltrb_offsets, model.head.reg_max)
            loss_dfl = dfl_loss(reg_p_fg, dfl_target_dist)

            pred_boxes_fg = model.head.decode_predictions(reg_p_fg.unsqueeze(0), anchor_points_fg, anchor_strides_fg).squeeze(0)
            loss_iou = tvops.complete_box_iou_loss(pred_boxes_fg, box_targets_fg, reduction='sum') / num_fg

            # --- Total Loss for Sample ---
            current_sample_total_loss = w_vfl * loss_vfl + w_dfl * loss_dfl + w_iou * loss_iou
            batch_total_loss += current_sample_total_loss
            num_samples_with_loss_in_batch += 1

            if num_samples_with_loss_in_batch == 1 and i == 0:
                loss_vfl_print, loss_dfl_print, loss_iou_print = loss_vfl.detach(), loss_dfl.detach(), loss_iou.detach()
        
        if num_samples_with_loss_in_batch > 0:
            averaged_batch_loss = batch_total_loss / num_samples_with_loss_in_batch
            opt.zero_grad(set_to_none=True)
            scaler.scale(averaged_batch_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(opt); scaler.update()
            tot_loss_accum += averaged_batch_loss.item() * num_samples_with_loss_in_batch
            total_samples_contributing_to_loss_epoch += num_samples_with_loss_in_batch

        if debug_prints and i == 0 and total_samples_contributing_to_loss_epoch > 0:
            print(f"E{epoch} B0 loss: "
                  f"vfl {loss_vfl_print.item()*w_vfl:.3f} | "
                  f"dfl {loss_dfl_print.item()*w_dfl:.3f} | "
                  f"iou {loss_iou_print.item()*w_iou:.3f} || "
                  f"Total {averaged_batch_loss.item():.3f}")

    return tot_loss_accum / total_samples_contributing_to_loss_epoch if total_samples_contributing_to_loss_epoch > 0 else 0.0

# ───────────────────── QAT helpers  ───────────────────────────────
def qat_prepare(model: nn.Module, example_input: torch.Tensor) -> torch.fx.GraphModule:
    global_qconfig = QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                             weight=MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    qconfig_mapping = get_default_qat_qconfig_mapping("x86").set_global(global_qconfig).set_module_name('pre', None)
    model.cpu().train()
    return prepare_qat_fx(model, qconfig_mapping, example_input.cpu())

# ─────────────────── Validation & NMS ──────────────────────
def apply_nms(
    raw_boxes_batch: torch.Tensor, # (B, Total_Anchors, 4)
    raw_scores_batch: torch.Tensor, # (B, Total_Anchors, NC) -> This is now (B, A, 1) after max
    score_thresh: float,
    iou_thresh: float,
    max_detections: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    
    batch_size, _, num_classes = raw_scores_batch.shape
    device = raw_boxes_batch.device

    final_boxes_list, final_scores_list, final_labels_list = [], [], []
    num_before_nms_list, num_after_nms_list = [], []

    for b_idx in range(batch_size):
        boxes_img = raw_boxes_batch[b_idx]     # (A, 4)
        scores_img = raw_scores_batch[b_idx]   # (A, NC)

        # Find max score and corresponding class for each anchor
        scores_per_anchor, labels_per_anchor = torch.max(scores_img, dim=1) # (A,)
        
        keep_mask = scores_per_anchor >= score_thresh
        boxes_pre_nms = boxes_img[keep_mask]
        scores_pre_nms = scores_per_anchor[keep_mask]
        labels_pre_nms = labels_per_anchor[keep_mask]
        
        num_before_nms_list.append(boxes_pre_nms.shape[0])
        
        if boxes_pre_nms.numel() == 0:
            nms_keep_indices = torch.empty(0, dtype=torch.long, device=device)
        else:
            nms_keep_indices = tvops.batched_nms(
                boxes_pre_nms, scores_pre_nms, labels_pre_nms, iou_thresh
            )
        
        # Keep top `max_detections`
        top_k_indices = nms_keep_indices[:max_detections]
        boxes_post_nms = boxes_pre_nms[top_k_indices]
        scores_post_nms = scores_pre_nms[top_k_indices]
        labels_post_nms = labels_pre_nms[top_k_indices]
        
        num_after_nms_list.append(boxes_post_nms.shape[0])

        # Pad results to max_detections
        pad_size = max_detections - boxes_post_nms.shape[0]
        padded_boxes = F.pad(boxes_post_nms, (0, 0, 0, pad_size), value=0.0)
        padded_scores = F.pad(scores_post_nms, (0, pad_size), value=0.0)
        padded_labels = F.pad(labels_post_nms, (0, pad_size), value=-1)

        final_boxes_list.append(padded_boxes)
        final_scores_list.append(padded_scores)
        final_labels_list.append(padded_labels)

    return (torch.stack(final_boxes_list), torch.stack(final_scores_list),
            torch.stack(final_labels_list), num_before_nms_list, num_after_nms_list)


@torch.no_grad()
def quick_val_iou(model: PicoDet, loader, device, epoch_num: int = -1, debug_prints: bool = True):
    model.eval()
    total_iou_sum, num_gt_total = 0., 0
    total_preds_before_nms, total_preds_after_nms = 0, 0
    num_images_processed = 0

    if debug_prints:
        print(f"\n--- quick_val_iou Start (Epoch: {epoch_num}) ---")
        print(f"Params: score_thresh={model.score_th}, iou_thresh={model.iou_th}, max_detections={model.max_det}")

    for imgs_batch, tgts_batch in loader:
        num_images_processed += imgs_batch.size(0)
        
        # 1. Get raw model outputs
        raw_pred_boxes, raw_pred_scores = model(imgs_batch.to(device))
        
        # 2. Apply NMS to get final, filtered predictions for the entire batch
        (final_boxes_batch,  # (B, max_det, 4)
         final_scores_batch, # (B, max_det)
         _, # final_labels_batch is not needed for this metric
         num_before_nms_list,
         num_after_nms_list
        ) = apply_nms(
            raw_pred_boxes, raw_pred_scores, model.score_th, model.iou_th, model.max_det
        )

        total_preds_before_nms += sum(num_before_nms_list)
        total_preds_after_nms += sum(num_after_nms_list)

        # 3. Iterate through images to compare final predictions with GT
        for i in range(imgs_batch.size(0)):
            gt_boxes = tgts_batch[i]["boxes"].to(device)
            if gt_boxes.numel() == 0:
                continue

            num_gt_total += gt_boxes.shape[0]
            
            # Get the actual number of detections for this image (before padding)
            num_dets_this_img = num_after_nms_list[i]
            if num_dets_this_img == 0:
                continue # No detections for this image, contributes 0 to IoU sum

            # Slice to get the real, unpadded predictions
            predicted_boxes_this_img = final_boxes_batch[i, :num_dets_this_img, :]
            
            # Calculate IoU matrix between final predictions and GT
            iou_matrix = tvops.box_iou(predicted_boxes_this_img, gt_boxes)
            
            if iou_matrix.numel() > 0:
                # For each GT box, find the prediction with the highest IoU
                max_iou_per_gt, _ = iou_matrix.max(dim=0)
                total_iou_sum += max_iou_per_gt.sum().item()

    avg_preds_before = total_preds_before_nms / num_images_processed if num_images_processed > 0 else 0
    avg_preds_after = total_preds_after_nms / num_images_processed if num_images_processed > 0 else 0
    final_mean_iou = total_iou_sum / num_gt_total if num_gt_total > 0 else 0
    
    if debug_prints:
        print("--- quick_val_iou End ---")
        print(f"Images processed: {num_images_processed}, Total GT boxes: {num_gt_total}")
        print(f"Avg preds/img (passed score_thresh): {avg_preds_before:.2f}")
        print(f"Avg preds/img (after NMS): {avg_preds_after:.2f}")
        print(f"Final Mean IoU (sum_max_iou_per_gt / total_gt_boxes): {final_mean_iou:.4f}")
    
    return final_mean_iou

# ────────────────── ONNX Export (Refactored) ────────────────────
class PostprocessorForONNX(nn.Module):
    """Performs decoding inside the ONNX graph using pre-computed anchors."""
    def __init__(self, core_model_ref: PicoDet):
        super().__init__()
        # Clone buffers from the model's AnchorManager to embed them in the ONNX graph
        self.register_buffer('anchor_points', core_model_ref.anchor_manager.anchor_points_cat.clone())
        self.register_buffer('strides', core_model_ref.anchor_manager.strides_cat.clone())
        # The head is now a sub-module that will be part of the graph
        self.head = core_model_ref.head

    def forward(self, neck_features: Tuple[torch.Tensor, ...]):
        # The core model in training mode (which is what gets traced) will output neck features.
        # This postprocessor will take them and perform the head computation + decoding.
        cls_logits, obj_logits, reg_logits = self.head(neck_features)
        
        # Flatten and decode
        cls_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, self.head.nc) for lvl in cls_logits], 1)
        obj_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 1) for lvl in obj_logits], 1)
        reg_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 4 * (self.head.reg_max + 1)) for lvl in reg_logits], 1)

        boxes = self.head.decode_predictions(reg_flat, self.anchor_points, self.strides)
        scores = cls_flat.sigmoid() * obj_flat.sigmoid()
        
        return boxes, scores

class ONNXExportablePicoDet(nn.Module):
    """Wrapper for ONNX export that combines Quantized core model + Postprocessor."""
    def __init__(self, core_model: nn.Module, postprocessor: PostprocessorForONNX):
        super().__init__()
        self.core_model = core_model
        self.postprocessor = postprocessor

    def forward(self, x: torch.Tensor):
        # The core_model is the quantized PicoDet up to the neck outputs
        # It's traced from a model that contains pre, backbone, and neck
        neck_p3, neck_p4, neck_p5 = self.core_model(x)
        
        # The postprocessor takes these neck features and does the rest
        return self.postprocessor((neck_p3, neck_p4, neck_p5))

# --- append_nms_to_onnx ---
def append_nms_to_onnx(in_path: str, out_path: str, score_thresh: float, iou_thresh: float, max_det: int, **_):
    m = onnx.load(in_path)
    g = m.graph
    g.initializer.extend([ oh.make_tensor("iou_th", TP.FLOAT, [], [iou_thresh]), oh.make_tensor("score_th", TP.FLOAT, [], [score_thresh]), oh.make_tensor("max_out", TP.INT64, [], [max_det])])
    # The ONNX NMS node expects scores in shape [batch, num_classes, num_boxes]
    scores_bca = "scores_bca"
    g.node.append(oh.make_node("Transpose", ["raw_scores"], [scores_bca], perm=[0, 2, 1]))
    selected_indices = "selected_indices"
    g.node.append(oh.make_node("NonMaxSuppression", ["raw_boxes", scores_bca, "max_out", "iou_th", "score_th"], [selected_indices]))
    # The output of NMS is a tensor of shape [num_selected_indices, 3]
    # where each row is [batch_index, class_index, box_index]
    # We need to gather the final boxes, scores, and labels based on these indices.
    final_boxes = "final_boxes"
    final_scores = "final_scores"
    final_labels = "final_labels"
    g.node.append(oh.make_node("GatherND", ["raw_boxes", selected_indices], [final_boxes], batch_dims=1))
    g.node.append(oh.make_node("GatherND", [scores_bca, selected_indices], [final_scores], batch_dims=1))
    # Extract class labels from the indices
    g.initializer.append(oh.make_tensor("labels_col_idx", TP.INT64, [1], [1]))
    labels_indices = "labels_indices"
    g.node.append(oh.make_node("Gather", [selected_indices, "labels_col_idx"], [labels_indices], axis=1))
    g.node.append(oh.make_node("Squeeze", [labels_indices], [final_labels], axes=[1]))
    del g.output[:]
    g.output.extend([
        oh.make_tensor_value_info(final_boxes, TP.FLOAT, ['num_dets', 4]),
        oh.make_tensor_value_info(final_scores, TP.FLOAT, ['num_dets']),
        oh.make_tensor_value_info(final_labels, TP.INT64, ['num_dets']),
    ])
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"[SAVE] Final ONNX with NMS → {out_path}")

class QATFineTuneModel(nn.Module):
    def __init__(self, core_qat, head_fp32, anchor_manager, strides):
        super().__init__()
        self.core_model = core_qat
        self.head = head_fp32
        self.anchor_manager = anchor_manager
        self.strides = strides
    
    def forward(self, x):
        # 1) run the quantized neck
        p3, p4, p5 = self.core_model(x)
        
        if self.training:
            # training: return raw logits  strides for train_epoch
            cls_logits, obj_logits, reg_logits = self.head((p3, p4, p5))
            return cls_logits, obj_logits, reg_logits, self.strides
        else:
            # inference: decode boxes & scores
            anchor_points, strides_flat = self.anchor_manager()
            boxes, scores = self.head(
                (p3, p4, p5),
                anchor_points=anchor_points,
                strides=strides_flat
            )
            return boxes, scores


def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', default='mnv4c', choices=['mnv4c'])
    pa.add_argument('--epochs', type=int, default=12) 
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    cfg = pa.parse_args(argv)

    TRAIN_SUBSET, VAL_SUBSET = 50000, 5000
    debug_prints = True
    BACKBONE_FREEZE_EPOCHS = 4

    dev = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[INFO] device = {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, ckpt=None, img_size=IMG_SIZE)
    model = PicoDet(backbone, feat_chs, num_classes=80, neck_out_ch=96, img_size=IMG_SIZE).to(dev)

    root = cfg.coco_root
    coco_label_map = CANONICAL_COCO80_MAP
    train_ds = CocoDetectionV2(f"{root}/train2017", f"{root}/annotations/instances_train2017.json", coco_label_map, build_transforms((IMG_SIZE, IMG_SIZE), True))
    val_ds = CocoDetectionV2(f"{root}/val2017", f"{root}/annotations/instances_val2017.json", coco_label_map, build_transforms((IMG_SIZE, IMG_SIZE), False))
    
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=False, num_samples=min(TRAIN_SUBSET, len(train_ds))) if TRAIN_SUBSET else None
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch, sampler=train_sampler, num_workers=cfg.workers, collate_fn=collate_v2, pin_memory=True, persistent_workers=bool(cfg.workers))

    if VAL_SUBSET:
        val_idx = torch.randperm(len(val_ds))[:VAL_SUBSET].tolist()
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    else:
        val_sampler = None
    vl_loader = DataLoader(
        val_ds,   batch_size=cfg.batch*2, sampler=val_sampler,
        num_workers=cfg.workers, collate_fn=collate_v2,
        pin_memory=True
    )

    base_lr, warmup_epochs = 0.005, 5
    opt = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    warmup_sch = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sch = CosineAnnealingLR(opt, T_max=cfg.epochs - warmup_epochs, eta_min=base_lr * 0.01)
    sch = SequentialLR(opt, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')

    assigner = SimOTAAssigner(nc=model.head.nc, debug_epochs=5 if debug_prints else 0)

    # --- FP32 training loop ---
    for ep in range(cfg.epochs):
        if ep < BACKBONE_FREEZE_EPOCHS:
            if ep == 0:
                for p in model.backbone.parameters(): p.requires_grad = False
                print(f"[INFO] Backbone frozen for {BACKBONE_FREEZE_EPOCHS} epochs…")
        elif ep == BACKBONE_FREEZE_EPOCHS:
            for p in model.backbone.parameters(): p.requires_grad = True
            print("[INFO] Backbone unfrozen – full network now training")

        l = train_epoch(model, tr_loader, opt, scaler, assigner, dev, ep, cfg.epochs, debug_prints=debug_prints)
        m = quick_val_iou(model, vl_loader, dev, epoch_num=ep, debug_prints=True)
        print(f'Epoch {ep + 1}/{cfg.epochs}  loss {l:.3f}  IoU {m:.3f} lr={opt.param_groups[0]["lr"]:.6f}\n')
        sch.step()
        assigner._dbg_iter = 0

    print("[INFO] Preparing model for QAT...")
    model.cpu()
    model.train()
    
    class CoreModel(nn.Module):
        def __init__(self, pre, backbone, neck):
            super().__init__()
            self.pre = pre
            self.backbone = backbone
            self.neck = neck
        def forward(self, x):
            x = self.pre(x)
            c3, c4, c5 = self.backbone(x)
            return self.neck(c3, c4, c5)

    # 2. Create an instance of this core structure from the trained FP32 model
    core_model_to_quantize = CoreModel(model.pre, model.backbone, model.neck)
    
    # 3. Prepare the core model for QAT
    example_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    quantized_core_model = qat_prepare(core_model_to_quantize, example_input)

    # Create the model for the QAT training loop
    qat_finetune_model = QATFineTuneModel(quantized_core_model, model.head, model.anchor_manager, model.strides).to(dev)

    # Now, use `qat_finetune_model` for the QAT loop
    qat_epochs = max(3, int(cfg.epochs * 0.2))
    # Note: Only pass the parameters of the new model to the optimizer
    opt_q = SGD(filter(lambda p: p.requires_grad, qat_finetune_model.parameters()), lr=base_lr/20, momentum=0.9, weight_decay=1e-5)
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))

    print(f"[INFO] Starting QAT finetuning for {qat_epochs} epochs...")
    for qep in range(qat_epochs):
        # Pass the new QAT model to the training loop
        lq = train_epoch(qat_finetune_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, qat_epochs, debug_prints=False)

        # For validation, we need to adapt the model to the quick_val_iou signature
        try:
            qat_finetune_model.eval()
            mq = quick_val_iou(qat_finetune_model, vl_loader, dev, epoch_num=qep)
        except Exception as e:
            print(f"quick val error {repr(e)}")
        qat_finetune_model.train() # Set back to train mode

        print(f'[QAT] Epoch {qep + 1}/{qat_epochs} Train Loss {lq:.3f}  Val IoU {mq:.3f}')

    print("[INFO] QAT finetuning completed. Converting and exporting to ONNX...")
    qat_finetune_model.cpu().eval()
    
    # Convert the final QAT-tuned core model to INT8
    # The core model is an attribute of our fine-tuning model
    final_quantized_core_model = convert_fx(qat_finetune_model.core_model)
    
    # For export, we use the original FP32 model as a reference for the postprocessor
    # as it holds the correctly configured head and anchor manager.
    onnx_postprocessor = PostprocessorForONNX(model)

    # Wrap the *final converted INT8 core* and the postprocessor for export
    final_exportable_model = ONNXExportablePicoDet(final_quantized_core_model, onnx_postprocessor)
    final_exportable_model.eval()

    temp_onnx_path = cfg.out.replace(".onnx", "_temp.onnx")
    torch.onnx.export(
        final_exportable_model,
        example_input,
        temp_onnx_path,
        input_names=['images_uint8'],
        output_names=['raw_boxes', 'raw_scores'],
        dynamic_axes={'images_uint8': {0: 'batch'}, 'raw_boxes': {0: 'batch'}, 'raw_scores': {0: 'batch'}},
        opset_version=18,
    )
    print(f'[SAVE] Intermediate ONNX (no NMS) → {temp_onnx_path}')
    
    append_nms_to_onnx(
        in_path=temp_onnx_path,
        out_path=cfg.out,
        score_thresh=model.score_th,
        iou_thresh=model.iou_th,
        max_det=model.max_det,
    )

if __name__ == '__main__':
    main()