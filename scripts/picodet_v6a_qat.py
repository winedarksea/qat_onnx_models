from __future__ import annotations
import argparse, random, time, warnings, math, copy, traceback
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.transforms import v2 as T
from torchvision.datasets import CocoDetection
from torchvision.tv_tensors import BoundingBoxes
import torchvision.ops as tvops
from pycocotools.coco import COCO

from torch.ao.quantization import (
    get_default_qat_qconfig_mapping, QConfig,
    MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

import onnx
from onnx import TensorProto as TP, helper as oh

if False:
    from picodet_lib import (
        PicoDet, get_backbone,
        VarifocalLoss, dfl_loss, build_dfl_targets,
        generate_anchors, ATSSAssigner, PicoDetHead,
    )

# ───────────────────── data & transforms ───────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 256
# COCO 80-class mapping
CANONICAL_COCO80_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,
                        18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,
                        37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,
                        54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,
                        74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
CANONICAL_COCO80_MAP = {cid:i for i,cid in enumerate(CANONICAL_COCO80_IDS)}

def contiguous_id_to_name(coco_api: COCO) -> dict[int,str]:
    return {i: coco_api.loadCats([coco_id])[0]["name"]
            for i,coco_id in enumerate(CANONICAL_COCO80_IDS)}

class CocoDetectionV2(CocoDetection):
    def __init__(self, img_dir, ann_file, lb_map, transforms=None):
        super().__init__(img_dir, ann_file)
        self.lb_map = lb_map
        self._tf    = transforms
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        boxes, labels = [], []
        W, H = img.size
        for a in anns:
            if a.get("iscrowd",0): continue
            cid = a["category_id"]
            if cid not in self.lb_map: continue
            x,y,w,h = a["bbox"]
            boxes.append([x,y,x+w,y+h])
            labels.append(self.lb_map[cid])
        if not boxes:
            bbx = BoundingBoxes(torch.zeros((0,4)), format="XYXY", canvas_size=(H,W))
            tgt = {"boxes": bbx, "labels": torch.zeros((0,),dtype=torch.int64)}
        else:
            bbx = BoundingBoxes(torch.as_tensor(boxes), format="XYXY", canvas_size=(H,W))
            tgt = {"boxes": bbx, "labels": torch.as_tensor(labels)}
        if self._tf:
            return self._tf(img, tgt)
        return img, tgt

class ComposeTransforms:
    def __init__(self, size, train: bool):
        ops = [T.ToImage()]
        if train:
            ops += [T.RandomHorizontalFlip(0.25),
                    T.RandomResizedCrop(size, scale=(0.8,1.0), antialias=True),
                    T.ColorJitter(0.2,0.2,0.2,0.1)]
        else:
            ops += [T.Resize(size, antialias=True)]
        ops.append(T.ToDtype(torch.uint8, scale=True))
        self._c = T.Compose(ops)
    def __call__(self, img, tgt): return self._c(img, tgt)

def collate_v2(batch):
    imgs, tgts = zip(*batch)
    return torch.stack(imgs,0), list(tgts)

# ───────────────────── train / val loops ───────────────────────
def train_epoch(
    model: nn.Module,
    loader,
    opt,
    scaler,
    assigner: ATSSAssigner,
    device: torch.device,
    epoch: int,
    head_nc: int,
    head_reg_max: int,
    dfl_buf: torch.Tensor,
    max_epochs: int = 500,
    quality_floor_vfl: float = 0.01,
    w_cls: float = 4.0,
    w_iou_initial: float = 5.0,
    w_iou_final: float = 2.0,
    debug: bool = True
) -> float:
    model.train()
    iou_chg_ep = int(max_epochs * 0.4)
    w_iou = w_iou_initial if epoch < iou_chg_ep else w_iou_final
    total_loss, total_count = 0.0, 0

    for imgs, tgts in loader:
        imgs = imgs.to(device)
        cls_preds, obj_preds, reg_preds, strides = model(imgs)
        # Build feature-map shapes
        fmap_shapes = [(p.shape[2], p.shape[3], float(s))
                       for p, s in zip(cls_preds, strides)]
        anchors_centers, anchor_strides = generate_anchors(
            fmap_shapes, model.head.strides_buffer, device)

        batch_loss, batch_count = 0.0, 0
        for b_idx in range(imgs.size(0)):
            tgt = tgts[b_idx]
            gt_boxes = tgt["boxes"].to(device).to(torch.float32)
            gt_labels = tgt["labels"].to(device)

            # Flatten predictions
            cls_p = torch.cat([p[b_idx].permute(1,2,0).reshape(-1, head_nc)
                                for p in cls_preds],0)
            obj_p = torch.cat([p[b_idx].permute(1,2,0).reshape(-1)
                                for p in obj_preds],0)
            reg_p = torch.cat([p[b_idx].permute(1,2,0).reshape(-1,4*(head_reg_max+1))
                                for p in reg_preds],0)

            # Assignment
            fg_mask, gt_lbl, gt_box, gt_iou = assigner.assign(
                anchors_centers, anchor_strides, gt_boxes, gt_labels)
            if not fg_mask.any(): continue

            # Build regression targets
            ctrs = anchors_centers[fg_mask]
            sfg = anchor_strides[fg_mask].unsqueeze(-1)
            ltrb = torch.stack([
                ctrs[:,0]-gt_box[fg_mask][:,0],
                ctrs[:,1]-gt_box[fg_mask][:,1],
                gt_box[fg_mask][:,2]-ctrs[:,0],
                gt_box[fg_mask][:,3]-ctrs[:,1],
            ],1) / sfg
            ltrb = ltrb.clamp(0, head_reg_max)
            dfl_tgt = build_dfl_targets(ltrb, head_reg_max)
            loss_dfl = dfl_loss(reg_p[fg_mask], dfl_tgt)

            # IoU loss
            pred_offset = PicoDetHead.dfl_decode_for_training(
                reg_p[fg_mask], dfl_buf.to(device), head_reg_max)
            pred_offset = pred_offset * sfg
            pred_boxes = torch.stack([
                ctrs[:,0]-pred_offset[:,0],
                ctrs[:,1]-pred_offset[:,1],
                ctrs[:,0]+pred_offset[:,2],
                ctrs[:,1]+pred_offset[:,3],
            ],1)
            loss_iou = tvops.complete_box_iou_loss(
                pred_boxes, gt_box[fg_mask], reduction='sum') / fg_mask.sum()

            # Classification (VFL)
            joint = cls_p + obj_p.unsqueeze(1)
            # Quality targets
            qt = torch.zeros_like(gt_iou)
            qt[fg_mask] = torch.maximum(gt_iou[fg_mask], torch.tensor(quality_floor_vfl, device=device))
            # Build VFL targets only for positives
            vfl_tgt = torch.zeros_like(joint)
            pos_idx = fg_mask.nonzero(as_tuple=True)[0]
            vfl_tgt[pos_idx, gt_lbl[pos_idx]] = qt[pos_idx]
            alpha_dyn = 0.25 + (0.75-0.25)*0.5*(1-math.cos(math.pi*epoch/max_epochs))
            vfl = VarifocalLoss(alpha=alpha_dyn, gamma=2.0, reduction='mean')
            loss_cls = vfl(joint, vfl_tgt)

            # Total loss
            loss = w_cls * loss_cls + loss_dfl + w_iou * loss_iou
            batch_loss += loss
            batch_count += 1

            if debug and (total_count % 2000 == 0):
                print(f"[DEBUG] Epoch {epoch:2d}  Batch {b_idx}  "
                      f"Img {b_idx+1}/{imgs.size(0)}  "
                      f"FGs {fg_mask.sum().item():3d}  "
                      f"loss_cls={loss_cls.item():.4f}  "
                      f"loss_dfl={loss_dfl.item():.4f}  "
                      f"loss_iou={loss_iou.item():.4f}  "
                      f"total={loss.item():.4f}")

        if batch_count > 0:
            avg_loss = batch_loss / batch_count
            opt.zero_grad(set_to_none=True)
            scaler.scale(avg_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            scaler.step(opt)
            scaler.update()
            total_loss += avg_loss.item() * batch_count
            total_count += batch_count

    return total_loss / total_count if total_count else 0.0


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
    model: PicoDet, # Assuming PicoDet or a compatible model
    loader, device,
    score_thresh: float, # Passed in for this specific evaluation run
    iou_thresh: float,   # Passed in for this specific evaluation run (NMS IoU)
    max_detections: int, # Passed in for this specific evaluation run
    epoch_num: int = -1, # Optional: for logging
    run_name: str = "N/A", # Optional: for logging if you have multiple eval runs
    debug_prints: bool = True,
):
    model.eval()
    total_iou_sum = 0.
    num_images_with_gt = 0
    num_images_processed = 0
    total_gt_boxes_across_images = 0
    total_preds_before_nms_filter_across_images = 0 # Anchors above score_thresh (before NMS)
    total_preds_after_nms_filter_across_images = 0  # Detections after NMS & final filter

    if debug_prints:
        print(f"\n--- quick_val_iou Start (Epoch: {epoch_num}, Run: {run_name}) ---")
        print(f"Params: score_thresh={score_thresh}, iou_thresh={iou_thresh}, max_detections={max_detections}")

    for batch_idx, (imgs_batch, tgts_batch) in enumerate(loader):
        raw_pred_boxes_batch, raw_pred_scores_batch = model(imgs_batch.to(device))
        # raw_pred_boxes_batch: (B, Total_Anchors, 4)
        # raw_pred_scores_batch: (B, Total_Anchors, NC)

        # DEBUG: Print shapes of raw model outputs for the first batch
        if debug_prints and batch_idx == 0:
            print(f"[Debug Eval Batch 0] raw_pred_boxes_batch shape: {raw_pred_boxes_batch.shape}")
            print(f"[Debug Eval Batch 0] raw_pred_scores_batch shape: {raw_pred_scores_batch.shape}")

            print(f"[DEBUG] batch {batch_idx}")
            print("  images  :", imgs_batch.shape, imgs_batch.dtype, imgs_batch.min().item(), imgs_batch.max().item())
            bb = tgts_batch[0]["boxes"]
            print("  boxes   :", bb.shape, "format:", getattr(bb, 'format', 'tensor'), "sample:", bb[:4])

        # Apply NMS and padding to raw outputs
        # This function itself needs to use the passed score_thresh and iou_thresh
        # It returns padded outputs and also needs to give us info for debugging
        (pred_boxes_batch_padded,
         pred_scores_batch_padded,
         pred_labels_batch_padded,
         debug_num_preds_before_nms_batch, # List of counts per image in batch
         debug_num_preds_after_nms_batch   # List of counts per image in batch
         ) = apply_nms_and_padding_to_raw_outputs_with_debug(
                raw_pred_boxes_batch, raw_pred_scores_batch,
                score_thresh, iou_thresh, max_detections
            )
        
        total_preds_before_nms_filter_across_images += sum(debug_num_preds_before_nms_batch)

        for i in range(imgs_batch.size(0)): # Iterate through images in the batch
            num_images_processed += 1
            current_img_annots_raw = tgts_batch[i]
            boxes_xyxy = current_img_annots_raw["boxes"]      # BoundingBoxes
            if boxes_xyxy.numel():                            # image has GT
                gt_boxes_tensor = boxes_xyxy.to(device).to(torch.float32)
                gt_boxes_list = gt_boxes_tensor.tolist()      # for stats / prints
            else:
                gt_boxes_tensor = torch.empty((0, 4), device=device)
                gt_boxes_list   = []

            if not gt_boxes_list:
                if debug_prints and batch_idx == 0 and i < 2: # Log for first few images if no GT
                    print(f"[Debug Eval Img {num_images_processed-1}] GTs: {len(gt_boxes_list)}. "
          f"Score Thresh: {score_thresh}.")
                continue # Skip if no ground truth for this image

            num_images_with_gt += 1
            total_gt_boxes_across_images += len(gt_boxes_list)
            gt_boxes_tensor = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
            
            # These are already filtered by score_thresh (inside apply_nms...) and NMSed
            predicted_boxes_for_img_padded = pred_boxes_batch_padded[i] # Padded to max_detections

            # Filter out padded predictions (score == 0 or implicitly label == -1 means padding)
            # The number of non-padded preds is debug_num_preds_after_nms_batch[i]
            num_actual_dets_this_img = debug_num_preds_after_nms_batch[i]
            total_preds_after_nms_filter_across_images += num_actual_dets_this_img

            actual_predicted_boxes = predicted_boxes_for_img_padded[:num_actual_dets_this_img]

            if debug_prints and batch_idx == 0 and i < 2: # Log for first few images with GT
                print(f"[Debug Eval Img {num_images_processed-1}] GTs: {len(gt_boxes_list)}. Score Thresh: {score_thresh}.")
                print(f"  Num Preds BEFORE NMS (passed score_thresh): {debug_num_preds_before_nms_batch[i]}")
                print(f"  Num Preds AFTER NMS (final): {num_actual_dets_this_img}")

            if actual_predicted_boxes.numel() == 0: # No detections after NMS for this image
                # image_avg_iou will effectively be 0 for this image if we consider max_iou_per_gt
                # We sum max_iou_per_gt, so if no preds, it contributes 0 to sum for these GTs
                if batch_idx == 0 and i < 2:
                    print(f"  No actual predicted boxes after NMS for Img {num_images_processed-1} with GTs.")
                continue

            iou_matrix = tvops.box_iou(actual_predicted_boxes, gt_boxes_tensor) # (Num_Preds, Num_GTs)
            
            if iou_matrix.numel() == 0: # Should not happen if actual_predicted_boxes and gt_boxes_tensor are non-empty
                if debug_prints and batch_idx == 0 and i < 2:
                    print(f"  IoU matrix is empty for Img {num_images_processed-1} despite having preds and GTs.")
                continue
                
            # For each GT box, find the max IoU with any predicted box
            if iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
                max_iou_per_gt, _ = iou_matrix.max(dim=0) # Max IoU for each GT box
                image_avg_iou_for_matched_gts = max_iou_per_gt.sum().item() # Sum of best IoUs for GTs in this image
                total_iou_sum += image_avg_iou_for_matched_gts
                if debug_prints and batch_idx == 0 and i < 2:
                     print(f"  Img {num_images_processed-1}: Max IoUs per GT: {max_iou_per_gt.cpu().numpy().round(3)}. Sum for image: {image_avg_iou_for_matched_gts:.3f}")
            # If iou_matrix.shape[0] == 0 (no preds but GTs exist), this block is skipped,
            # and image_avg_iou_for_matched_gts is effectively 0 for this image's contribution.
            # This is handled by the actual_predicted_boxes.numel() == 0 check earlier.

    if debug_prints:
        print(f"--- quick_val_iou End (Epoch: {epoch_num}, Run: {run_name}) ---")
        print(f"Images processed: {num_images_processed}, Images with GT: {num_images_with_gt}")
        print(f"Total GT boxes: {total_gt_boxes_across_images}")
        avg_preds_before_nms = total_preds_before_nms_filter_across_images / num_images_processed if num_images_processed > 0 else 0
        avg_preds_after_nms = total_preds_after_nms_filter_across_images / num_images_processed if num_images_processed > 0 else 0
        print(f"Avg preds/img (passed score_thresh, before NMS): {avg_preds_before_nms:.2f}")
        print(f"Avg preds/img (after NMS & final filter): {avg_preds_after_nms:.2f}")
        
    # The mAP definition usually averages over classes and IoU thresholds.
    # This quick_val_iou averages the (sum of max_iou_per_gt) over all GT boxes.
    # It's a proxy for recall-oriented localization quality.
    if total_gt_boxes_across_images > 0:
        final_mean_iou = total_iou_sum / total_gt_boxes_across_images
    else:
        final_mean_iou = 0.0
    print(f"Final Mean IoU (sum_max_iou_per_gt / total_gt_boxes): {final_mean_iou:.4f}")
    return final_mean_iou


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

    # 5. Prepare the model for QAT using FX graph mode
    model.cpu().train() # Ensure model is on CPU and in train mode
    
    prepared_model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_input # Must also be on CPU
    )
    
    return prepared_model


# ────────────────── append_nms_to_onnx ────────────────────
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
            obj_logit: torch.Tensor,  # (B, 1,  H_feat, W_feat)
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
        obj_logit_perm = obj_logit.permute(0, 2, 3, 1).contiguous().view(B, num_anchors_level, 1)
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

        # For ONNX, sigmoid should be torch.sigmoid
        scores_level = torch.sigmoid(cls_logit_perm) * torch.sigmoid(obj_logit_perm)
        # scores_level = torch.sigmoid(cls_logit_perm + obj_logit_perm)

        return boxes_xyxy_level, scores_level

    def forward(self, raw_model_outputs_nested_tuple: Tuple[Tuple[torch.Tensor, ...], ...]):
        if not isinstance(raw_model_outputs_nested_tuple, tuple) or len(raw_model_outputs_nested_tuple) < 3:
            raise ValueError(
                f"PostprocessorForONNX expected a nested tuple with at least 3 inner tuples, "
                f"got type {type(raw_model_outputs_nested_tuple)} with length {len(raw_model_outputs_nested_tuple) if isinstance(raw_model_outputs_nested_tuple, tuple) else 'N/A'}"
            )

        raw_cls_logits_levels_tuple = raw_model_outputs_nested_tuple[0]
        raw_obj_logits_levels_tuple = raw_model_outputs_nested_tuple[1]
        raw_reg_logits_levels_tuple = raw_model_outputs_nested_tuple[2]

        if not (isinstance(raw_cls_logits_levels_tuple, tuple) and
                  isinstance(raw_obj_logits_levels_tuple, tuple) and
                  isinstance(raw_reg_logits_levels_tuple, tuple)):
            raise ValueError("Inner elements of the input to PostprocessorForONNX are not tuples as expected.")

        if not (len(raw_cls_logits_levels_tuple) == self.nl and
                len(raw_obj_logits_levels_tuple) == self.nl and
                len(raw_reg_logits_levels_tuple) == self.nl):
            raise ValueError(
                f"PostprocessorForONNX: Inner tuples do not have the expected length ({self.nl}). "
                f"Got lengths: cls={len(raw_cls_logits_levels_tuple)}, "
                f"obj={len(raw_obj_logits_levels_tuple)}, "
                f"reg={len(raw_reg_logits_levels_tuple)}"
            )

        decoded_boxes_all_levels_list: List[torch.Tensor] = []
        decoded_scores_all_levels_list: List[torch.Tensor] = []

        for i in range(self.nl):
            cls_l = raw_cls_logits_levels_tuple[i]
            obj_l = raw_obj_logits_levels_tuple[i]
            reg_l = raw_reg_logits_levels_tuple[i]

            boxes_level, scores_level = self._decode_predictions_for_level_onnx(
                cls_l, obj_l, reg_l, i
            )
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

    def forward(self, x: torch.Tensor): # x is images_uint8
        # core_model is expected to return the tuple of (typically 12) flattened tensors
        # from the baked-in training output path of PicoDetHead.
        raw_feature_outputs_tuple = self.core_model(x)

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
        k_value: int = 800,
):
    m = onnx.load(in_path)
    g = m.graph

    # ───────── constants ─────────
    g.initializer.extend([
        oh.make_tensor("nms_iou_th",   TP.FLOAT, [], [iou_thresh]),
        oh.make_tensor("nms_score_th", TP.FLOAT, [], [score_thresh]),
        oh.make_tensor("nms_max_det",  TP.INT64, [], [max_det]),
        oh.make_tensor("nms_axis0", TP.INT64, [1], [0]),
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
    b_idx, cls_idx, anc_idx = "batch_idx", "class_idx", "anchor_idx"
    for src, dst in [(b_col, b_idx), (c_col, cls_idx), (a_col, anc_idx)]:
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
        "Unsqueeze", [cls_idx, "nms_axis1"], [cls_unsq],
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
        oh.make_tensor_value_info(cls_idx,    TP.INT64, ['N']),
        oh.make_tensor_value_info(b_idx,      TP.INT64, ['N']),
    ])

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"[SAVE] Final ONNX with NMS → {out_path}")

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


# ───────────────────────── main ────────────────────────────────
def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', default='mnv4c', choices=['mnv3', 'mnv4c'])
    pa.add_argument('--epochs', type=int, default=10)
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    cfg = pa.parse_args(argv)

    dev = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[INFO] device = {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, None, IMG_SIZE)
    model = PicoDet(backbone, feat_chs, num_classes=80,
                    neck_out_ch=96, img_size=IMG_SIZE,
                    head_reg_max=8).to(dev)

    # Datasets
    train_ds = CocoDetectionV2(
        f"{cfg.coco_root}/train2017",
        f"{cfg.coco_root}/annotations/instances_train2017.json",
        CANONICAL_COCO80_MAP,
        transforms=ComposeTransforms((IMG_SIZE,IMG_SIZE), train=True)
    )
    val_ds = CocoDetectionV2(
        f"{cfg.coco_root}/val2017",
        f"{cfg.coco_root}/annotations/instances_val2017.json",
        CANONICAL_COCO80_MAP,
        transforms=ComposeTransforms((IMG_SIZE,IMG_SIZE), train=False)
    )
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch,
                            shuffle=True, num_workers=cfg.workers,
                            collate_fn=collate_v2)
    vl_loader = DataLoader(val_ds, batch_size=cfg.batch,
                            shuffle=False, num_workers=cfg.workers,
                            collate_fn=collate_v2)

    # Optimizer + scheduler
    debug_prints = True
    opt = SGD(model.parameters(), lr=0.006, momentum=0.9, weight_decay=1e-4)
    warmup = LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=3)
    cosine = CosineAnnealingLR(opt, T_max=cfg.epochs-3, eta_min=0)
    sch = SequentialLR(opt, schedulers=[warmup,cosine], milestones=[3])
    scaler = torch.amp.GradScaler(enabled=(dev.type=='cuda'))

    # Assigner
    assigner = ATSSAssigner(top_k=9)

    original_head_nc = model.head.nc
    original_head_reg_max = model.head.reg_max
    dfl_buffer = model.head.dfl_project_buffer
    max_detections = 100

    # FP32 training
    for ep in range(cfg.epochs):
        if ep==0:
            for p in model.backbone.parameters(): p.requires_grad=False
            print("[INFO] Backbone frozen for 2 epochs…")
        if ep==2:
            for p in model.backbone.parameters(): p.requires_grad=True
            print("[INFO] Backbone unfrozen – full training")

        model.train()
        l = train_epoch(
            model, tr_loader, opt, scaler, assigner, dev, ep,
            # CANONICAL_COCO80_MAP,
            original_head_nc,
            original_head_reg_max,
            dfl_buffer,
            max_epochs=cfg.epochs,
            debug=debug_prints
        )
        model.eval()
        m = quick_val_iou(model, vl_loader, dev,
                          score_thresh=model.head.score_th,
                          iou_thresh=model.head.iou_th,
                          max_detections=max_detections,
                          epoch_num=ep, run_name="val", debug_prints=debug_prints)
        print(f"Epoch {ep+1}/{cfg.epochs} loss={l:.3f} IoU={m:.3f} lr={opt.param_groups[0]['lr']:.6f}")
        sch.step()

    # QAT preparation + finetune + ONNX export…

    print("[INFO] Evaluating FP32 model...")
    model.eval()
    try:
        iou_05 = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.05,
                               iou_thresh=model.head.iou_th,
                               max_detections=max_detections,
                               epoch_num=ep,
                               run_name="score_thresh_0.05",
                               debug_prints=debug_prints,
                               )
        print(f"[INFO] Validation IoU (score_th=0.05): {iou_05:.4f}")
        
        # Run for score_thresh = 0.25
        iou_25 = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.25,
                               iou_thresh=model.head.iou_th,
                               max_detections=max_detections,
                               epoch_num=ep,
                               run_name="score_thresh_0.25",
                               debug_prints=debug_prints,
                               )
        print(f"[INFO] Validation IoU (score_th=0.25): {iou_25:.4f}")
    except Exception as e:
        print(repr(e))
    model.train() # Set back for QAT

    # --- QAT Preparation ---
    print("[INFO] Preparing model for QAT...")
    dummy_uint8_input_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8, device='cpu')

    # The 'model' contains 'model.pre' (ResizeNorm).
    # The 'example' for qat_prepare should be the input to 'model' itself.
    # ResizeNorm is configured to take uint8 input and convert/normalize.
    example_input_for_qat_entire_model = dummy_uint8_input_cpu.cpu()

    model.train()
    model.cpu()

    print("[INFO] Running qat_prepare...")
    # qat_prepare will trace the 'model', including 'model.pre'.
    # 'model.pre' will be skipped for quantization inserts due to set_module_name('pre', None)
    # but it will be part of the traced graph.
    qat_model = qat_prepare(model, example_input_for_qat_entire_model)
    qat_model = qat_model.to(dev)
    freeze_qat_stats = False
    if freeze_qat_stats:
        for m in qat_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()          # freeze running stats
                m.track_running_stats = False
    print("[INFO] QAT model prepared and moved to device.")

    # --- QAT Finetuning ---
    qat_model.train()

    qat_epochs = int(cfg.epochs * 0.2)
    qat_epochs = 3 if qat_epochs < 3 else qat_epochs

    qat_initial_lr = 0.00025
    
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
        
        lq = train_epoch(
            qat_model, tr_loader, opt_q_params, scaler_q, assigner, dev, qep,
            # CANONICAL_COCO80_MAP,
            original_head_nc, original_head_reg_max,
            dfl_buffer,
            max_epochs=qat_epochs,
            debug=debug_prints,
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
            # IMPORTANT: Use the evaluation wrapper for quick_val_iou
            # model.head is the original FP32 head, used to get parameters for PostprocessorForONNX
            eval_compatible_qat_model = ONNXExportablePicoDet(qat_model, PostprocessorForONNX(model.head))
            eval_compatible_qat_model.to(dev).eval() # Ensure it's on device and in eval mode

            # Using a consistent score_thresh for tracking QAT progress, e.g., 0.05 or 0.25
            # The one used for final ONNX NMS params can be different.
            current_qat_val_iou = quick_val_iou(
                                   eval_compatible_qat_model, vl_loader, dev,
                                   score_thresh=0.05, # Consistent validation score_thresh
                                   iou_thresh=model.head.iou_th,
                                   max_detections=max_detections,
                                   epoch_num=qep, # Pass QAT epoch number
                                   run_name=f"QAT_ep{qep+1}_score0.05",
                                   debug_prints=debug_prints,
                                )
            print(f"[QAT Eval] Epoch {qep + 1}/{qat_epochs} Val IoU (score_th=0.05): {current_qat_val_iou:.4f}")

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
        max_det=int(max_detections),  # 100
    )


if __name__=='__main__':
    main()
