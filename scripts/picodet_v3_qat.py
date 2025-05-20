# train_picodet_qat.py – minimal pipeline: COCO ➜ FP32 ➜ QAT ➜ INT8 ➜ ONNX (with NMS)
from __future__ import annotations
import argparse, random, time, warnings
from typing import List, Tuple
import traceback

import torch, torch.nn as nn
import torchvision.transforms.v2 as T_v2
import torchvision.datasets as tvsets
import torchvision.ops as tvops
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

try: 
    from picodet_lib import (
        PicoDet, get_backbone, VarifocalLoss, dfl_loss,
        build_dfl_targets
    )
except Exception:
    # when running manually in console, these are already loaded
    pass

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 128  # also subset for speed, 224

# ───────────────────── data & transforms ───────────────────────
import torchvision.transforms.functional as F_tv

def build_transforms(train: bool):
    tfs: List[torch.nn.Module] = []
    if train:
        tfs += [
            T_v2.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), antialias=True),
            T_v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T_v2.RandomHorizontalFlip()
        ]
    else:
        tfs.append(T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
    return T_v2.Compose(tfs)

def collate(batch):
    imgs, tgts = zip(*batch)
    img_ts = []
    for im in imgs:
        t = F_tv.pil_to_tensor(im).contiguous()  # uint8
        img_ts.append(t)
    stacked = torch.stack(img_ts)
    # normalise boxes to absolute pixels (they already are). nothing else.
    return stacked, list(tgts)

def get_loader(root: str, split: str, bsz: int, workers: int = 0, subset_size: int = None):
    ds = tvsets.CocoDetection(
        img_folder := f"{root}/{split}2017",
        ann_file := f"{root}/annotations/instances_{split}2017.json",
        transform=build_transforms(split == 'train')
    )
    ds_to_load = ds # By default, use the full dataset

    if subset_size is not None:
        if subset_size > len(ds):
            print(f"[WARNING] subset_size ({subset_size}) is larger than dataset size ({len(ds)}). Using full dataset for {split}.")
        else:
            all_indices = list(range(len(ds)))
            random.shuffle(all_indices) # Shuffle all indices
            selected_indices = all_indices[:subset_size] # Take the first N shuffled indices
            ds_to_load = Subset(ds, selected_indices)
            print(f"[INFO] Using a subset of the {split} dataset: {len(ds_to_load)} images.")
    return DataLoader(ds_to_load, batch_size=bsz, shuffle=split == 'train',
                      collate_fn=collate, num_workers=workers, pin_memory=True,
                      persistent_workers=bool(workers))

# ───────────────────── assigner (SimOTA, cached) ────────────────
class SimOTACache:
    def __init__(self, nc: int, ctr: float = 2.5, topk: int = 10):
        self.nc, self.r, self.k = nc, ctr, topk
        self.cache = {}

    @torch.no_grad()
    def __call__(self, f_shapes: Tuple[Tuple[int, int, int], ...], device: torch.device,
                 tgt: dict):
        # print(f"\n[SimOTACache DEBUG] --- Invoking __call__ --- Device: {device}")
        try:
            centres, strides_list = [], []
            # print(f"[SimOTACache DEBUG] f_shapes: {f_shapes}")

            for H, W, s_val in f_shapes: # Renamed s to s_val to avoid conflict
                # Use str(device) for cache key as torch.device object might not be consistently hashable
                key = (H, W, s_val, str(device))
                if key not in self.cache:
                    # print(f"[SimOTACache DEBUG] Cache miss for key: {key}. Computing centres.")
                    yv, xv = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device), indexing='ij')
                    c = torch.stack((xv, yv), dim=2).reshape(-1, 2).to(torch.float32) * float(s_val) + float(s_val) * 0.5
                    self.cache[key] = c
                    # print(f"[SimOTACache DEBUG] Cached centres for key {key} with shape {c.shape}")
                # else:
                    # print(f"[SimOTACache DEBUG] Cache hit for key: {key}.")
                centres.append(self.cache[key])
                strides_list.append(torch.full((H * W,), float(s_val), dtype=torch.float32, device=device))
            
            if not centres: # Should only happen if f_shapes is empty
                # print("[SimOTACache WARNING] centres list is empty (f_shapes likely empty). Returning empty tensors.")
                return torch.zeros(0, dtype=torch.bool, device=device), \
                       torch.zeros((0, self.nc), device=device), \
                       torch.zeros((0, 4), device=device)
            
            centres = torch.cat(centres, dim=0)
            strides = torch.cat(strides_list, dim=0)
            # print(f"[SimOTACache DEBUG] centres concatenated. Shape: {centres.shape}")
            # print(f"[SimOTACache DEBUG] strides concatenated. Shape: {strides.shape}")

            A = centres.size(0)
            M = tgt['boxes'].size(0)
            # print(f"[SimOTACache DEBUG] Num anchors (A): {A}, Num GT objects (M): {M}")
            # print(f"[SimOTACache DEBUG] tgt['boxes'].shape: {tgt['boxes'].shape}, tgt['labels'].shape: {tgt['labels'].shape}, tgt['labels'] values: {tgt['labels']}")

            if M == 0:
                print(f"[SimOTACache INFO] M == 0 (no GT objects). Returning zeros for {A} anchors.")
                return torch.zeros(A, dtype=torch.bool, device=device), \
                       torch.zeros((A, self.nc), device=device), \
                       torch.zeros((A, 4), device=device)

            # --- Crucial Label Validation ---
            if tgt['labels'].numel() > 0:
                max_label, min_label = tgt['labels'].max().item(), tgt['labels'].min().item()
                if max_label >= self.nc:
                    err_msg = f"Max label in tgt['labels'] ({max_label}) is >= self.nc ({self.nc}). Labels: {tgt['labels']}"
                    print(f"[SimOTACache CRITICAL ERROR] {err_msg}")
                    raise IndexError(err_msg)
                if min_label < 0:
                    err_msg = f"Min label in tgt['labels'] ({min_label}) is negative. Labels: {tgt['labels']}"
                    print(f"[SimOTACache CRITICAL ERROR] {err_msg}")
                    raise IndexError(err_msg)
            elif M > 0: # Boxes exist but no labels
                print(f"[SimOTACache WARNING] M={M} but tgt['labels'] is empty. Inconsistent target. Returning zeros.")
                return torch.zeros(A, dtype=torch.bool, device=device), \
                       torch.zeros((A, self.nc), device=device), \
                       torch.zeros((A, 4), device=device)

            cxcy = (tgt['boxes'][:, :2] + tgt['boxes'][:, 2:]) / 2
            dist = (centres[:, None, :] - cxcy[None, :, :]).abs().max(dim=-1).values # Shape (A, M)
            centre_mask = dist < self.r * strides[:, None] # Shape (A, M)
            # print(f"[SimOTACache DEBUG] centre_mask shape: {centre_mask.shape}, sum true: {centre_mask.sum()}")

            s_div_2 = strides[:, None] / 2
            anchor_candidate_boxes = torch.cat([centres - s_div_2, centres + s_div_2], dim=-1) # Shape (A, 4)
            # print(f"[SimOTACache DEBUG] anchor_candidate_boxes shape: {anchor_candidate_boxes.shape}")

            iou = tvops.box_iou(tgt['boxes'], anchor_candidate_boxes) # Shape (M, A)
            # print(f"[SimOTACache DEBUG] iou shape: {iou.shape}")
            
            # Original cls_cost was -torch.eye(self.nc)[tgt['labels']][None,:,:].max(-1).values.T
            # This simplifies to a tensor of -1s of shape (M,1) if labels are one-hot represented.
            cls_cost_val = -1.0 # Assuming this is the intended constant classification cost component.
            cls_cost = torch.full((M, 1), cls_cost_val, device=device, dtype=torch.float32)
            # print(f"[SimOTACache DEBUG] cls_cost shape: {cls_cost.shape}, value: {cls_cost_val}")

            # Cost matrix: lower is better. Penalty for not being in center region.
            cost = (1.0 - iou) + cls_cost + (~centre_mask.T) * 1e5 # Shape (M, A)
            # print(f"[SimOTACache DEBUG] cost matrix shape: {cost.shape}")

            fg_mask = centre_mask.any(dim=1) # Shape (A,), True for anchors in any GT's center region
            # print(f"[SimOTACache DEBUG] fg_mask shape: {fg_mask.shape}, num_fg_anchors: {fg_mask.sum()}")

            # For each anchor, find the GT it's best matched to (minimum cost)
            assigned_gt_idx_per_anchor = cost.argmin(dim=0) # Shape (A,)
            # print(f"[SimOTACache DEBUG] assigned_gt_idx_per_anchor shape: {assigned_gt_idx_per_anchor.shape}")

            cls_t = torch.zeros((A, self.nc), device=device)
            box_t = torch.zeros((A, 4), device=device)

            num_fg = fg_mask.sum().item()
            if num_fg > 0:
                # print(f"[SimOTACache DEBUG] Processing {num_fg} foreground anchors.")
                fg_assigned_gt_indices = assigned_gt_idx_per_anchor[fg_mask]
                # print(f"[SimOTACache DEBUG] fg_assigned_gt_indices (for fg anchors): {fg_assigned_gt_indices}")
                
                # These indices should be valid due to argmin logic and M > 0.
                # Max index in fg_assigned_gt_indices should be < M.
                if fg_assigned_gt_indices.max().item() >= M or fg_assigned_gt_indices.min().item() < 0:
                    err = f"Internal consistency error: fg_assigned_gt_indices out of bounds. Max: {fg_assigned_gt_indices.max().item()}, Min: {fg_assigned_gt_indices.min().item()}, M: {M}"
                    print(f"[SimOTACache CRITICAL ERROR] {err}")
                    raise IndexError(err)

                fg_gt_labels = tgt['labels'][fg_assigned_gt_indices]
                # print(f"[SimOTACache DEBUG] fg_gt_labels (for fg_anchors): {fg_gt_labels}")
                
                # Labels already validated against self.nc earlier.
                cls_t[fg_mask, fg_gt_labels] = 1.0
                
                fg_gt_boxes = tgt['boxes'][fg_assigned_gt_indices]
                box_t[fg_mask] = fg_gt_boxes
            # else:
                # print(f"[SimOTACache INFO] No foreground anchors found for this target.")
            
            # print(f"[SimOTACache DEBUG] --- Returning from __call__ --- fg_mask sum: {fg_mask.sum()}, cls_t sum: {cls_t.sum()}, box_t sum: {box_t.sum() if A > 0 else 0.0}")
            return fg_mask, cls_t, box_t

        except Exception as e_outer:
            print(f"\n[SimOTACache CRITICAL ERROR] Exception caught in __call__ method: {type(e_outer).__name__}: {e_outer}")
            traceback.print_exc()
            # Re-raise to make sure the error propagates and isn't masked as a None return.
            raise e_outer

# ───────────────────── COCO Label Mapping Helper ─────────────────────
def create_coco_contiguous_label_map(coco_api):
    """
    Creates a mapping from original COCO category IDs to contiguous 0-indexed labels.
    This map will include all categories present in the annotation file.
    The order of contiguous labels is determined by the sorted order of original COCO category IDs.
    """
    cat_ids = coco_api.getCatIds()  # Get all category IDs present in the annotation file
    # It's good practice to sort them to ensure a consistent mapping,
    # especially if self.nc is based on a specific ordered subset (like the 80 standard COCO classes).
    # If your model uses a specific predefined 80-class list, you'd map against that.
    # For a general case, sorting all present IDs works.
    sorted_cat_ids = sorted(cat_ids)
    
    contiguous_map = {coco_id: i for i, coco_id in enumerate(sorted_cat_ids)}
    
    # If your model is strictly for the 80 common COCO categories, you might want a fixed map:
    # For example:
    # coco_80_categories = [ # List of the 80 COCO category IDs in the desired order for your model
    #    1, 2, 3, ..., 90 # (actual IDs for the 80 classes)
    # ]
    # contiguous_map = {coco_id: i for i, coco_id in enumerate(coco_80_categories)}
    # This ensures that, e.g., COCO ID 1 always maps to 0, ID 2 to 1, etc., *if* they are in your list.

    print(f"[INFO] Created COCO contiguous label map. {len(contiguous_map)} categories mapped.")
    # print(f"   First 5 mappings: {list(contiguous_map.items())[:5]}")
    # print(f"   Last 5 mappings: {list(contiguous_map.items())[-5:]}")
    return contiguous_map

# ───────────────────── train / val loops ────────────────────────
def train_epoch(model: PicoDet, loader, opt, scaler, assigner: SimOTACache,
                device: torch.device, epoch: int, coco_label_map: dict): # coco_label_map is now required
    model.train(); t0, tot_loss_accum = time.time(), 0.
    
    # print(f"[train_epoch INFO] Starting Epoch {epoch}. Model nc: {model.head.nc}. Assigner nc: {assigner.nc}")
    # The assigner.nc should align with the number of classes produced by coco_label_map (or be >=).

    for i, (imgs, tgts_batch) in enumerate(loader):
        imgs = imgs.to(device)
        cls_preds_levels, obj_preds_levels, reg_preds_levels = model(imgs)
        bs = imgs.size(0)
        batch_total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_samples_with_loss = 0

        fmap_shapes = []
        for lv in range(len(cls_preds_levels)):
            H, W = cls_preds_levels[lv].shape[2:]
            # s = model.head.strides[lv]
            # or just use the tensor s = model.head.strides_buffer[lv]
            s = model.head.strides_buffer[lv].item() # NEW (use .item() if you need scalar Python float)
            fmap_shapes.append((H, W, s))

        for b_idx in range(bs):
            current_sample_annots_raw = tgts_batch[b_idx]
            
            # Process annotations for mapping
            processed_boxes = []
            processed_labels = []

            for annot_item in current_sample_annots_raw:
                if not isinstance(annot_item, dict) or 'bbox' not in annot_item or 'category_id' not in annot_item:
                    continue # Skip malformed annotations
                
                original_coco_id = annot_item['category_id']
                mapped_label = coco_label_map.get(original_coco_id)

                if mapped_label is not None and mapped_label < model.head.nc: # Ensure mapped label is valid for the model
                    processed_boxes.append([
                        annot_item['bbox'][0], annot_item['bbox'][1],
                        annot_item['bbox'][0] + annot_item['bbox'][2], annot_item['bbox'][1] + annot_item['bbox'][3]
                    ])
                    processed_labels.append(mapped_label)
                # else:
                    # print(f"[train_epoch DEBUG] Skipping ann: COCO ID {original_coco_id} mapped to {mapped_label}, model.nc={model.head.nc}")


            if not processed_labels: # No valid annotations for this sample
                continue

            gt_boxes = torch.tensor(processed_boxes, dtype=torch.float32, device=device)
            gt_labels = torch.tensor(processed_labels, dtype=torch.int64, device=device)
            
            target_dict_for_assigner = {'boxes': gt_boxes, 'labels': gt_labels}
            fg_mask, cls_targets, box_targets = assigner(fmap_shapes, device, target_dict_for_assigner)
            
            num_fg = fg_mask.sum().item()
            if num_fg == 0:
                continue

            # --- Loss Calculation (unchanged from here for this part) ---
            cls_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, model.head.nc) for lvl in cls_preds_levels], dim=0)
            obj_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1) for lvl in obj_preds_levels], dim=0)
            reg_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, 4 * (model.head.reg_max + 1)) for lvl in reg_preds_levels], dim=0)

            cls_p_fg = cls_p[fg_mask]
            obj_p_fg = obj_p[fg_mask]
            reg_p_fg = reg_p[fg_mask]
            cls_targets_fg = cls_targets[fg_mask]
            box_targets_fg = box_targets[fg_mask]

            joint_logits_fg = cls_p_fg + obj_p_fg.unsqueeze(-1)
            loss_cls = VarifocalLoss()(joint_logits_fg, cls_targets_fg)

            # strides_all_anchors = torch.cat([torch.full(((IMG_SIZE // s_val) ** 2,), float(s_val), device=device) for H_lvl, W_lvl, s_val in fmap_shapes], dim=0)
            # strides_fg = strides_all_anchors[fg_mask]
            # --- Corrected strides_all_anchors construction ---
            strides_tensor_list = []
            for H_level, W_level, s_level_val in fmap_shapes:
                num_anchors_this_level = H_level * W_level
                strides_tensor_list.append(
                    torch.full((num_anchors_this_level,), float(s_level_val), device=device)
                )
            strides_all_anchors = torch.cat(strides_tensor_list, dim=0)
            # Now, len(strides_all_anchors) will be sum(H_level * W_level) from fmap_shapes,
            # which *must* match len(fg_mask) from SimOTACache because both use the same H_level, W_level.
            
            # fg_mask comes from assigner, and assigner uses fmap_shapes.
            # Its length is A = sum of (H*W for each level in fmap_shapes).
            # The new strides_all_anchors also has length A.
            # So, strides_all_anchors[fg_mask] should now work.
            if strides_all_anchors.shape[0] != fg_mask.shape[0]:
                print("[CRITICAL ERROR in train_epoch] Mismatch after strides_all_anchors correction!")
                print(f"  strides_all_anchors.shape: {strides_all_anchors.shape}")
                print(f"  fg_mask.shape: {fg_mask.shape}")
                print(f"  fmap_shapes used for both: {fmap_shapes}")
                # This should ideally not happen with the fix.
                # Example values from error: fg_mask is 1029, strides_all_anchors was 336.
                # After fix, strides_all_anchors should also be 1029.
            
            strides_fg = strides_all_anchors[fg_mask] # This line was failing
            
            target_offsets_for_dfl = (box_targets_fg / strides_fg.unsqueeze(-1)).clamp(min=0., max=model.head.reg_max - 1e-6)

            # target_offsets_for_dfl = (box_targets_fg / strides_fg.unsqueeze(-1)).clamp(min=0., max=model.head.reg_max - 1e-6)
            dfl_target_dist = build_dfl_targets(target_offsets_for_dfl, model.head.reg_max)
            loss_dfl = dfl_loss(reg_p_fg, dfl_target_dist)

            current_num_fg_for_iou = reg_p_fg.shape[0] # Same as num_fg
            if current_num_fg_for_iou == 0:
                loss_iou = torch.tensor(0.0, device=device)
            else:
                # reshaped_reg_p_fg_for_dfl = reg_p_fg.view(current_num_fg_for_iou, -1, 1, 1)
                # pred_ltrb_offsets_raw = model.head._dfl_to_ltrb(reshaped_reg_p_fg_for_dfl)
                # pred_ltrb_offsets_fg = pred_ltrb_offsets_raw.view(current_num_fg_for_iou, 4)

                # This matches the 2D case in _dfl_to_ltrb_original_for_training_etc
                pred_ltrb_offsets_fg = model.head._dfl_to_ltrb_original_for_training_etc(reg_p_fg)
                # pred_ltrb_offsets_fg will be (current_num_fg_for_iou, 4)
            
                centres_all_anchors = torch.cat(
                    [assigner.cache[(H_lvl, W_lvl, s_lvl, str(device))] 
                     for H_lvl, W_lvl, s_lvl in fmap_shapes], 
                    dim=0
                )
                centres_fg = centres_all_anchors[fg_mask]
                
                pred_boxes_fg = torch.stack((
                    centres_fg[:, 0] - pred_ltrb_offsets_fg[:, 0], centres_fg[:, 1] - pred_ltrb_offsets_fg[:, 1],
                    centres_fg[:, 0] + pred_ltrb_offsets_fg[:, 2], centres_fg[:, 1] + pred_ltrb_offsets_fg[:, 3]
                ), dim=1)
                
                loss_iou = tvops.complete_box_iou_loss(pred_boxes_fg, box_targets_fg, reduction='sum') / current_num_fg_for_iou
            
            current_sample_total_loss = loss_cls + loss_dfl + loss_iou
            batch_total_loss += current_sample_total_loss
            num_samples_with_loss +=1
        # ... (rest of batch/epoch aggregation and printing) ...
        if num_samples_with_loss > 0:
            averaged_batch_loss = batch_total_loss / num_samples_with_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(averaged_batch_loss).backward()
            scaler.step(opt); scaler.update()
            tot_loss_accum += averaged_batch_loss.item() * num_samples_with_loss
        if i % 50 == 0:
            processed_batches = i + 1
            avg_loss_so_far = tot_loss_accum / (processed_batches * num_samples_with_loss if processed_batches * num_samples_with_loss > 0 else 1) # Approximate
            time_per_batch = (time.time() - t0) / processed_batches if processed_batches > 0 else 0
            print(f"E{epoch} {i:04d}/{len(loader)} loss {avg_loss_so_far:.3f} {time_per_batch:.2f}s/batch")
    
    return tot_loss_accum / len(loader) if len(loader) > 0 else 0.0
      
@torch.no_grad()
def quick_val_iou(model: PicoDet, loader, device):
    model.eval()
    total_iou_sum = 0.
    num_images_with_gt = 0 # Count images that actually have ground truth boxes

    for imgs_batch, tgts_batch in loader: # Renamed for clarity
        # imgs_batch: (BatchSize, C, H, W)
        # tgts_batch: List of lists of annotation dicts. Length is BatchSize.
        # tgts_batch[sample_idx] is a list of dicts for that sample.

        pred_boxes_batch, pred_scores_batch, pred_labels_batch = model(imgs_batch.to(device))
        # pred_boxes_batch, etc., are lists of tensors, one per image in the batch

        for i in range(imgs_batch.size(0)): # Iterate over each image in the batch
            # Get ground truth for the i-th image
            current_img_annots_raw = tgts_batch[i] # This is a list of dicts
            
            gt_boxes_list = []
            for annot in current_img_annots_raw:
                if 'bbox' in annot: # Basic check
                    # Assuming bbox is [x,y,w,h]
                    x, y, w, h = annot['bbox']
                    gt_boxes_list.append([x, y, x + w, y + h]) # Convert to x1,y1,x2,y2

            if not gt_boxes_list: # No ground truth boxes for this image
                continue

            gt_boxes_tensor = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
            
            # Predicted boxes for the i-th image from the model output
            # model output (bx, sc, lb) is typically a list of tensors for each image in the batch.
            # So bx[i] should be the tensor of predicted boxes for the i-th image.
            # Ensure model output pred_boxes_batch[i] is already on the correct device.
            # If model is on device, its output should also be.
            predicted_boxes_for_img = pred_boxes_batch[i] 

            if predicted_boxes_for_img.numel() == 0: # No predictions for this image
                # If there are GTs but no predictions, IoU is 0 for all GTs with preds.
                # The average IoU for this image would be 0.
                # Depending on how you want to score this, you might add 0 to total_iou_sum
                # and increment num_images_with_gt. For simplicity, let's assume if no preds, IoU is 0.
                # If you only want to average over images with predictions, then 'continue' here.
                # For mAP style, GTs without matching preds are False Positives or contribute to recall misses.
                # Here, we are calculating a simple average IoU of best matches.
                num_images_with_gt += 1 # Still counts as an image we tried to evaluate
                continue


            # Calculate IoU between predicted boxes and ground truth boxes for this image
            # tvops.box_iou(boxes1, boxes2) returns (N, M) tensor where N=len(boxes1), M=len(boxes2)
            # predicted_boxes_for_img: (Num_preds, 4)
            # gt_boxes_tensor: (Num_GTs, 4)
            # iou_matrix will be (Num_preds, Num_GTs)
            iou_matrix = tvops.box_iou(predicted_boxes_for_img, gt_boxes_tensor)

            if iou_matrix.numel() == 0: # Should not happen if both preds and GTs exist
                num_images_with_gt +=1
                continue
                
            # For each predicted box, find the max IoU with any GT box.
            # Or, for each GT box, find the max IoU with any predicted box.
            # The original code did: iou.max(1)[0] which means for each PREDICTED box, find best GT match.
            # Let's stick to that for now. (Num_preds,)
            # If you want mAP-style, usually it's for each GT, find best pred.
            
            if iou_matrix.shape[0] > 0: # If there are predictions
                # max_iou_per_pred, _ = iou_matrix.max(dim=1) # (Num_preds,) - max IoU for each prediction
                # image_avg_iou = max_iou_per_pred.mean().item()
                
                # A more common metric for "mean IoU" in this context is often:
                # For each GT, find its best matching prediction. Average these IoUs.
                if iou_matrix.shape[1] > 0: # If there are GTs
                    max_iou_per_gt, _ = iou_matrix.max(dim=0) # (Num_GTs,) - max IoU for each GT
                    image_avg_iou = max_iou_per_gt.mean().item()
                    total_iou_sum += image_avg_iou
                    num_images_with_gt += 1
            # If no predictions but GTs exist, image_avg_iou is 0, already handled by continue earlier if desired.
            # Or if we count it:
            # elif iou_matrix.shape[1] > 0: # No preds, but GTs exist
            #    num_images_with_gt += 1 # image_avg_iou is effectively 0

    return total_iou_sum / num_images_with_gt if num_images_with_gt > 0 else 0.

# ───────────────────── QAT helpers ───────────────────────────────
from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfig
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

def qat_prepare(model: nn.Module, example: torch.Tensor):
    qmap = get_default_qat_qconfig_mapping('x86')
    wobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    aobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_channel_affine)
    qmap = qmap.set_global(QConfig(aobs, wobs))
    # skip preprocess from quant
    qmap = qmap.set_module_name('pre', None)
    return prepare_qat_fx(model.cpu(), qmap, (example,))

def unwrap_dataset(ds):
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds

def warn_missing_categories(coco_api, subset_indices, base_dataset):
    """
    Warn if some COCO categories are not present in the subset.
    
    Parameters:
    - coco_api: COCO object (from dataset.coco)
    - subset_indices: indices into the base COCO dataset (not subset)
    - base_dataset: unwrapped base dataset (CocoDetection)
    """
    full_cat_ids = set(coco_api.getCatIds())

    # Collect category IDs actually used in the subset annotations
    used_cat_ids = set()
    for ds_idx in subset_indices:
        _, anns = base_dataset[ds_idx]  # ← base_dataset is the full CocoDetection dataset
        used_cat_ids.update(ann['category_id'] for ann in anns)

    missing = full_cat_ids - used_cat_ids
    if missing:
        print(f"[WARNING] {len(missing)} COCO categories are not present in the selected subset:")
        print(f"   Missing category IDs: {sorted(missing)}")

# ───────────────────── main ─────────────────────────────────────
def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', choices=['mnv3', 'mnv4s', 'mnv4m'], default='mnv3')
    pa.add_argument('--epochs', type=int, default=5) # Reduced for quick test
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    cfg = pa.parse_args(argv)

    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(cfg.device)
    print(f'[INFO] device = {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, ckpt=None)
    model = PicoDet(
        backbone, 
        feat_chs,
        num_classes=80, # Or from config/dataset
        neck_out_ch=96, # Or from config
        img_size=IMG_SIZE,
        # Optionally pass head params if they are configurable:
        # head_reg_max=...,
        # head_max_det=...,
        # head_score_thresh=...,
        # head_nms_iou=...
    ).to(dev)

    tr_loader = get_loader(cfg.coco_root, 'train', cfg.batch, cfg.workers, subset_size=1200)
    vl_loader = get_loader(cfg.coco_root, 'val', cfg.batch, cfg.workers, subset_size=200)

    train_base_ds = unwrap_dataset(tr_loader.dataset)
    # val_base_ds = unwrap_dataset(vl_loader.dataset)
    if hasattr(train_base_ds, 'coco') and train_base_ds.coco is not None:
        coco_label_map = create_coco_contiguous_label_map(train_base_ds.coco)
    
        if isinstance(tr_loader.dataset, torch.utils.data.Subset):
            warn_missing_categories(train_base_ds.coco, tr_loader.dataset.indices, train_base_ds)

        if len(coco_label_map) != model.head.nc:
            print(f"[main WARNING] Number of categories in generated map ({len(coco_label_map)}) "
                  f"does not match model.head.nc ({model.head.nc}). "
                  "Annotations will be filtered in train_epoch if mapped label >= model.head.nc.")
    else:
        raise RuntimeError("Could not access COCO API to create label map. "
                           "Ensure CocoDetection dataset is used and initialized correctly.")

    opt = SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-5)
    sch = CosineAnnealingLR(opt, cfg.epochs, 5e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    
    # Assigner.nc should match model.head.nc, which is the number of *mapped* classes the model predicts.
    assigner = SimOTACache(model.head.nc)

    for ep in range(cfg.epochs):
        l = train_epoch(model, tr_loader, opt, scaler, assigner, dev, ep, coco_label_map)
        # quick_val_iou will also need to use the coco_label_map if it processes raw category_ids for mAP
        # For now, assuming quick_val_iou uses model outputs which are already in mapped space.
        m = quick_val_iou(model, vl_loader, dev)
        sch.step()
        print(f'Epoch {ep + 1}/{cfg.epochs}  loss {l:.3f}  IoU {m:.3f}')
    
    # ... (QAT and ONNX export - remember to pass coco_label_map to QAT's train_epoch)
    example = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    qat = qat_prepare(model, example).to(dev).train()
    for p in qat.backbone.parameters():
        p.requires_grad = True
    opt_q = SGD(qat.parameters(), lr=0.002, momentum=0.9)
    scaler_q = torch.amp.GradScaler(enabled=False)
    for qep in range(5):
        lq = train_epoch(qat, tr_loader, opt_q, scaler_q, assigner, dev, qep, coco_label_map) # Pass map here
        print(f'[QAT] {qep + 1}/5  loss {lq:.3f}')

    qat.cpu().eval()
    int8 = convert_fx(qat)

    # ---------------- ONNX export ------------------------------
    int8.eval()
    torch.onnx.export(int8, example, cfg.out,
                      input_names=['images'],
                      output_names=['boxes', 'scores', 'labels'],
                      dynamic_axes={'images': {0: 'B', 2: 'H', 3: 'W'},
                                    'boxes': {0: 'B'},
                                    'scores': {0: 'B'},
                                    'labels': {0: 'B'}},
                      opset_version=18, do_constant_folding=True)
    print(f'[SAVE] ONNX → {cfg.out}')

if __name__ == '__main__':
    main()
