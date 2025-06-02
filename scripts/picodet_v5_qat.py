# train_picodet_qat.py – minimal pipeline: COCO ➜ FP32 ➜ QAT ➜ INT8 ➜ ONNX (with NMS)
from __future__ import annotations
import argparse, random, time, warnings, math
from typing import List, Tuple
import traceback

import torch, torch.nn as nn
import torchvision.transforms.v2 as T_v2
import torchvision.datasets as tvsets
import torchvision.ops as tvops
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import onnx
from onnx import TensorProto as TP, helper as oh

try: 
    from picodet_lib_v2 import (
        PicoDet, get_backbone, VarifocalLoss, dfl_loss,
        build_dfl_targets, PicoDetHead
    )
except Exception:
    # when running manually in console, these are already loaded
    pass

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 256  # also subset for speed, 224, PicoDet’s anchors assume stride-divisible sizes. Divisible by 32

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
# picodet_lib.py (SimOTACache class - UPDATED)

class SimOTACache:
    def __init__(self, nc: int, ctr: float = 2.5, topk: int = 10, cls_cost_weight: float = 0.5):
        """
        SimOTA Assigner with caching for anchor centers.

        Args:
            nc (int): Number of classes.
            ctr (float): Center radius factor. Anchors must be within ctr * stride of a GT center.
            topk (int): Max number of anchors to consider per GT for dynamic-k selection (candidate_k).
            cls_cost_weight (float): Weight for the classification cost term in the matching cost matrix.
                                     Should be > 0 for multi-class to prevent high-IoU wrong-class matches.
                                     Set to 0 if you want to revert to only localization cost.
        """
        self.nc = nc
        self.r = ctr
        self.k = topk
        self.cls_cost_weight = cls_cost_weight # New parameter for classification cost
        self.cache = {} # Caches anchor center coordinates and strides per feature level

    @torch.no_grad()
    def __call__(self, f_shapes: Tuple[Tuple[int, int, int], ...], device: torch.device,
                 tgt: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns:
        #   fg_mask (A,): Boolean mask, True for foreground anchors.
        #   assigned_gt_labels_full (A,): Long tensor, GT label index (0..nc-1) for fg_mask positions, -1 elsewhere.
        #   assigned_gt_boxes_full (A, 4): Float tensor, GT box [x1,y1,x2,y2] for fg_mask positions, zeros elsewhere.
        #   assigned_iou_full (A,): Float tensor, IoU with assigned GT for fg_mask positions, zeros elsewhere.
        try:
            centres_list, strides_list = [], []
            for H, W, s_val_level in f_shapes:
                key = (H, W, s_val_level, str(device))
                if key not in self.cache:
                    yv, xv = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device), indexing='ij')
                    s_float = float(s_val_level)
                    level_centers = (torch.stack((xv, yv), dim=2).reshape(-1, 2).to(torch.float32) + 0.5) * s_float
                    self.cache[key] = level_centers
                
                centres_list.append(self.cache[key])
                strides_list.append(torch.full((H * W,), float(s_val_level), dtype=torch.float32, device=device))
            
            if not centres_list:
                return (torch.zeros(0, dtype=torch.bool, device=device),
                        torch.full((0,), -1, dtype=torch.long, device=device),
                        torch.zeros((0, 4), device=device),
                        torch.zeros(0, device=device))

            anchor_centers = torch.cat(centres_list, dim=0) # (A, 2)
            anchor_strides = torch.cat(strides_list, dim=0) # (A,)
            A = anchor_centers.size(0) # Total number of anchors
            
            gt_boxes = tgt['boxes']    # (M, 4)
            gt_labels = tgt['labels']  # (M,)
            M = gt_boxes.size(0)       # Number of GT objects

            if M == 0:
                return (torch.zeros(A, dtype=torch.bool, device=device),
                        torch.full((A,), -1, dtype=torch.long, device=device),
                        torch.zeros((A, 4), device=device),
                        torch.zeros(A, device=device))

            # --- Label Validation ---
            if gt_labels.numel() > 0:
                if gt_labels.max().item() >= self.nc or gt_labels.min().item() < 0:
                    raise IndexError(f"GT labels out of bounds [0, {self.nc-1}]. Got min: {gt_labels.min()}, max: {gt_labels.max()}")
            elif M > 0:
                 return (torch.zeros(A, dtype=torch.bool, device=device),
                        torch.full((A,), -1, dtype=torch.long, device=device),
                        torch.zeros((A, 4), device=device),
                        torch.zeros(A, device=device))

            # --- Cost matrix preparation (common terms) ---
            # 1. IoU between GTs and anchor boxes
            s_div_2 = anchor_strides.unsqueeze(1) / 2.0
            anchor_candidate_boxes = torch.cat([anchor_centers - s_div_2, anchor_centers + s_div_2], dim=1)
            iou = tvops.box_iou(gt_boxes, anchor_candidate_boxes) # Shape: (M, A)

            # 2. Center Prior Mask
            gt_centers_xy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0 # (M, 2)
            dist = (anchor_centers.unsqueeze(1) - gt_centers_xy.unsqueeze(0)).abs().max(dim=-1).values # (A, M)
            center_region_mask = dist < (self.r * anchor_strides.unsqueeze(1)) # (A, M)

            # --- Dynamic-k Candidate Selection (Foreground Estimation) ---
            # This selects initial candidates for fg_mask.
            iou_sum_per_gt = iou.sum(dim=1) # (M,)
            # Corrected dynamic_ks: Clamp also by self.k (max candidates per GT)
            dynamic_ks = torch.clamp(iou_sum_per_gt.int(), min=1, max=self.k) # (M,)

            fg_mask_candidates = torch.zeros(A, dtype=torch.bool, device=device)
            for gt_idx in range(M):
                # Anchors in the center region of this GT
                anchors_in_center_for_gt = center_region_mask[:, gt_idx] # (A,)
                
                if not anchors_in_center_for_gt.any():
                    continue # No anchors in center region for this GT

                # Consider only IoUs of anchors within the center region for this GT
                # Mask out IoUs of anchors far from this GT
                ious_for_gt_in_center = iou[gt_idx] * anchors_in_center_for_gt # (A,)
                
                # Number of anchors to select for this GT:
                # min(dynamic_k_for_this_gt, num_anchors_actually_in_center_region)
                # Also, k cannot exceed the total number of anchors A.
                num_candidates_for_gt = min(dynamic_ks[gt_idx].item(), anchors_in_center_for_gt.sum().item())
                num_candidates_for_gt = min(num_candidates_for_gt, A) # Ensure k <= A

                if num_candidates_for_gt > 0:
                    # Select top-k anchors (by IoU, within center region) for this GT
                    _, topk_indices_for_gt = ious_for_gt_in_center.topk(num_candidates_for_gt, largest=True)
                    fg_mask_candidates[topk_indices_for_gt] = True
            
            # --- Cost Matrix for Final Assignment ---
            # Only consider anchors that are candidates (in fg_mask_candidates)
            # Cost_loc = 1.0 - IoU
            cost_loc = (1.0 - iou) # (M, A)

            # Cost_cls: Penalizes assigning an anchor to a GT of a different class.
            # For multi-class, this helps prevent high-IoU but wrong-class matches.
            # Create a one-hot representation of GT labels: (M, nc)
            # gt_labels_one_hot = F.one_hot(gt_labels, num_classes=self.nc).float() # (M, nc)
            
            # For each anchor, imagine it's predicted uniformly for all classes (or some other prior).
            # A simple cls cost can be derived by assuming an anchor *could* match any class,
            # and penalizing if it doesn't match the GT's class.
            # Or, more directly, for a given (gt_m, anchor_a) pair, if we assign anchor_a to gt_m,
            # the classification "loss" if anchor_a's (hypothetical perfect) prediction for gt_m's class is 1
            # and 0 for others.
            # A simpler heuristic: add a cost if anchor_class_prediction != gt_class.
            # Since we don't have anchor predictions here, we can introduce a class cost
            # that is low when GT label matches a "hypothetical" anchor prediction (which we don't have).
            # More practically, from TAL papers like OTA:
            # cls_cost is often based on -log(pred_score_for_gt_class)
            # If no prediction:
            # We can use a fixed penalty if this GT is not the "primary" class for an anchor.
            # For prediction-free:
            #   Consider a scenario where anchor_a is assigned to gt_m.
            #   If this assignment is "good" class-wise, cost_cls should be low.
            #   The original SimOTA in YOLOX uses predicted classification scores.
            #   If prediction-free, a common simplification is to either omit cls_cost
            #   or use a very simple one.
            #   Let's use a simplified one that penalizes assigning anchor `a` to `gt_m`
            #   if `gt_m`'s class is not the class `anchor_a` is "most likely" to predict (if it could).
            #   Since we don't know what an anchor is likely to predict, we can use a small,
            #   uniform classification cost. This is a bit of a placeholder if not using scores.
            #   The provided suggestion `if multi-class keep a small class-cost (e.g. 0.5)`
            #   implies a constant cost for *all* potential (gt, anchor) pairs if they were to be matched.
            #   This doesn't help discriminate *between* GTs for a given anchor if their IoUs are similar.

            # Let's re-think the `cls_cost` from the user's suggestion "keep a small class-cost (e.g. 0.5)".
            # This usually means adding a fixed value to the cost if this pair is chosen.
            # This doesn't differentiate class compatibility if IoUs are similar.
            # However, if the intent is to ensure that the IoU needs to be significantly better
            # to overcome this "base" cost of matching, then it's a global addition.

            # A more effective prediction-free class cost would be if we had prior class probabilities for anchors.
            # Lacking that, let's make it a small constant if self.cls_cost_weight > 0, effectively
            # making localization (IoU) slightly more important to overcome this base cost.
            cost_cls = torch.zeros_like(cost_loc) # (M, A)
            if self.cls_cost_weight > 0 and self.nc > 1:
                 # This is a constant addition. Its main effect is to scale the overall cost.
                 # It doesn't help an anchor decide *which* GT class to match if multiple GTs
                 # have similar IoU.
                 # A more sophisticated prediction-free cls_cost might involve e.g.,
                 # if an anchor is very specialized for one class due to its aspect ratio/scale.
                 # For now, let's follow the spirit of a "small class-cost".
                 cost_cls += self.cls_cost_weight # Add a small constant cost

            # Total cost:
            # Add a large penalty if anchor is not in GT's center region (transposed mask)
            # Add a large penalty if anchor is not a candidate (from dynamic-k selection)
            cost_matrix = (
                cost_loc
                + cost_cls
                + (~center_region_mask.T) * 1e4  # Penalty for not in center region
                + (~fg_mask_candidates.unsqueeze(0)) * 1e4 # Penalty for not being a dynamic-k candidate
            ) # Shape: (M, A)

            # For each anchor (column in cost_matrix), find the GT (row) that has the minimum cost
            # This resolves conflicts if an anchor is a candidate for multiple GTs.
            assigned_gt_idx_per_anchor = cost_matrix.argmin(dim=0) # Shape: (A,)

            # --- Final Foreground Mask and Targets ---
            # An anchor is a final foreground if:
            # 1. It was a candidate from dynamic-k selection (fg_mask_candidates)
            # 2. It's in the center region of the GT it was finally assigned to.
            # 3. The GT it was assigned to must also accept this anchor (based on cost).
            
            # Gather the center_region_mask values for the (assigned_gt, anchor) pairs
            # assigned_gt_idx_per_anchor: (A,)
            # center_region_mask: (A, M)
            # We need center_region_mask[anchor_idx, assigned_gt_idx_per_anchor[anchor_idx]]
            # This can be done with gather or by iterating.
            # Easier: make final_fg_mask based on fg_mask_candidates and then check if the assigned GT
            # for that candidate also considers it "in_center".
            
            final_fg_mask = torch.zeros(A, dtype=torch.bool, device=device)
            
            # Indices of anchors that were candidates
            candidate_anchor_indices = torch.where(fg_mask_candidates)[0] # (num_candidates,)

            if candidate_anchor_indices.numel() > 0:
                # For these candidates, get their assigned GT index
                gt_indices_for_candidates = assigned_gt_idx_per_anchor[candidate_anchor_indices] # (num_candidates,)

                # Check if these candidates are in the center region of their *assigned* GT
                # center_region_mask is (A, M)
                # We need: center_region_mask[candidate_anchor_indices[i], gt_indices_for_candidates[i]]
                in_center_for_assigned_gt = center_region_mask[candidate_anchor_indices, gt_indices_for_candidates] # (num_candidates,)

                # Final foreground anchors are candidates that are also in center of their assigned GT
                final_fg_anchor_indices_relative_to_candidates = torch.where(in_center_for_assigned_gt)[0]
                
                if final_fg_anchor_indices_relative_to_candidates.numel() > 0:
                    actual_final_fg_indices = candidate_anchor_indices[final_fg_anchor_indices_relative_to_candidates]
                    final_fg_mask[actual_final_fg_indices] = True

            # --- Prepare Output Tensors ---
            assigned_gt_labels_full = torch.full((A,), -1, dtype=torch.long, device=device)
            assigned_gt_boxes_full  = torch.zeros((A, 4), dtype=torch.float32, device=device)
            assigned_iou_full       = torch.zeros((A,), dtype=torch.float32, device=device)

            num_final_fg = final_fg_mask.sum().item()
            if num_final_fg > 0:
                # Get the GT indices ONLY for the final foreground anchors
                gt_indices_for_final_fg_anchors = assigned_gt_idx_per_anchor[final_fg_mask]

                assigned_gt_labels_full[final_fg_mask] = gt_labels[gt_indices_for_final_fg_anchors]
                assigned_gt_boxes_full[final_fg_mask]  = gt_boxes[gt_indices_for_final_fg_anchors]
                
                # IoU between each final_fg_anchor and its *assigned* GT.
                final_fg_mask_indices_tensor = torch.where(final_fg_mask)[0]
                assigned_iou_full[final_fg_mask] = iou[gt_indices_for_final_fg_anchors, final_fg_mask_indices_tensor]

            return final_fg_mask, assigned_gt_labels_full, assigned_gt_boxes_full, assigned_iou_full

        except Exception as e_outer:
            print(f"\n[SimOTACache CRITICAL ERROR] Exception caught in __call__ method: {type(e_outer).__name__}: {e_outer}")
            import traceback
            traceback.print_exc()
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
    sorted_cat_ids = sorted(cat_ids)
    
    contiguous_map = {coco_id: i for i, coco_id in enumerate(sorted_cat_ids)}

    print(f"[INFO] Created COCO contiguous label map. {len(contiguous_map)} categories mapped.")
    return contiguous_map

# ───────────────────── train / val loops ────────────────────────
def train_epoch(
        model: nn.Module, loader, opt, scaler, assigner: SimOTACache, # Assigner type is now the new one
        device: torch.device, epoch: int, coco_label_map: dict,
        head_nc_for_loss: int, 
        head_reg_max_for_loss: int,
        dfl_project_buffer_for_decode: torch.Tensor,
        max_epochs: int = 500, # Make sure this matches total training epochs for alpha_dyn
        quality_floor_vfl: float = 0.2 # Example value, can be tuned
):
    model.train()
    _, tot_loss_accum = time.time(), 0.
    total_samples_contributing_to_loss_epoch = 0
    
    for i, (imgs, tgts_batch) in enumerate(loader):
        imgs = imgs.to(device)
        model_outputs = model(imgs) 

        if not model.training:
             raise RuntimeError("train_epoch called with model not in training mode.")
        if len(model_outputs) != 4:
            raise ValueError(
                f"Expected 4 outputs from model in training mode (preds_cls, preds_obj, preds_reg, strides), got {len(model_outputs)}"
            )
        cls_preds_levels, obj_preds_levels, reg_preds_levels, strides_per_level_tensors = model_outputs

        bs = imgs.size(0)
        fmap_shapes = [] 
        for lv in range(len(cls_preds_levels)):
            H, W = cls_preds_levels[lv].shape[2:]
            s = strides_per_level_tensors[lv].item() 
            fmap_shapes.append((H, W, s))

        batch_total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_samples_with_loss_in_batch = 0

        for b_idx in range(bs):
            current_sample_annots_raw = tgts_batch[b_idx]
            processed_boxes, processed_labels = [], []
            for annot_item in current_sample_annots_raw:
                if not isinstance(annot_item, dict) or 'bbox' not in annot_item or 'category_id' not in annot_item:
                    continue
                original_coco_id = annot_item['category_id']
                mapped_label = coco_label_map.get(original_coco_id)
                if mapped_label is not None and mapped_label < head_nc_for_loss:
                    processed_boxes.append([
                        annot_item['bbox'][0], annot_item['bbox'][1],
                        annot_item['bbox'][0] + annot_item['bbox'][2], annot_item['bbox'][1] + annot_item['bbox'][3]
                    ])
                    processed_labels.append(mapped_label)
            
            if not processed_labels: continue

            gt_boxes_img = torch.tensor(processed_boxes, dtype=torch.float32, device=device)
            gt_labels_img = torch.tensor(processed_labels, dtype=torch.int64, device=device)
            target_dict_for_assigner = {'boxes': gt_boxes_img, 'labels': gt_labels_img}
            
            if assigner.nc != head_nc_for_loss: # This check is good to keep
                 warnings.warn(f"Warning: assigner.nc ({assigner.nc}) != head_nc_for_loss ({head_nc_for_loss}).")

            # fg_mask_img: (A,), bool
            # assigned_gt_labels_img_full: (A,), long, label for fg_mask positions, -1 else
            # assigned_gt_boxes_img_full: (A, 4), float, box for fg_mask positions, 0 else
            # assigned_iou_img_full: (A,), float, IoU for fg_mask positions, 0 else
            fg_mask_img, assigned_gt_labels_img_full, \
            assigned_gt_boxes_img_full, assigned_iou_img_full = assigner(
                fmap_shapes, device, target_dict_for_assigner
            )

            num_fg_img = fg_mask_img.sum().item()
            if num_fg_img == 0: continue 

            # Concatenate predictions for all levels for this image
            cls_p_img = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, head_nc_for_loss) for lvl in cls_preds_levels], dim=0)
            obj_p_img = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1) for lvl in obj_preds_levels], dim=0)
            reg_p_img = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, 4 * (head_reg_max_for_loss + 1)) for lvl in reg_preds_levels], dim=0)

            # --- Varifocal Loss (VFL) ---
            joint_logits_img = cls_p_img + obj_p_img.unsqueeze(-1)

            loss_cls = torch.tensor(0.0, device=device) # Initialize loss_cls

            use_focal_loss = True
            if use_focal_loss:
                # Standard Focal Loss parameters
                focal_loss_alpha = 0.25
                focal_loss_gamma = 2.0
                focal_targets_img = torch.zeros_like(joint_logits_img) # (A, C)
                fg_indices = fg_mask_img.nonzero(as_tuple=True)[0]
                gt_labels_for_fg = assigned_gt_labels_img_full[fg_mask_img]
                focal_targets_img[fg_indices, gt_labels_for_fg] = 1.0

                loss_cls_unreduced = tvops.sigmoid_focal_loss(
                    joint_logits_img,
                    focal_targets_img,
                    alpha=focal_loss_alpha,
                    gamma=focal_loss_gamma,
                    reduction='sum'
                )
                loss_cls = loss_cls_unreduced / float(num_fg_img) # Normalize by number of positive anchors
            else:  # Varifocal Loss, isn't working as well, might be buggy
                vfl_targets_img = torch.zeros_like(joint_logits_img)
                fg_gt_labels = assigned_gt_labels_img_full[fg_mask_img]
                fg_iou_quality = assigned_iou_img_full[fg_mask_img]

                final_vfl_quality = torch.maximum(fg_iou_quality, torch.tensor(quality_floor_vfl, device=device))

                vfl_targets_img[fg_mask_img.nonzero(as_tuple=True)[0], fg_gt_labels] = final_vfl_quality

                progress = epoch / max_epochs if max_epochs > 0 else 0.0
                alpha_dyn = 0.25 + (0.75 - 0.25) * 0.5 * (1 - math.cos(math.pi * progress))
                
                # Using gamma=2.0 for VFL as generally recommended
                current_vfl_instance = VarifocalLoss(alpha=alpha_dyn, gamma=2.0, reduction='sum')
                loss_cls_unreduced = current_vfl_instance(joint_logits_img, vfl_targets_img)
                loss_cls = loss_cls_unreduced / float(num_fg_img) # Normalize by num_fg_img

            # --- Distribution Focal Loss (DFL) & IoU Loss for BBox Regression ---
            # These losses apply only to foreground anchors
            reg_p_fg_img = reg_p_img[fg_mask_img]                         
            box_targets_fg_img = assigned_gt_boxes_img_full[fg_mask_img] 

            # Strides for foreground anchors
            strides_all_anchors_img = torch.cat(
                [torch.full((H_lvl * W_lvl,), float(s_lvl_val), device=device)
                 for H_lvl, W_lvl, s_lvl_val in fmap_shapes], dim=0
            ) 
            strides_fg_img = strides_all_anchors_img[fg_mask_img].unsqueeze(-1) 

            # DFL Targets
            # Clamp target offsets for DFL: divide by stride, then clamp to [0, reg_max - eps]
            target_offsets_for_dfl = (box_targets_fg_img / strides_fg_img).clamp(min=0., max=head_reg_max_for_loss - 1e-6)
            dfl_target_dist = build_dfl_targets(target_offsets_for_dfl, head_reg_max_for_loss) 
            loss_dfl = dfl_loss(reg_p_fg_img, dfl_target_dist)

            # IoU Loss (using predicted boxes from DFL output)
            pred_ltrb_offsets_fg_img = PicoDetHead.dfl_decode_for_training(
                reg_p_fg_img, 
                dfl_project_buffer_for_decode.to(reg_p_fg_img.device),
                head_reg_max_for_loss
            ) # (num_fg_img, 4) - these are l,t,r,b *offsets* (not scaled by stride yet)
            
            anchor_centers_all_img = torch.cat(
                [assigner.cache[(H_lvl, W_lvl, s_lvl_val, str(device))]
                 for H_lvl, W_lvl, s_lvl_val in fmap_shapes], dim=0
            ) 
            anchor_centers_fg_img = anchor_centers_all_img[fg_mask_img] 

            # Predicted boxes for foreground anchors: scale offsets by stride
            # strides_fg_img is (num_fg_img, 1), pred_ltrb_offsets_fg_img is (num_fg_img, 4)
            # We need to scale each of l,t,r,b by the corresponding stride.
            pred_ltrb_pixels_fg_img = pred_ltrb_offsets_fg_img * strides_fg_img # (num_fg_img, 4)

            pred_boxes_fg_img = torch.stack((
                anchor_centers_fg_img[:, 0] - pred_ltrb_pixels_fg_img[:, 0], # x1
                anchor_centers_fg_img[:, 1] - pred_ltrb_pixels_fg_img[:, 1], # y1
                anchor_centers_fg_img[:, 0] + pred_ltrb_pixels_fg_img[:, 2], # x2
                anchor_centers_fg_img[:, 1] + pred_ltrb_pixels_fg_img[:, 3]  # y2
            ), dim=1) 

            loss_iou = tvops.complete_box_iou_loss(pred_boxes_fg_img, box_targets_fg_img, reduction='sum') / num_fg_img

            w_iou = 2.0 
            current_sample_total_loss = loss_cls + loss_dfl + w_iou * loss_iou
            batch_total_loss += current_sample_total_loss
            num_samples_with_loss_in_batch += 1
            
        if num_samples_with_loss_in_batch > 0:
            averaged_batch_loss = batch_total_loss / num_samples_with_loss_in_batch
            opt.zero_grad(set_to_none=True)
            scaler.scale(averaged_batch_loss).backward()
            scaler.step(opt); scaler.update()
            tot_loss_accum += averaged_batch_loss.item() * num_samples_with_loss_in_batch
            total_samples_contributing_to_loss_epoch += num_samples_with_loss_in_batch
        if i % 50 == 0 and num_samples_with_loss_in_batch > 0: # Print batch loss
            # Make sure loss_cls, loss_dfl, loss_iou are defined even if num_fg_img was 0 for the last sample in batch
            # This is generally okay as they are overwritten each sample loop.
            # If num_fg_img is 0, that sample is skipped, so these would hold values from previous sample.
            # It's better to print averaged_batch_loss as it's more stable.
            print(f"E{epoch} {i:04d}/{len(loader)} loss {averaged_batch_loss.item():.3f} (batch avg)")


    if total_samples_contributing_to_loss_epoch > 0:
        avg_epoch_loss_per_sample = tot_loss_accum / total_samples_contributing_to_loss_epoch
        return avg_epoch_loss_per_sample
    else:
        print(f"E{epoch} No samples contributed to loss this epoch.")
        return 0.0



# ───────────────────── QAT helpers ───────────────────────────────
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping, QConfig,
    MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver # Add MovingAverageMinMaxObserver
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx


def qat_prepare(model: nn.Module, example: torch.Tensor):
    # backend = 'x86' # or 'fbgemm' which is often the default for x86, 'qnnpack' for arm
    backend = 'x86' # More explicit, often used for server-side QAT
    qmap = get_default_qat_qconfig_mapping(backend) # Use fbgemm or qnnpack

    # For weights: Per-channel quantization is good.
    weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric # or torch.per_channel_affine_float_qparams
    )

    # For activations: Per-tensor quantization is generally more robust and common.
    # MovingAverageMinMaxObserver is a per-tensor observer.
    activation_observer = MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine # Use per-tensor scheme
        # reduce_range=False # common default, set True for some mobile backends if needed
    )
    
    # Create a new QConfig with these choices
    # The default qconfig from get_default_qat_qconfig_mapping might already be good,
    # but if you want to be explicit:
    custom_qconfig = QConfig(activation=activation_observer, weight=weight_observer)
    
    # Apply this qconfig globally
    qmap = qmap.set_global(custom_qconfig) # This might be too broad if some ops need special handling

    # Let's refine the global qconfig:
    # Use per-channel for weights, per-tensor for activations.
    qconfig_global_refined = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
            # reduce_range=True # Sometimes recommended for QNNPACK
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric # Symmetric for weights is common
        )
    )
    qmap_new = get_default_qat_qconfig_mapping(backend) # Start fresh
    qmap_new = qmap_new.set_global(qconfig_global_refined)

    # Skip quantization for the 'pre' module
    qmap_new = qmap_new.set_module_name('pre', None)

    return prepare_qat_fx(model.cpu(), qmap_new, (example,))

def unwrap_dataset(ds):
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds


def apply_nms_and_padding_to_raw_outputs(
    raw_boxes_batch: torch.Tensor, # (B, Total_Anchors, 4)
    raw_scores_batch: torch.Tensor, # (B, Total_Anchors, NC)
    score_thresh: float,
    iou_thresh: float,
    max_detections: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # (B, max_det, 4), (B, max_det), (B, max_det)
    
    device = raw_boxes_batch.device
    batch_size = raw_boxes_batch.shape[0]

    final_boxes_list = []
    final_scores_list = []
    final_labels_list = []

    for b_idx in range(batch_size):
        boxes_img = raw_boxes_batch[b_idx]     # (Total_Anchors, 4)
        scores_img = raw_scores_batch[b_idx]   # (Total_Anchors, NC)

        conf_per_anchor, labels_per_anchor = torch.max(scores_img, dim=1)
        
        keep_by_score_mask = conf_per_anchor >= score_thresh
        
        boxes_above_thresh = boxes_img[keep_by_score_mask]
        scores_above_thresh = conf_per_anchor[keep_by_score_mask]
        labels_above_thresh = labels_per_anchor[keep_by_score_mask]

        if boxes_above_thresh.numel() == 0:
            # Pad and append if no boxes pass threshold
            padded_boxes = torch.zeros((max_detections, 4), dtype=boxes_img.dtype, device=device)
            padded_scores = torch.zeros((max_detections,), dtype=scores_img.dtype, device=device)
            padded_labels = torch.full((max_detections,), -1, dtype=torch.long, device=device)
        else:
            nms_keep_indices = tvops.nms(boxes_above_thresh, scores_above_thresh, iou_thresh)
            
            boxes_after_nms = boxes_above_thresh[nms_keep_indices[:max_detections]]
            scores_after_nms = scores_above_thresh[nms_keep_indices[:max_detections]]
            labels_after_nms = labels_above_thresh[nms_keep_indices[:max_detections]]
            
            num_current_dets = boxes_after_nms.shape[0]
            pad_size = max_detections - num_current_dets
            
            padded_boxes = F.pad(boxes_after_nms, (0, 0, 0, pad_size), mode='constant', value=0.0)
            padded_scores = F.pad(scores_after_nms, (0, pad_size), mode='constant', value=0.0)
            padded_labels = F.pad(labels_after_nms, (0, pad_size), mode='constant', value=-1) 

        final_boxes_list.append(padded_boxes)
        final_scores_list.append(padded_scores)
        final_labels_list.append(padded_labels)

    return torch.stack(final_boxes_list), torch.stack(final_scores_list), torch.stack(final_labels_list)


def apply_nms_and_padding_to_raw_outputs_with_debug( # Renamed
    raw_boxes_batch: torch.Tensor, # (B, Total_Anchors, 4)
    raw_scores_batch: torch.Tensor, # (B, Total_Anchors, NC)
    score_thresh: float,
    iou_thresh: float,
    max_detections: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]: # Added debug lists
    
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
            nms_keep_indices = tvops.nms(boxes_above_thresh, scores_above_thresh, iou_thresh)
            
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
    run_name: str = "N/A" # Optional: for logging if you have multiple eval runs
):
    model.eval()
    total_iou_sum = 0.
    num_images_with_gt = 0
    num_images_processed = 0
    total_gt_boxes_across_images = 0
    total_preds_before_nms_filter_across_images = 0 # Anchors above score_thresh (before NMS)
    total_preds_after_nms_filter_across_images = 0  # Detections after NMS & final filter

    print(f"\n--- quick_val_iou Start (Epoch: {epoch_num}, Run: {run_name}) ---")
    print(f"Params: score_thresh={score_thresh}, iou_thresh={iou_thresh}, max_detections={max_detections}")

    for batch_idx, (imgs_batch, tgts_batch) in enumerate(loader):
        # Model now outputs raw boxes and scores
        raw_pred_boxes_batch, raw_pred_scores_batch = model(imgs_batch.to(device))
        # raw_pred_boxes_batch: (B, Total_Anchors, 4)
        # raw_pred_scores_batch: (B, Total_Anchors, NC)

        # DEBUG: Print shapes of raw model outputs for the first batch
        if batch_idx == 0:
            print(f"[Debug Eval Batch 0] raw_pred_boxes_batch shape: {raw_pred_boxes_batch.shape}")
            print(f"[Debug Eval Batch 0] raw_pred_scores_batch shape: {raw_pred_scores_batch.shape}")

        # Apply NMS and padding to raw outputs
        # This function itself needs to use the passed score_thresh and iou_thresh
        # It returns padded outputs and also needs to give us info for debugging
        (pred_boxes_batch_padded,
         pred_scores_batch_padded,
         pred_labels_batch_padded,
         debug_num_preds_before_nms_batch, # List of counts per image in batch
         debug_num_preds_after_nms_batch   # List of counts per image in batch
         ) = apply_nms_and_padding_to_raw_outputs_with_debug( # MODIFIED FUNCTION CALL
                raw_pred_boxes_batch, raw_pred_scores_batch,
                score_thresh, iou_thresh, max_detections
            )
        
        total_preds_before_nms_filter_across_images += sum(debug_num_preds_before_nms_batch)

        for i in range(imgs_batch.size(0)): # Iterate through images in the batch
            num_images_processed += 1
            current_img_annots_raw = tgts_batch[i]
            gt_boxes_list = []
            for annot in current_img_annots_raw:
                if 'bbox' in annot:
                    x, y, w, h = annot['bbox']
                    gt_boxes_list.append([x, y, x + w, y + h])

            if not gt_boxes_list:
                if batch_idx == 0 and i < 2: # Log for first few images if no GT
                    print(f"[Debug Eval Img {num_images_processed-1}] No GT boxes. Score Thresh: {score_thresh}. Num preds after NMS for this image: {debug_num_preds_after_nms_batch[i]}")
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
            # actual_predicted_scores = predicted_scores_for_img_padded[:num_actual_dets_this_img]

            if batch_idx == 0 and i < 2: # Log for first few images with GT
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
                if batch_idx == 0 and i < 2:
                    print(f"  IoU matrix is empty for Img {num_images_processed-1} despite having preds and GTs.")
                continue
                
            # For each GT box, find the max IoU with any predicted box
            if iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
                max_iou_per_gt, _ = iou_matrix.max(dim=0) # Max IoU for each GT box
                image_avg_iou_for_matched_gts = max_iou_per_gt.sum().item() # Sum of best IoUs for GTs in this image
                total_iou_sum += image_avg_iou_for_matched_gts
                if batch_idx == 0 and i < 2:
                     print(f"  Img {num_images_processed-1}: Max IoUs per GT: {max_iou_per_gt.cpu().numpy().round(3)}. Sum for image: {image_avg_iou_for_matched_gts:.3f}")
            # If iou_matrix.shape[0] == 0 (no preds but GTs exist), this block is skipped,
            # and image_avg_iou_for_matched_gts is effectively 0 for this image's contribution.
            # This is handled by the actual_predicted_boxes.numel() == 0 check earlier.

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
        scores_level = torch.sigmoid(cls_logit_perm + obj_logit_perm) # Shape (B, num_anchors_level, NC)

        return boxes_xyxy_level, scores_level

    def forward(self, raw_model_outputs_nested_tuple: Tuple[Tuple[torch.Tensor, ...], ...]):
        # raw_model_outputs_nested_tuple is now expected to be the 4-element tuple,
        # where the first 3 elements are themselves tuples of 3 tensors each.
        # ( (cls_l0, cls_l1, cls_l2),
        #   (obj_l0, obj_l1, obj_l2),
        #   (reg_l0, reg_l1, reg_l2),
        #   (str_l0, str_l1, str_l2)  <- strides, might not be needed if using self.strides_buffer
        # )

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
                 quantized_core_model: nn.Module, # This is int8_model_with_preprocessor
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
        raw_boxes: str = "raw_boxes",
        raw_scores: str = "raw_scores",
):
    """
    Turns an INT8 PicoDet graph that ends with raw anchor boxes / scores into
    one that emits post-NMS detections:

        det_boxes  [N,4]   det_scores [N]
        class_idx  [N]     batch_idx  [N]
    """
    m = onnx.load(in_path)
    g = m.graph

    # ──────────────── constants ────────────────
    g.initializer.extend([
        oh.make_tensor("iou_th",   TP.FLOAT, [], [iou_thresh]),
        oh.make_tensor("score_th", TP.FLOAT, [], [score_thresh]),
        oh.make_tensor("max_det",  TP.INT64, [], [max_det]),
        oh.make_tensor("axis1",    TP.INT64, [1], [1]),       # for Squeeze ops
        oh.make_tensor("split111", TP.INT64, [3], [1, 1, 1]), # Split(N,3)
        # reshape helpers
        oh.make_tensor("shape_boxes",  TP.INT64, [3], [0, -1, 4]),   # [B, A, 4]
        oh.make_tensor("shape_scores", TP.INT64, [3], [0, 0, -1]),   # keep B,A  flatten rest
    ])

    # ──────────────── BOXES  [B,?,4] ────────────────
    boxes3d = "boxes3d"
    g.node.append(oh.make_node(        # [B,*,4]
        "Reshape", [raw_boxes, "shape_boxes"], [boxes3d],
        name="Reshape_Boxes"))

    # ──────────────── SCORES [B,A,C,(1)] → [B,C,A] ────────────────
    scores3d = "scores3d"
    g.node.append(oh.make_node(        # flattens any trailing dims
        "Reshape", [raw_scores, "shape_scores"], [scores3d],
        name="Reshape_Scores"))

    scores_bca = "scores_bca"
    g.node.append(oh.make_node(
        "Transpose", [scores3d], [scores_bca],
        perm=[0, 2, 1], name="Transpose_BCA"))  # [B,C,A]

    # ──────────────── Non-Max Suppression ────────────────
    sel = "selected_idx"                                   # [N,3]
    g.node.append(oh.make_node(
        "NonMaxSuppression",
        [boxes3d, scores_bca, "max_det", "iou_th", "score_th"],
        [sel], name="NMS"))

    # ───────── split batch / class / anchor ─────────
    b_col, c_col, a_col = "b_col", "c_col", "a_col"
    g.node.append(oh.make_node(
        "Split", [sel, "split111"], [b_col, c_col, a_col],
        axis=1, name="SplitIdx"))

    # squeeze to 1-D
    b_idx, cls_idx, anc_idx = "batch_idx", "class_idx", "anchor_idx"
    for src, dst in [(b_col, b_idx), (c_col, cls_idx), (a_col, anc_idx)]:
        g.node.append(oh.make_node(
            "Squeeze", [src, "axis1"], [dst], name=f"Squeeze_{dst}"))

    # ───────── GatherND helpers ─────────
    b_u, a_u, cls_u = b_idx + "_u", anc_idx + "_u", cls_idx + "_u"
    g.node.extend([
        oh.make_node("Unsqueeze", [b_idx,  "axis1"], [b_u],  name="UnsqB"),
        oh.make_node("Unsqueeze", [anc_idx,"axis1"], [a_u],  name="UnsqA"),
        oh.make_node("Unsqueeze", [cls_idx,"axis1"], [cls_u],name="UnsqC"),
    ])

    # boxes: GatherND(raw_boxes, [batch, anchor])
    idx_boxes = "idx_boxes"
    g.node.append(oh.make_node(
        "Concat", [b_u, a_u], [idx_boxes], axis=1,
        name="IdxBoxes"))
    det_boxes = "det_boxes"
    g.node.append(oh.make_node(
        "GatherND", [boxes3d, idx_boxes], [det_boxes],
        name="GatherBoxes"))

    # scores: GatherND(scores_bca, [batch, class, anchor])
    idx_scores = "idx_scores"
    g.node.append(oh.make_node(
        "Concat", [b_u, cls_u, a_u], [idx_scores], axis=1,
        name="IdxScores"))
    det_scores = "det_scores"
    g.node.append(oh.make_node(
        "GatherND", [scores_bca, idx_scores], [det_scores],
        name="GatherScores"))

    # ───────── declare final graph outputs ─────────
    del g.output[:]   # clear any existing outputs
    g.output.extend([
        oh.make_tensor_value_info(det_boxes,  TP.FLOAT, ['N', 4]),
        oh.make_tensor_value_info(det_scores, TP.FLOAT, ['N']),
        oh.make_tensor_value_info(cls_idx,    TP.INT64, ['N']),
        oh.make_tensor_value_info(b_idx,      TP.INT64, ['N']),
    ])

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"[SAVE] Final ONNX with NMS → {out_path}")


def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', choices=['mnv3', 'mnv4s', 'mnv4m'], default='mnv3')
    pa.add_argument('--epochs', type=int, default=12) 
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    pa.add_argument('--no_inplace_head_neck', action='store_true', help="Disable inplace activations in head/neck")
    cfg = pa.parse_args(argv)

    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(cfg.device)
    print(f'[INFO] device = {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, ckpt=None, img_size=IMG_SIZE) # Pass img_size
    model = PicoDet(
        backbone, 
        feat_chs,
        num_classes=80, 
        neck_out_ch=96,
        img_size=IMG_SIZE,
        inplace_act_for_head_neck=not cfg.no_inplace_head_neck # Control from arg
    ).to(dev)
    
    tr_loader = get_loader(cfg.coco_root, 'train', cfg.batch, cfg.workers, subset_size=70000)
    vl_loader = get_loader(cfg.coco_root, 'val', cfg.batch, cfg.workers, subset_size=2000)

    train_base_ds = unwrap_dataset(tr_loader.dataset)
    if hasattr(train_base_ds, 'coco') and train_base_ds.coco is not None:
        coco_label_map = create_coco_contiguous_label_map(train_base_ds.coco)

        if len(coco_label_map) != model.head.nc:
            print(f"[main WARNING] Number of categories in generated map ({len(coco_label_map)}) "
                  f"does not match model.head.nc ({model.head.nc}). "
                  "Annotations will be filtered in train_epoch if mapped label >= model.head.nc.")
    else:
        raise RuntimeError("Could not access COCO API to create label map. "
                           "Ensure CocoDetection dataset is used and initialized correctly.")

    if False:
        head_biases = [p for n,p in model.named_parameters() if ".bias" in n]
        others      = [p for n,p in model.named_parameters() if ".bias" not in n]
        opt = AdamW([{'params': head_biases, 'weight_decay':0.0},
                     {'params': others     , 'weight_decay':1e-4}],
                    lr=4e-4)
        warm, total = 500, cfg.epochs * len(tr_loader)        # 500 iters warm-up
        def lr_lambda(step):
            if step < warm:
                return step / warm                     # linear warm-up
            progress = (step - warm) / (total - warm)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        opt = SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-5)
        sch = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0005)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    
    # Assigner.nc should match model.head.nc, which is the number of *mapped* classes the model predicts.
    assigner = SimOTACache(
        nc=model.head.nc,  # Number of classes from the model's head
        ctr=2.5,           # Center radius factor (default from new SimOTA)
        topk=10,           # Max candidates per GT (default from new SimOTA)
        cls_cost_weight=0.5 # Small classification cost weight for multi-class (default from new SimOTA)
    )


    original_model_head_nc = model.head.nc
    original_model_head_reg_max = model.head.reg_max
    # Fetch the dfl_project_buffer from the original model's head
    original_dfl_project_buffer = model.head.dfl_project_buffer


    # ... (FP32 training loop) ...
    for ep in range(cfg.epochs):
        BACKBONE_FREEZE_EPOCHS = 2  # 0 to disable
        if ep == 0:
            for p in model.backbone.parameters():
                p.requires_grad = False
            print(f"[INFO] Backbone frozen for {BACKBONE_FREEZE_EPOCHS} epochs…")
        
        if ep == BACKBONE_FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("[INFO] Backbone unfrozen – full network now training")

        model.train()
        l = train_epoch(
            model, tr_loader, opt, scaler, assigner, dev, ep, coco_label_map,
            head_nc_for_loss=original_model_head_nc,
            head_reg_max_for_loss=original_model_head_reg_max,
            dfl_project_buffer_for_decode=original_dfl_project_buffer,
            max_epochs=cfg.epochs, # Pass total epochs for VFL alpha scheduling
            quality_floor_vfl=0.2  # Example, tune as needed
        )
        # ... (validation) ...
        model.eval() 
        m = quick_val_iou(
            model, vl_loader, dev,
            score_thresh=model.head.score_th,
            iou_thresh=model.head.iou_th,
            max_detections=model.head.max_det,
            epoch_num=ep, run_name="in epoch",
        )
        sch.step()
        print(f'Epoch {ep + 1}/{cfg.epochs}  loss {l:.3f}  IoU {m:.3f}')

    print("[INFO] Evaluating FP32 model...")
    model.eval()
    try:
        iou_05 = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.05, # Specifically pass 0.05
                               iou_thresh=model.head.iou_th, # Or your chosen NMS IoU
                               max_detections=model.head.max_det,
                               epoch_num=ep, # If in training loop
                               run_name="score_thresh_0.05")
        print(f"[INFO] Validation IoU (score_th=0.05): {iou_05:.4f}")
        
        # Run for score_thresh = 0.25
        iou_25 = quick_val_iou(model, vl_loader, dev,
                               score_thresh=0.25, # Specifically pass 0.25
                               iou_thresh=model.head.iou_th,
                               max_detections=model.head.max_det,
                               epoch_num=ep,
                               run_name="score_thresh_0.25")
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
    print("[INFO] QAT model prepared and moved to device.")

    # ... (QAT Finetuning) ...
    qat_model.train()
    opt_q_params = filter(lambda p: p.requires_grad, qat_model.parameters())
    opt_q = SGD(opt_q_params, lr=0.002, momentum=0.9) # Potentially adjust LR for QAT
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))

    print("[INFO] Starting QAT finetuning epochs...")
    qat_epochs = 2  # cfg.epochs // 2 or 1
    for qep in range(qat_epochs): # Example: finetune for half the FP32 epochs or at least 1
        lq = train_epoch(
            qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, coco_label_map,
            head_nc_for_loss=original_model_head_nc,
            head_reg_max_for_loss=original_model_head_reg_max,
            dfl_project_buffer_for_decode=original_dfl_project_buffer,
            max_epochs=qat_epochs, # Pass total QAT epochs for VFL alpha scheduling
            quality_floor_vfl=0.2  # Consistent VFL floor
        )
        if lq is not None : # Check if loss was computed
            print(f'[QAT] Epoch {qep + 1}/{(cfg.epochs // 2 or 1)}  loss {lq:.3f}')
        else:
            print(f'[QAT] Epoch {qep + 1}/{(cfg.epochs // 2 or 1)}  loss N/A (no samples contributed)')

    print("[INFO] Evaluating QAT model...")
    qat_model.eval()
    try:
        qat_iou_high_thresh = quick_val_iou(qat_model, vl_loader, dev,
                               score_thresh=0.25, # Specifically pass 0.25
                               iou_thresh=model.head.iou_th,
                               max_detections=model.head.max_det,
                               epoch_num=qep,
                               run_name="score_thresh_0.25")
        print(f"[INFO] Validation IoU (score_th=0.25): {qat_iou_high_thresh:.4f}")
    except Exception as e:
        print(repr(e))
    qat_model.train()

    # --- Convert QAT model to INT8 ---
    qat_model.cpu().eval()
    int8_model_with_preprocessor = convert_fx(qat_model) # This model still contains 'pre'
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
    final_exportable_int8_model.cpu().eval() # Ensure CPU and eval mode for export

    # ---------------- ONNX export (model WITH preprocessor AND wrapped postprocessor, without NMS) ----------------
    temp_onnx_path = cfg.out.replace(".onnx", "_temp_no_nms.onnx")

    actual_onnx_input_example = dummy_uint8_input_cpu.cpu() # uint8 tensor

    print("[INFO] Exporting final INT8 model with wrapped postprocessing to ONNX...")
    torch.onnx.export(
        final_exportable_int8_model, # <--- Use this new composite model
        actual_onnx_input_example,
        temp_onnx_path,
        input_names=['images_uint8'],
        # These output names now correspond to the (B, TotalAnchors, 4/NC) tensors
        output_names=['raw_boxes', 'raw_scores'],
        dynamic_axes={
            'images_uint8': {0: 'batch', 2: 'h', 3: 'w'},
            'raw_boxes':    {0: 'batch', 1: 'anchors'}, # Shape [batch, total_anchors, 4]
            'raw_scores':   {0: 'batch', 1: 'anchors'}  # Shape [batch, total_anchors, num_classes]
                                                        # The 3rd dim (num_classes) is static
        },
        opset_version=18, # Your script uses 18
        keep_initializers_as_inputs=False, # Important for some runtimes
        do_constant_folding=True,
    )
    print(f'[SAVE] Intermediate ONNX (no NMS, with wrapped postprocessor) → {temp_onnx_path}')

    # DEBUG: Inspect intermediate ONNX model outputs
    intermediate_model_check = onnx.load(temp_onnx_path)
    print("[DEBUG] Intermediate ONNX model input ValueInfo:")
    for input_vi in intermediate_model_check.graph.input:
        # ... (same detailed print logic as for outputs) ...
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
        # Ensure model is on CPU for this test if actual_onnx_input_example is on CPU
        int8_model_with_preprocessor.cpu().eval() # Ensure CPU and eval
        actual_onnx_input_example_cpu = actual_onnx_input_example.cpu()

        py_outputs = int8_model_with_preprocessor(actual_onnx_input_example_cpu) # Call it

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
    int8_model_with_preprocessor.cpu().eval() # Ensure it's on CPU and eval
    dummy_input_for_inspection = actual_onnx_input_example.cpu() # Or generate a new one
    
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
        import traceback
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
        import traceback
        traceback.print_exc()

    # ---------------- Append NMS to the ONNX model ------------------------------
    append_nms_to_onnx(
        in_path=temp_onnx_path,
        out_path=cfg.out,
        score_thresh=float(model.head.score_th), # NMS params from original head
        iou_thresh=float(model.head.iou_th),
        max_det=int(model.head.max_det),
    )


if __name__ == '__main__':
    main()
