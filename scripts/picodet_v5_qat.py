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
IMG_SIZE = 160  # also subset for speed, 224, PicoDet’s anchors assume stride-divisible sizes. Divisible by 32

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

                centres.append(self.cache[key])
                strides_list.append(torch.full((H * W,), float(s_val), dtype=torch.float32, device=device))
            
            if not centres: # Should only happen if f_shapes is empty
                # print("[SimOTACache WARNING] centres list is empty (f_shapes likely empty). Returning empty tensors.")
                return torch.zeros(0, dtype=torch.bool, device=device), \
                       torch.zeros((0, self.nc), device=device), \
                       torch.zeros((0, 4), device=device)
            
            centres = torch.cat(centres, dim=0)
            strides = torch.cat(strides_list, dim=0)
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
            cls_cost_val = -1.0 # This might need to be changed
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
    sorted_cat_ids = sorted(cat_ids)
    
    contiguous_map = {coco_id: i for i, coco_id in enumerate(sorted_cat_ids)}
    
    # If the model is strictly for the 80 common COCO categories, you might want a fixed map:
    # For example:
    # coco_80_categories = [ # List of the 80 COCO category IDs in the desired order for your model
    #    1, 2, 3, ..., 90 # (actual IDs for the 80 classes)
    # ]
    # contiguous_map = {coco_id: i for i, coco_id in enumerate(coco_80_categories)}
    # This ensures that, e.g., COCO ID 1 always maps to 0, ID 2 to 1, etc., *if* they are in your list.

    print(f"[INFO] Created COCO contiguous label map. {len(contiguous_map)} categories mapped.")
    return contiguous_map

# ───────────────────── train / val loops ────────────────────────
def train_epoch(
        model: nn.Module, loader, opt, scaler, assigner: SimOTACache,
        device: torch.device, epoch: int, coco_label_map: dict,
        # Pass head_nc and head_reg_max explicitly
        head_nc_for_loss: int, 
        head_reg_max_for_loss: int,
        dfl_project_buffer_for_decode: torch.Tensor,
):
    model.train()
    _, tot_loss_accum = time.time(), 0.
    total_samples_contributing_to_loss_epoch = 0
    
    for i, (imgs, tgts_batch) in enumerate(loader):
        imgs = imgs.to(device)
        model_outputs = model(imgs) # qat_model or original model

        # Unpacking depends on whether it's the original model (for FP32) or QAT model (for QAT finetune)
        # Both should return 4 items in training mode after the PicoDetHead.forward change.
        if not model.training:
             raise RuntimeError("train_epoch called with model not in training mode.")
        if len(model_outputs) != 4:
            raise ValueError(
                f"Expected 4 outputs from model in training mode (preds_cls, preds_obj, preds_reg, strides), got {len(model_outputs)}"
            )
        cls_preds_levels, obj_preds_levels, reg_preds_levels, strides_per_level_tensors = model_outputs

        bs = imgs.size(0)
        # ... (fmap_shapes calculation using strides_per_level_tensors) ...
        fmap_shapes = []
        for lv in range(len(cls_preds_levels)):
            H, W = cls_preds_levels[lv].shape[2:]
            s = strides_per_level_tensors[lv].item()
            fmap_shapes.append((H, W, s))

        batch_total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_samples_with_loss_in_batch = 0

        for b_idx in range(bs):
            # ... (annotation processing using head_nc_for_loss) ...
            current_sample_annots_raw = tgts_batch[b_idx]
            processed_boxes, processed_labels = [], []
            for annot_item in current_sample_annots_raw:
                if not isinstance(annot_item, dict) or 'bbox' not in annot_item or 'category_id' not in annot_item:
                    continue
                original_coco_id = annot_item['category_id']
                mapped_label = coco_label_map.get(original_coco_id)
                if mapped_label is not None and mapped_label < head_nc_for_loss: # Use passed arg
                    # ... (append to processed_boxes, processed_labels)
                    processed_boxes.append([
                        annot_item['bbox'][0], annot_item['bbox'][1],
                        annot_item['bbox'][0] + annot_item['bbox'][2], annot_item['bbox'][1] + annot_item['bbox'][3]
                    ])
                    processed_labels.append(mapped_label)
            if not processed_labels: continue
            gt_boxes = torch.tensor(processed_boxes, dtype=torch.float32, device=device)
            gt_labels = torch.tensor(processed_labels, dtype=torch.int64, device=device)
            target_dict_for_assigner = {'boxes': gt_boxes, 'labels': gt_labels}
            
            # Assigner SimOTACache was initialized with model.head.nc from the *original* model.
            # This should be consistent with head_nc_for_loss if it also comes from original model.
            if assigner.nc != head_nc_for_loss:
                 warnings.warn(f"Warning: assigner.nc ({assigner.nc}) != head_nc_for_loss ({head_nc_for_loss}).")

            fg_mask, cls_targets, box_targets = assigner(fmap_shapes, device, target_dict_for_assigner)
            
            num_fg = fg_mask.sum().item()
            if num_fg == 0: continue

            # Loss calculations using head_nc_for_loss and head_reg_max_for_loss
            cls_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, head_nc_for_loss) for lvl in cls_preds_levels], dim=0)
            obj_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1) for lvl in obj_preds_levels], dim=0)
            reg_p = torch.cat([lvl[b_idx].permute(1, 2, 0).reshape(-1, 4 * (head_reg_max_for_loss + 1)) for lvl in reg_preds_levels], dim=0)
            # ... (cls_p_fg, obj_p_fg, reg_p_fg, cls_targets_fg, box_targets_fg) ...
            cls_p_fg = cls_p[fg_mask]
            obj_p_fg = obj_p[fg_mask]
            reg_p_fg = reg_p[fg_mask]
            cls_targets_fg = cls_targets[fg_mask]
            box_targets_fg = box_targets[fg_mask]
            
            joint_logits_fg = cls_p_fg + obj_p_fg.unsqueeze(-1)
            loss_cls = VarifocalLoss()(joint_logits_fg, cls_targets_fg)
            
            # ... (strides_all_anchors, strides_fg calculation) ...
            strides_tensor_list = []
            for H_level, W_level, s_level_val in fmap_shapes:
                num_anchors_this_level = H_level * W_level
                strides_tensor_list.append(torch.full((num_anchors_this_level,), float(s_level_val), device=device))
            strides_all_anchors = torch.cat(strides_tensor_list, dim=0)
            strides_fg = strides_all_anchors[fg_mask]

            target_offsets_for_dfl = (box_targets_fg / strides_fg.unsqueeze(-1)).clamp(min=0., max=head_reg_max_for_loss - 1e-6)
            dfl_target_dist = build_dfl_targets(target_offsets_for_dfl, head_reg_max_for_loss)
            loss_dfl = dfl_loss(reg_p_fg, dfl_target_dist)

            current_num_fg_for_iou = reg_p_fg.shape[0]
            if current_num_fg_for_iou == 0:
                loss_iou = torch.tensor(0.0, device=device)
            else:
                # Call the static method from PicoDetHead class directly
                # Pass the dfl_project_buffer and head_reg_max_for_loss explicitly
                pred_ltrb_offsets_fg = PicoDetHead.dfl_decode_for_training( # Static call
                    reg_p_fg, 
                    dfl_project_buffer_for_decode.to(reg_p_fg.device), # Ensure buffer is on correct device
                    head_reg_max_for_loss
                )
                
                # ... (rest of IoU loss calculation using pred_ltrb_offsets_fg) ...
                centres_all_anchors = torch.cat(
                    [assigner.cache[(H_lvl, W_lvl, s_lvl, str(device))] for H_lvl, W_lvl, s_lvl in fmap_shapes], 
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
            num_samples_with_loss_in_batch += 1
            
        # ... (optimizer step, loss accumulation, printing) ...
        if num_samples_with_loss_in_batch > 0:
            averaged_batch_loss = batch_total_loss / num_samples_with_loss_in_batch
            opt.zero_grad(set_to_none=True)
            scaler.scale(averaged_batch_loss).backward()
            scaler.step(opt); scaler.update()
            tot_loss_accum += averaged_batch_loss.item() * num_samples_with_loss_in_batch
            total_samples_contributing_to_loss_epoch += num_samples_with_loss_in_batch
        if i % 50 == 0 and num_samples_with_loss_in_batch > 0:
            print(f"E{epoch} {i:04d}/{len(loader)} loss {averaged_batch_loss.item():.3f} (batch avg)")

    if total_samples_contributing_to_loss_epoch > 0:
        avg_epoch_loss_per_sample = tot_loss_accum / total_samples_contributing_to_loss_epoch
        return avg_epoch_loss_per_sample
    else:
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


@torch.no_grad()
def quick_val_iou(
    model: PicoDet, loader, device,
    score_thresh: float, iou_thresh: float, max_detections: int
):
    model.eval()
    total_iou_sum = 0.
    num_images_with_gt = 0

    for imgs_batch, tgts_batch in loader:
        # Model now outputs raw boxes and scores
        raw_pred_boxes_batch, raw_pred_scores_batch = model(imgs_batch.to(device))
        
        # Apply NMS and padding to raw outputs
        pred_boxes_batch_padded, pred_scores_batch_padded, _ = \
            apply_nms_and_padding_to_raw_outputs(
                raw_pred_boxes_batch, raw_pred_scores_batch,
                score_thresh, iou_thresh, max_detections
            )

        for i in range(imgs_batch.size(0)):
            current_img_annots_raw = tgts_batch[i]
            gt_boxes_list = []
            for annot in current_img_annots_raw:
                if 'bbox' in annot:
                    x, y, w, h = annot['bbox']
                    gt_boxes_list.append([x, y, x + w, y + h])

            if not gt_boxes_list: continue
            gt_boxes_tensor = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
            
            predicted_boxes_for_img = pred_boxes_batch_padded[i] # Padded to max_detections
            predicted_scores_for_img = pred_scores_batch_padded[i] # Padded

            # Filter out padded predictions (score == 0 or label == -1)
            valid_preds_mask = predicted_scores_for_img > 0 # Assuming 0 score means padding
            actual_predicted_boxes = predicted_boxes_for_img[valid_preds_mask]

            if actual_predicted_boxes.numel() == 0:
                num_images_with_gt += 1
                continue

            iou_matrix = tvops.box_iou(actual_predicted_boxes, gt_boxes_tensor)
            if iou_matrix.numel() == 0:
                num_images_with_gt +=1
                continue
                
            if iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
                max_iou_per_gt, _ = iou_matrix.max(dim=0)
                image_avg_iou = max_iou_per_gt.mean().item()
                total_iou_sum += image_avg_iou
                num_images_with_gt += 1
            elif iou_matrix.shape[1] > 0: # GTs exist, but no valid preds matched (or no preds after filter)
                 num_images_with_gt +=1


    return total_iou_sum / num_images_with_gt if num_images_with_gt > 0 else 0.


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
        
        # If anchor_points were generated for a max H_feat, W_feat and current H_feat, W_feat are smaller
        # (e.g. due to dynamic input image size leading to smaller feature maps),
        # a more robust way to get the *correct subset* of anchor_points is needed.
        # However, PicoDetHead pre-calculates anchors based on a fixed img_size for training,
        # and QAT example uses this fixed img_size. So, H_feat, W_feat should match the
        # dimensions for which anchor_points_level_{level_idx} were created.

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
        # optional_strides_tuple = raw_model_outputs_nested_tuple[3] # If needed

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

            # Debugging shapes received by postprocessor
            # print(f"[PostprocessorForONNX Level {i}] cls_l shape: {cls_l.shape}, obj_l shape: {obj_l.shape}, reg_l shape: {reg_l.shape}")

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
    pa.add_argument('--epochs', type=int, default=25) 
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
    
    tr_loader = get_loader(cfg.coco_root, 'train', cfg.batch, cfg.workers, subset_size=12000)
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

    opt = SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-5)
    sch = CosineAnnealingLR(opt, cfg.epochs, 5e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    
    # Assigner.nc should match model.head.nc, which is the number of *mapped* classes the model predicts.
    assigner = SimOTACache(model.head.nc)


    original_model_head_nc = model.head.nc
    original_model_head_reg_max = model.head.reg_max
    # Fetch the dfl_project_buffer from the original model's head
    # It's a registered buffer, so it should be accessible.
    original_dfl_project_buffer = model.head.dfl_project_buffer


    # ... (FP32 training loop) ...
    for ep in range(cfg.epochs):
        model.train()
        l = train_epoch(model, tr_loader, opt, scaler, assigner, dev, ep, coco_label_map,
                        head_nc_for_loss=original_model_head_nc,
                        head_reg_max_for_loss=original_model_head_reg_max,
                        dfl_project_buffer_for_decode=original_dfl_project_buffer)
        # ... (validation) ...
        model.eval() 
        m = quick_val_iou(model, vl_loader, dev, 
                          model.head.score_th, model.head.iou_th, model.head.max_det)
        sch.step()
        print(f'Epoch {ep + 1}/{cfg.epochs}  loss {l:.3f}  IoU {m:.3f}')

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
    for qep in range(cfg.epochs // 2 or 1): # Example: finetune for half the FP32 epochs or at least 1
        lq = train_epoch(qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, coco_label_map,
                        head_nc_for_loss=original_model_head_nc,
                        head_reg_max_for_loss=original_model_head_reg_max,
                        dfl_project_buffer_for_decode=original_dfl_project_buffer)
        if lq is not None : # Check if loss was computed
            print(f'[QAT] Epoch {qep + 1}/{(cfg.epochs // 2 or 1)}  loss {lq:.3f}')
        else:
            print(f'[QAT] Epoch {qep + 1}/{(cfg.epochs // 2 or 1)}  loss N/A (no samples contributed)')

    # --- Convert QAT model to INT8 ---
    qat_model.cpu().eval()
    int8_model_with_preprocessor = convert_fx(qat_model) # This model still contains 'pre'
    print("[INFO] QAT model converted to INT8.")

    # ---------------- Create the wrapper for ONNX export ----------------
    # `model` here is your original FP32 PicoDet model, used to get head parameters
    # Ensure `model` is accessible here. If it was modified, load weights if necessary,
    # or ensure `model.head` has the correct buffers.
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
    # This should now work correctly as raw_boxes and raw_scores will have the expected shapes.
    append_nms_to_onnx(
        in_path=temp_onnx_path,
        out_path=cfg.out,
        score_thresh=float(model.head.score_th), # NMS params from original head
        iou_thresh=float(model.head.iou_th),
        max_det=int(model.head.max_det),
    )


if __name__ == '__main__':
    main()
