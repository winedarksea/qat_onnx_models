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
from onnx import helper as onnx_helper
from onnx import TensorProto as onnx_TensorProto

try: 
    from picodet_lib_v2 import (
        PicoDet, get_backbone, VarifocalLoss, dfl_loss,
        build_dfl_targets, ResizeNorm, PicoDetHead
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
        
        # head_nc and head_reg_max are now passed as arguments
        # No need for:
        # original_head_module = model.head
        # ...
        # head_nc = original_head_module.nc 
        # head_reg_max = original_head_module.reg_max

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


def qat_prepare_OLD(model: nn.Module, example: torch.Tensor):
    qmap = get_default_qat_qconfig_mapping('x86')
    wobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    aobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_channel_affine)
    qmap = qmap.set_global(QConfig(aobs, wobs))
    # skip preprocess from quant
    qmap = qmap.set_module_name('pre', None)
    return prepare_qat_fx(model.cpu(), qmap, (example,))

def qat_prepare(model: nn.Module, example: torch.Tensor):
    # backend = 'x86' # or 'fbgemm' which is often the default for x86
    backend = 'qnnpack' # More explicit, often used for server-side QAT
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

    # A more targeted approach if set_global is problematic:
    # Iterate through module types and set specific qconfigs.
    # For now, let's try correcting the global observers first.
    # If `get_default_qat_qconfig_mapping` for 'fbgemm' already uses per-tensor for activations,
    # you might not even need to manually set `aobs` and `wobs` globally like this
    # unless you specifically want `MovingAveragePerChannelMinMaxObserver` for weights.

    # Let's refine the global qconfig:
    # Use per-channel for weights, per-tensor for activations.
    qconfig_global_refined = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
            # reduce_range=True # SOmetimes recommended for QNNPACK
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric # Symmetric for weights is common
        )
    )
    qmap_new = get_default_qat_qconfig_mapping(backend) # Start fresh
    qmap_new = qmap_new.set_global(qconfig_global_refined)


    # Skip quantization for the 'pre' module (ResizeNorm)
    qmap_new = qmap_new.set_module_name('pre', None)

    # It's also common to skip quantization for specific operations if they cause issues
    # or are not well-suited for quantization, e.g., some types of F.interpolate if they are problematic.
    # For example, to skip a specific function:
    # qmap = qmap.set_object_type(F.interpolate, None) # If interpolate was an issue

    # Check if any other parts of PicoDetHead (like DFL projection buffers) are being observed
    # when they shouldn't be. Buffers are usually not quantized unless part of a quantizable op.
    # The error points to `_tensor_constant1` which is a *lifted attribute* in the QAT graph,
    # not directly a buffer from your original model.

    return prepare_qat_fx(model.cpu(), qmap_new, (example,))

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

# NMS and padding utility function (adapted from old _batch_nms_and_pad)
# This will be used by quick_val_iou
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
def quick_val_iou(model: PicoDet, loader, device, 
                  # Pass NMS params needed if model head no longer holds them directly for this
                  score_thresh: float, iou_thresh: float, max_detections: int):
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

def main(argv: List[str] | None = None):
    # ... (argparsing, device setup, get_backbone remain same) ...
    # Make sure to pass inplace_act_for_head_neck=False to PicoDet constructor
    # if you want to disable inplace operations in head and neck globally.
    # Example:
    # model = PicoDet(
    #     backbone, feat_chs, num_classes=80, img_size=IMG_SIZE,
    #     inplace_act_for_head_neck=False # <--- Add this
    # ).to(dev)
    
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', choices=['mnv3', 'mnv4s', 'mnv4m'], default='mnv3')
    pa.add_argument('--epochs', type=int, default=5) 
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    # Add argument to control inplace for head/neck for easier experimentation
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

    # ... (QAT preparation: model.train(), model.cpu(), qat_prepare(model, ...)) ...
    print("[INFO] Preparing model for QAT...")
    dummy_uint8_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8, device='cpu')
    preprocessor = ResizeNorm(size=(IMG_SIZE, IMG_SIZE))
    example_float_input_for_qat = preprocessor(dummy_uint8_input.float()).cpu()

    model.train() 
    model.cpu()    
    
    print("[INFO] Running qat_prepare...")
    qat_model = qat_prepare(model, example_float_input_for_qat) 
    qat_model = qat_model.to(dev)
    print("[INFO] QAT model prepared and moved to device.")

    # Debug inspection for _tensor_constant1 (if it reappears)
    # The constant might be specific to the inference path trace if PicoDetHead.forward has different ops
    if hasattr(qat_model, '_tensor_constant1'): # Check on qat_model directly
        print(f"qat_model._tensor_constant1 exists. Value: {getattr(qat_model, '_tensor_constant1')}")
    else:
        # Try to find it deeper if it's a submodule attribute
        found_constant = False
        for name, sub_module in qat_model.named_modules():
            if hasattr(sub_module, '_tensor_constant1'):
                print(f"Found _tensor_constant1 in submodule {name}: {getattr(sub_module, '_tensor_constant1')}")
                found_constant = True
                break
        if not found_constant:
            print("Inspecting qat_model for _tensor_constant1... NOT FOUND anywhere.")


    # --- QAT Finetuning ---
    qat_model.train()
    opt_q_params = filter(lambda p: p.requires_grad, qat_model.parameters())
    opt_q = SGD(opt_q_params, lr=0.002, momentum=0.9)
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))
    
    print("[INFO] Starting QAT finetuning epochs...")
    for qep in range(5):
        lq = train_epoch(qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, coco_label_map,
                        head_nc_for_loss=original_model_head_nc, # Use values from original model
                        head_reg_max_for_loss=original_model_head_reg_max, # Use values from original model
                        dfl_project_buffer_for_decode=original_dfl_project_buffer) # Pass it
        print(f'[QAT] Epoch {qep + 1}/5  loss {lq:.3f}')
    qat_model.cpu().eval()
    int8_model_no_nms = convert_fx(qat_model)

    # ---------------- ONNX export (model without NMS) ------------------------------
    int8_model_no_nms.eval()
    temp_onnx_path = cfg.out.replace(".onnx", "_temp_no_nms.onnx")

    torch.onnx.export(int8_model_no_nms, example_float_input_for_qat.cpu(), temp_onnx_path,
                      input_names=['images_normalized_float'],
                      output_names=['raw_boxes', 'raw_scores'], # From PicoDetHead's inference
                      dynamic_axes={'images_normalized_float': {0: 'batch_size', 2: 'height', 3: 'width'},
                                    'raw_boxes': {0: 'batch_size'},
                                    'raw_scores': {0: 'batch_size'}},
                      opset_version=17, # 11 min opset for NonMaxSuppression attributes as inputs
                      do_constant_folding=True)
    print(f'[SAVE] Intermediate ONNX (no NMS) → {temp_onnx_path}')

    # ---------------- Append NMS to the ONNX model ------------------------------
    onnx_model = onnx.load(temp_onnx_path)
    graph = onnx_model.graph

    # NMS parameters from the model's head (ensure these are what you want)
    score_thresh_val = float(model.head.score_th)
    iou_thresh_val = float(model.head.iou_th)
    # For ONNX NonMaxSuppression, max_output_boxes_per_class can be set.
    # If we want a total of model.head.max_det detections,
    # we can set max_output_boxes_per_class to model.head.max_det.
    # This means NMS might keep up to model.head.max_det *per class*.
    # A subsequent TopK operation across all classes would be needed for a strict global top-K.
    # For simplicity, we'll use this, and the output might have more than max_det boxes if from different classes.
    # Or, set it to a smaller number like ceil(max_det / num_classes) if that's desired.
    # Let's set it to model.head.max_det for now.
    max_output_boxes_per_class_val = int(model.head.max_det) 

    # Original outputs from the PyTorch model
    # Their names were 'raw_boxes', 'raw_scores' from the export step.
    # Find them in the graph to ensure correct naming if changed.
    onnx_raw_boxes_name = ""
    onnx_raw_scores_name = ""
    for output_node in graph.output:
        if output_node.name == 'raw_boxes':
            onnx_raw_boxes_name = output_node.name
        elif output_node.name == 'raw_scores':
            onnx_raw_scores_name = output_node.name
    
    if not onnx_raw_boxes_name or not onnx_raw_scores_name:
        raise RuntimeError("Could not find raw_boxes or raw_scores in the ONNX graph outputs.")

    # 1. Transpose scores: from (B, NumAnchors, NC) to (B, NC, NumAnchors)
    # Output of PyTorch model raw_scores: (batch_size, num_total_anchors, num_classes)
    # ONNX NMS expects scores: (batch_size, num_classes, num_boxes)
    scores_transposed_name = "scores_transposed_for_nms"
    transpose_node = onnx_helper.make_node(
        "Transpose",
        inputs=[onnx_raw_scores_name],
        outputs=[scores_transposed_name],
        perm=[0, 2, 1] # Batch, Class, Box
    )
    graph.node.append(transpose_node)

    # 2. Create constant tensors for NMS parameters
    # These need to be float32 for thresholds, int64 for max_output_boxes_per_class
    iou_threshold_const_name = "nms_iou_threshold"
    score_threshold_const_name = "nms_score_threshold"
    max_boxes_per_class_const_name = "nms_max_boxes_per_class"

    graph.initializer.append(onnx_helper.make_tensor(iou_threshold_const_name, onnx_TensorProto.FLOAT, [], [iou_thresh_val]))
    graph.initializer.append(onnx_helper.make_tensor(score_threshold_const_name, onnx_TensorProto.FLOAT, [], [score_thresh_val]))
    graph.initializer.append(onnx_helper.make_tensor(max_boxes_per_class_const_name, onnx_TensorProto.INT64, [], [max_output_boxes_per_class_val]))

    # 3. Add NonMaxSuppression node
    # Output selected_indices: [num_selected_indices, 3] where columns are [batch_index, class_index, box_index]
    selected_indices_output_name = "selected_nms_indices"
    nms_node = onnx_helper.make_node(
        "NonMaxSuppression",
        inputs=[onnx_raw_boxes_name, scores_transposed_name, max_boxes_per_class_const_name, iou_threshold_const_name, score_threshold_const_name],
        outputs=[selected_indices_output_name],
        center_point_box=0 # 0 for (x1,y1,x2,y2) or (y1,x1,y2,x2)
    )
    graph.node.append(nms_node)

    # 4. Gather the actual boxes, scores, and labels using selected_indices
    # selected_indices: [num_selected_total, 3]
    # Column 0: batch_index
    # Column 1: class_index
    # Column 2: box_index (original index within the num_anchors for that batch item)

    # Extract components from selected_indices
    # These will be 1D tensors of length num_selected_total
    # Need an 'axes' constant for Gather
    axes_col_const = onnx_helper.make_tensor("gather_axes_col", onnx_TensorProto.INT64, [], [1])
    graph.initializer.append(axes_col_const)
    
    batch_idx_flat_name = "nms_batch_indices_flat"
    class_idx_flat_name = "nms_class_indices_flat"
    box_idx_flat_name = "nms_box_indices_flat"

    # indices_0 = onnx_helper.make_tensor("indices_0", onnx_TensorProto.INT64, [], [0]) # For 0th col
    # indices_1 = onnx_helper.make_tensor("indices_1", onnx_TensorProto.INT64, [], [1]) # For 1st col
    # indices_2 = onnx_helper.make_tensor("indices_2", onnx_TensorProto.INT64, [], [2]) # For 2nd col
    # graph.initializer.extend([indices_0, indices_1, indices_2])
    
    # Slicing columns using Gather is verbose. Alternative: Split or multiple Gathers.
    # For simplicity, let's assume we can use Split if available and easy, or just Gather cols.
    # Let's use Gather to pick columns:
    idx_0_tensor = onnx_helper.make_tensor("idx_0_tensor", onnx_TensorProto.INT64, [], [0])
    idx_1_tensor = onnx_helper.make_tensor("idx_1_tensor", onnx_TensorProto.INT64, [], [1])
    idx_2_tensor = onnx_helper.make_tensor("idx_2_tensor", onnx_TensorProto.INT64, [], [2])
    graph.initializer.extend([idx_0_tensor, idx_1_tensor, idx_2_tensor])

    graph.node.append(onnx_helper.make_node("Gather", [selected_indices_output_name, idx_0_tensor], [batch_idx_flat_name], axis=1))
    graph.node.append(onnx_helper.make_node("Squeeze", [batch_idx_flat_name, axes_col_const], [batch_idx_flat_name])) # Remove dim 1

    graph.node.append(onnx_helper.make_node("Gather", [selected_indices_output_name, idx_1_tensor], [class_idx_flat_name], axis=1))
    graph.node.append(onnx_helper.make_node("Squeeze", [class_idx_flat_name, axes_col_const], [class_idx_flat_name]))

    graph.node.append(onnx_helper.make_node("Gather", [selected_indices_output_name, idx_2_tensor], [box_idx_flat_name], axis=1))
    graph.node.append(onnx_helper.make_node("Squeeze", [box_idx_flat_name, axes_col_const], [box_idx_flat_name]))
    
    # We need to gather boxes from onnx_raw_boxes_name [B, NumAnchors, 4]
    # using batch_idx_flat_name and box_idx_flat_name. This needs GatherND.
    # GatherND indices: [num_selected_total, 2] where cols are [batch_idx, box_idx]
    gathernd_indices_for_boxes_name = "gathernd_indices_for_boxes"
    batch_idx_unsqueeze_name = "batch_idx_unsqueeze_for_stack"
    box_idx_unsqueeze_name = "box_idx_unsqueeze_for_stack"
    graph.node.append(onnx_helper.make_node("Unsqueeze", [batch_idx_flat_name, axes_col_const], [batch_idx_unsqueeze_name]))
    graph.node.append(onnx_helper.make_node("Unsqueeze", [box_idx_flat_name, axes_col_const], [box_idx_unsqueeze_name]))
    graph.node.append(onnx_helper.make_node("Concat", [batch_idx_unsqueeze_name, box_idx_unsqueeze_name], [gathernd_indices_for_boxes_name], axis=1))
    
    gathered_boxes_name = "final_nms_boxes" # Shape: [num_selected_total, 4]
    graph.node.append(onnx_helper.make_node("GatherND", [onnx_raw_boxes_name, gathernd_indices_for_boxes_name], [gathered_boxes_name]))

    # Gather scores from onnx_raw_scores_name [B, NumAnchors, NC]
    # using batch_idx_flat_name, box_idx_flat_name, and class_idx_flat_name.
    # GatherND indices: [num_selected_total, 3] where cols are [batch_idx, box_idx, class_idx]
    class_idx_unsqueeze_name = "class_idx_unsqueeze_for_stack"
    graph.node.append(onnx_helper.make_node("Unsqueeze", [class_idx_flat_name, axes_col_const], [class_idx_unsqueeze_name]))
    gathernd_indices_for_scores_name = "gathernd_indices_for_scores"
    graph.node.append(onnx_helper.make_node("Concat", [batch_idx_unsqueeze_name, box_idx_unsqueeze_name, class_idx_unsqueeze_name], [gathernd_indices_for_scores_name], axis=1))

    gathered_scores_name = "final_nms_scores" # Shape: [num_selected_total]
    graph.node.append(onnx_helper.make_node("GatherND", [onnx_raw_scores_name, gathernd_indices_for_scores_name], [gathered_scores_name]))
    
    # Labels are simply class_idx_flat_name
    gathered_labels_name = "final_nms_labels"
    graph.node.append(onnx_helper.make_node("Identity", [class_idx_flat_name], [gathered_labels_name]))

    # 5. Update graph outputs
    graph.ClearField("output") # Remove raw_boxes and raw_scores
    graph.output.extend([
        onnx_helper.make_tensor_value_info(gathered_boxes_name, onnx_TensorProto.FLOAT, ["num_total_selected_detections", 4]),
        onnx_helper.make_tensor_value_info(gathered_scores_name, onnx_TensorProto.FLOAT, ["num_total_selected_detections"]),
        onnx_helper.make_tensor_value_info(gathered_labels_name, onnx_TensorProto.INT64, ["num_total_selected_detections"]),
        # Optionally also output batch_idx_flat_name if user needs to know which batch item each flat detection belongs to
        onnx_helper.make_tensor_value_info(batch_idx_flat_name, onnx_TensorProto.INT64, ["num_total_selected_detections"], name="final_nms_batch_indices") 
    ])
    
    # Clean up graph from old outputs if they are no longer needed by any node
    # This is usually handled automatically if they are not graph outputs.

    # Save the final ONNX model with NMS
    onnx.save(onnx_model, cfg.out)
    print(f'[SAVE] Final ONNX with NMS graph → {cfg.out}')
    
    # (Optional) Validate the final model
    onnx.checker.check_model(cfg.out)


if __name__ == '__main__':
    main()
