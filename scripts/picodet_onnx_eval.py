#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
picodet_onnx_eval_revised.py
───────────────────────────────────────────────────────────────────────────────
Evaluate `picodet_int8.onnx` exported by the training/QAT pipeline.
Revised for better alignment with training and robust image_id handling.
"""

from __future__ import annotations
import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.v2 as T_v2 # Ensure you have torchvision >= 0.13 for v2
import torchvision.transforms.functional as F_tv
from torchvision.datasets import CocoDetection
import torchvision.ops as tvops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ───────────────────────── constants ──────────────────────────
SEED = 42
IMG_SIZE = 256      # Target size for the model's internal ResizeNorm
VAL_BATCH_SIZE = 1  # RECOMMENDED: 1 for simplicity with variable input sizes
                    # If > 1, collate_fn needs padding for original image sizes.
NUM_WORKERS = 0
DEFAULT_MODEL = "picodet_int8.onnx" # Should be the one with NMS appended
RAND_EXAMPLES = 5
DTYPE_EXPECTED = np.uint8 # Model input dtype (before internal ResizeNorm)

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────── Custom Dataset Wrapper to include image_id ───────
class CocoDetectionWithImageId(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index) # target is list of annotation dicts
        image_id = self.ids[index] # Get the COCO image_id
        return img, target, image_id

# ───────────────────────── data loading ───────────────────────
def build_val_transform() -> T_v2.Compose | None:
    # NO RESIZE HERE. ONNX model's ResizeNorm handles it.
    # We only need to convert PIL to Tensor if the dataset doesn't.
    # CocoDetection returns PIL, so collate_fn will handle ToTensor.
    return None # Or T_v2.Compose([T_v2.ToImage()]) if you prefer transforms pipeline

def collate_fn_eval(batch: List[Tuple[Any, Any, int]]) -> Tuple[torch.Tensor, List[Any], List[int]]:
    imgs_pil, targets_batch, image_ids_batch = zip(*batch)

    # Convert PIL images to uint8 tensors (CHW)
    # Input to ONNX model is original H, W images.
    imgs_tensor_list = [F_tv.pil_to_tensor(img_pil) for img_pil in imgs_pil] # List of [C, H_orig, W_orig]

    if VAL_BATCH_SIZE == 1:
        imgs_stacked = torch.stack(imgs_tensor_list) # Shape: [1, C, H_orig, W_orig]
    else:
        # If VAL_BATCH_SIZE > 1, images must be padded to the same H_orig, W_orig within the batch
        # before stacking, IF the ONNX model expects a stacked tensor where H,W are fixed for the batch.
        # Given dynamic_axes on H,W for the input node, this implies the ONNX runtime *might*
        # handle a list of tensors or some internal batching. However, `session.run` typically
        # expects a single NumPy array for a given input name.
        # For robust evaluation with batch_size > 1 and varying original sizes,
        # you'd usually pad images to max_H, max_W in batch.
        # For simplicity here, we strongly recommend VAL_BATCH_SIZE = 1.
        # If you insist on VAL_BATCH_SIZE > 1, this part needs careful implementation of padding.
        # Assuming VAL_BATCH_SIZE = 1, so this else block is not strictly needed with that setting.
        max_h = max(img.shape[1] for img in imgs_tensor_list)
        max_w = max(img.shape[2] for img in imgs_tensor_list)
        padded_imgs = []
        for img in imgs_tensor_list:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            # F_tv.pad takes [..., H, W] and padding is (left, top, right, bottom)
            padded_img = F_tv.pad(img, [0, 0, pad_w, pad_h], fill=0)
            padded_imgs.append(padded_img)
        imgs_stacked = torch.stack(padded_imgs)

    return imgs_stacked.contiguous(), list(targets_batch), list(image_ids_batch)


def get_loader(coco_root: str, subset_size: int | None, coco_api_instance: COCO) -> DataLoader:
    # Use the custom dataset
    val_ds_full = CocoDetectionWithImageId(
        root=f"{coco_root}/val2017",
        annFile=f"{coco_root}/annotations/instances_val2017.json",
        transform=build_val_transform() # PIL images, no resize yet
    )

    if subset_size and subset_size < len(val_ds_full):
        # Ensure subset indices are valid for val_ds_full.ids
        all_indices = list(range(len(val_ds_full)))
        chosen_indices = random.sample(all_indices, subset_size)
        val_ds = Subset(val_ds_full, chosen_indices)
        print(f"[INFO] Using a subset of {len(val_ds)} validation images.")
    else:
        val_ds = val_ds_full
        print(f"[INFO] Using the full validation set of {len(val_ds)} images.")

    return DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, # No shuffle for repeatable eval
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn_eval, persistent_workers=bool(NUM_WORKERS)
    )

# ───────────────────────── coco helpers ───────────────────────
def get_coco_maps(coco_api_instance: COCO) -> Tuple[Dict[int, int], Dict[int, str]]:
    """ Returns coco_category_id_to_contiguous_id, contiguous_id_to_name """
    cat_ids = sorted(coco_api_instance.getCatIds())
    coco_cat_id_to_cont_id = {coco_id: i for i, coco_id in enumerate(cat_ids)}
    cont_id_to_name = {i: coco_api_instance.loadCats([coco_id])[0]["name"] for coco_id, i in coco_cat_id_to_cont_id.items()}
    # Check if num_classes implied by map (len) matches model expectation (e.g. 80)
    # This is important if the model was trained with a fixed 80-class COCO.
    # The current map includes ALL categories in val2017.json.
    # If your model predicts 0-79 for the standard 80 COCO, ensure this map aligns.
    # For now, this generic map is used.
    return coco_cat_id_to_cont_id, cont_id_to_name

# ───────────────────────── plotting ───────────────────────────
def draw_image(img_uint8_chw: torch.Tensor, # Expects [C, H, W]
               pred_boxes_xyxy: np.ndarray, # At original image scale
               pred_labels_contiguous: np.ndarray,
               pred_scores: np.ndarray,
               contig_id_to_name_map: Dict[int, str],
               score_th: float = 0.25,
               title: str = ""):
    """Draw predicted boxes on image using matplotlib."""
    # Permute CHW to HWC for imshow
    img_hwc_np = img_uint8_chw.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120) # Adjusted size/dpi
    ax.imshow(img_hwc_np)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    for box, label_contig, score in zip(pred_boxes_xyxy, pred_labels_contiguous, pred_scores):
        if score < score_th:
            continue
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Clip boxes to image boundaries (important if predictions can go out of bounds)
        h_img, w_img = img_hwc_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w == 0 or h == 0: continue

        color = plt.cm.get_cmap("tab20", len(contig_id_to_name_map))(label_contig % 20) # Cycle through colors
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        class_name = contig_id_to_name_map.get(label_contig, f"ID:{label_contig}")
        ax.text(x1, y1 - 5, f"{class_name}:{score:.2f}", color='white', fontsize=7,
                bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor='none'))
    return fig

# ───────────────────────── evaluation core ─────────────────────
@torch.no_grad()
def evaluate(session: ort.InferenceSession,
             loader: DataLoader, # DataLoader yields (stacked_orig_imgs_uint8, targets_list, image_ids_list)
             coco_api: COCO,
             coco_cat_id_to_cont_id: Dict[int, int] # Map from COCO cat ID to 0..N-1
             ) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (mean_IoU_at_IMG_SIZE, mean_time_per_image_seconds, coco_eval_results_dict)
    """
    inp_name = session.get_inputs()[0].name
    # Expected output names from ONNX model with NMS appended
    out_names_expected = ["det_boxes", "det_scores", "class_idx", "batch_idx"]
    actual_out_names = [o.name for o in session.get_outputs()]
    
    # Verify output names match
    if not all(name in actual_out_names for name in out_names_expected):
        raise ValueError(f"ONNX model output names mismatch. Expected: {out_names_expected}, Got: {actual_out_names}")


    io_binding_supported = hasattr(session, "io_binding") and \
                           session.get_providers()[0] not in ["CPUExecutionProvider", "OpenVINOExecutionProvider"] # OpenVINO EP doesn't always play well with bind_external_data

    total_iou_sum_img_size_scale = 0.0
    num_images_with_gt_for_iou = 0
    total_inference_time = 0.0
    num_images_processed = 0

    coco_formated_detections = [] # For mAP calculation

    for batch_imgs_uint8, batch_targets, batch_coco_image_ids in tqdm(loader, desc="Evaluating", unit="batch"):
        # batch_imgs_uint8: Tensor [B, C, H_orig, W_orig] (uint8)
        # batch_targets: List of lists of annotation dicts
        # batch_coco_image_ids: List of COCO image_ids

        # Prepare input for ONNX
        # Input numpy array dtype should match DTYPE_EXPECTED (uint8)
        input_feed_np = batch_imgs_uint8.cpu().numpy()
        if input_feed_np.dtype != DTYPE_EXPECTED:
            input_feed_np = input_feed_np.astype(DTYPE_EXPECTED)

        start_time = time.perf_counter()
        if io_binding_supported:
            io_binding = session.io_binding()
            # Bind input: OrtValue from numpy
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_feed_np, device_type='cpu', device_id=0) # Assuming CPU for input prep before transfer if on GPU
            io_binding.bind_ortvalue_input(inp_name, input_ortvalue)

            for name in actual_out_names: # Bind all outputs
                io_binding.bind_output(name)
            session.run_with_iobinding(io_binding)
            ort_outputs_list = io_binding.copy_outputs_to_cpu() # List of numpy arrays
        else:
            ort_outputs_list = session.run(None, {inp_name: input_feed_np})
        total_inference_time += (time.perf_counter() - start_time)
        num_images_processed += input_feed_np.shape[0]

        # Unpack ONNX outputs (order matters, based on out_names_expected)
        # These are for the *entire batch* after NMS.
        # pred_boxes_nms_batch: [TotalNumDetsInBatch, 4] (XYXY at IMG_SIZE scale)
        # pred_scores_nms_batch: [TotalNumDetsInBatch]
        # pred_labels_contig_nms_batch: [TotalNumDetsInBatch] (contiguous 0..N-1)
        # pred_batch_indices_nms: [TotalNumDetsInBatch] (indicates which image in batch each det belongs to)
        pred_boxes_nms_batch = ort_outputs_list[actual_out_names.index("det_boxes")]
        pred_scores_nms_batch = ort_outputs_list[actual_out_names.index("det_scores")]
        pred_labels_contig_nms_batch = ort_outputs_list[actual_out_names.index("class_idx")]
        pred_batch_indices_nms = ort_outputs_list[actual_out_names.index("batch_idx")]


        # Process each image in the batch
        for i in range(input_feed_np.shape[0]): # Iterate through images in the current batch
            # Original image dimensions (input to ONNX before its internal resize)
            # H_orig, W_orig = input_feed_np.shape[2], input_feed_np.shape[3] # If VAL_BATCH_SIZE > 1 and padded
            H_orig, W_orig = batch_imgs_uint8[i].shape[1], batch_imgs_uint8[i].shape[2]


            current_coco_image_id = batch_coco_image_ids[i]
            current_gt_annots = batch_targets[i] # List of annot dicts for this image

            # --- Mean IoU Calculation (at IMG_SIZE scale) ---
            # Filter predictions for the current image
            img_preds_mask = (pred_batch_indices_nms == i)
            if not img_preds_mask.any() and not current_gt_annots:
                continue # No preds, no GT for this image

            # Predicted boxes for this image (already at IMG_SIZE scale from ONNX output)
            # These are XYXY
            current_pred_boxes_img_size = pred_boxes_nms_batch[img_preds_mask]

            if current_gt_annots:
                gt_boxes_orig_xywh = [ann['bbox'] for ann in current_gt_annots]
                
                # Scale GT boxes to IMG_SIZE for IoU calculation
                # sx_orig_to_imgsize = IMG_SIZE / W_orig
                # sy_orig_to_imgsize = IMG_SIZE / H_orig
                # scaled_gt_boxes_img_size_xyxy = []
                # for x, y, w, h in gt_boxes_orig_xywh:
                #     scaled_gt_boxes_img_size_xyxy.append([
                #         x * sx_orig_to_imgsize, y * sy_orig_to_imgsize,
                #         (x + w) * sx_orig_to_imgsize, (y + h) * sy_orig_to_imgsize
                #     ])

                # More robust GT scaling: Use actual H_orig, W_orig from the input tensor dimensions
                # if the input to ONNX was indeed the original size.
                # (This is critical if padding was used for batch_size > 1)
                # W_orig_for_scaling = batch_imgs_uint8[i].shape[2] # Actual width fed to ONNX for this image
                # H_orig_for_scaling = batch_imgs_uint8[i].shape[1] # Actual height
                
                # Use W_orig, H_orig obtained from coco_api.loadImgs for canonical original size
                img_meta_info = coco_api.loadImgs([current_coco_image_id])[0]
                W_canonical_orig = img_meta_info['width']
                H_canonical_orig = img_meta_info['height']

                sx_canonical_orig_to_imgsize = IMG_SIZE / W_canonical_orig
                sy_canonical_orig_to_imgsize = IMG_SIZE / H_canonical_orig
                
                scaled_gt_boxes_img_size_xyxy = []
                for x, y, w, h in gt_boxes_orig_xywh:
                    scaled_gt_boxes_img_size_xyxy.append([
                        x * sx_canonical_orig_to_imgsize, y * sy_canonical_orig_to_imgsize,
                        (x + w) * sx_canonical_orig_to_imgsize, (y + h) * sy_canonical_orig_to_imgsize
                    ])


                if scaled_gt_boxes_img_size_xyxy and current_pred_boxes_img_size.shape[0] > 0:
                    gt_tensor_img_size = torch.tensor(scaled_gt_boxes_img_size_xyxy, dtype=torch.float32)
                    pred_tensor_img_size = torch.tensor(current_pred_boxes_img_size, dtype=torch.float32)
                    
                    iou_matrix = tvops.box_iou(pred_tensor_img_size, gt_tensor_img_size)
                    if iou_matrix.numel() > 0:
                        # For each GT, find max IoU with a prediction
                        max_iou_per_gt, _ = iou_matrix.max(dim=0) # Max over predictions for each GT
                        total_iou_sum_img_size_scale += max_iou_per_gt.sum().item() # Sum of these max IoUs
                        # num_images_with_gt_for_iou += 1 # Count images
                        num_images_with_gt_for_iou += len(scaled_gt_boxes_img_size_xyxy) # Count GT boxes
                elif not scaled_gt_boxes_img_size_xyxy and current_pred_boxes_img_size.shape[0] > 0:
                    pass # False positives, IoU metric here doesn't directly capture this
                elif scaled_gt_boxes_img_size_xyxy and current_pred_boxes_img_size.shape[0] == 0:
                    pass # False negatives / Misses

            # --- COCO Detection Formatting for mAP (predictions at original image scale) ---
            if img_preds_mask.any(): # If there are any predictions for this image
                img_meta_info = coco_api.loadImgs([current_coco_image_id])[0]
                W_canonical_orig = img_meta_info['width']
                H_canonical_orig = img_meta_info['height']

                # Scaling factors from IMG_SIZE (prediction space) to canonical original image space
                sx_imgsize_to_canonical_orig = W_canonical_orig / IMG_SIZE
                sy_imgsize_to_canonical_orig = H_canonical_orig / IMG_SIZE

                current_pred_scores = pred_scores_nms_batch[img_preds_mask]
                current_pred_labels_contig = pred_labels_contig_nms_batch[img_preds_mask]

                # Create reverse map: contiguous_id (0..N-1) to COCO category_id
                reverse_coco_cat_id_map = {v: k for k, v in coco_cat_id_to_cont_id.items()}

                for box_img_size, score, label_contig in zip(current_pred_boxes_img_size, current_pred_scores, current_pred_labels_contig):
                    x1_img, y1_img, x2_img, y2_img = box_img_size

                    # Scale to original image dimensions
                    x1_orig = x1_img * sx_imgsize_to_canonical_orig
                    y1_orig = y1_img * sy_imgsize_to_canonical_orig
                    x2_orig = x2_img * sx_imgsize_to_canonical_orig
                    y2_orig = y2_img * sy_imgsize_to_canonical_orig
                    
                    # Clip to original image boundaries
                    x1_orig = max(0, x1_orig)
                    y1_orig = max(0, y1_orig)
                    x2_orig = min(W_canonical_orig, x2_orig) # Use W_canonical_orig, not W_canonical_orig - 1 for width/height calc
                    y2_orig = min(H_canonical_orig, y2_orig)

                    width_orig = max(0, x2_orig - x1_orig)
                    height_orig = max(0, y2_orig - y1_orig)

                    if width_orig == 0 or height_orig == 0:
                        continue

                    coco_cat_id = reverse_coco_cat_id_map.get(int(label_contig), -1) # Default to -1 if not found
                    if coco_cat_id == -1:
                        warnings.warn(f"Contiguous label {label_contig} not found in reverse map!")
                        continue

                    coco_formated_detections.append({
                        "image_id": current_coco_image_id,
                        "category_id": coco_cat_id,
                        "bbox": [float(x1_orig), float(y1_orig), float(width_orig), float(height_orig)],
                        "score": float(score)
                    })

    # Calculate final metrics
    mean_iou_val = total_iou_sum_img_size_scale / num_images_with_gt_for_iou if num_images_with_gt_for_iou > 0 else 0.0
    mean_time_per_image = total_inference_time / num_images_processed if num_images_processed > 0 else 0.0

    # --- COCO mAP Calculation ---
    coco_dt = coco_api.loadRes(coco_formated_detections) if coco_formated_detections else coco_api.loadRes([]) # Handle empty dets
    coco_eval_obj = COCOeval(coco_api, coco_dt, 'bbox')
    coco_eval_obj.evaluate()
    coco_eval_obj.accumulate()
    coco_eval_obj.summarize()

    coco_results_dict = {
        "mAP_0.50:0.95_all_100": coco_eval_obj.stats[0],
        "mAP_0.50_all_100": coco_eval_obj.stats[1],
        "mAP_0.75_all_100": coco_eval_obj.stats[2],
        "mAP_small_100": coco_eval_obj.stats[3],
        "mAP_medium_100": coco_eval_obj.stats[4],
        "mAP_large_100": coco_eval_obj.stats[5],
        "AR_0.50:0.95_all_1": coco_eval_obj.stats[6],
        "AR_0.50:0.95_all_10": coco_eval_obj.stats[7],
        "AR_0.50:0.95_all_100": coco_eval_obj.stats[8],
        "AR_small_100": coco_eval_obj.stats[9],
        "AR_medium_100": coco_eval_obj.stats[10],
        "AR_large_100": coco_eval_obj.stats[11],
    }
    return mean_iou_val, mean_time_per_image, coco_results_dict

# ───────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate PicoDet ONNX model.")
    ap.add_argument("--onnx", default=DEFAULT_MODEL, help="Path to ONNX model file.")
    ap.add_argument("--coco_root", default=os.path.expanduser("~/datasets/coco"), help="Root directory of COCO dataset.")
    ap.add_argument("--provider", default=None, choices=["CUDAExecutionProvider", "CPUExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider"],
                    help="Specific ONNX Runtime execution provider.")
    ap.add_argument("--subset", type=int, default=800,
                    help="Number of random validation images to use (0 for all).")
    ap.add_argument("--plot", action="store_true", help="Show a few example predictions.")
    ap.add_argument("--batch_size", type=int, default=VAL_BATCH_SIZE, help="Evaluation batch size (1 recommended).")

    args = ap.parse_args()
    
    # Update global VAL_BATCH_SIZE if user provides it
    global VAL_BATCH_SIZE
    VAL_BATCH_SIZE = args.batch_size
    if VAL_BATCH_SIZE > 1:
        print(f"[WARNING] VAL_BATCH_SIZE is {VAL_BATCH_SIZE}. If original image sizes vary, ensure padding in collate_fn is robust or use VAL_BATCH_SIZE=1.")


    if not Path(args.onnx).exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if not Path(args.coco_root).exists():
        raise FileNotFoundError(f"COCO root directory not found: {args.coco_root}")

    # -- Initialize COCO API first --
    val_annot_file = f"{args.coco_root}/annotations/instances_val2017.json"
    if not Path(val_annot_file).exists():
        raise FileNotFoundError(f"COCO validation annotations not found: {val_annot_file}")
    coco_api_gt = COCO(val_annot_file)

    # -- data loader & label maps --
    loader = get_loader(args.coco_root, args.subset if args.subset > 0 else None, coco_api_gt)
    coco_cat_id_to_cont_id, cont_id_to_name = get_coco_maps(coco_api_gt)

    # -- ONNX Runtime session --
    providers = ort.get_available_providers()
    chosen_providers = []
    if args.provider:
        if args.provider in providers:
            chosen_providers = [args.provider]
        else:
            warnings.warn(f"Provider {args.provider} not available. Available: {providers}. Falling back.")
    
    if not chosen_providers: # Auto-pick
        if "CUDAExecutionProvider" in providers: chosen_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in providers: chosen_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else: chosen_providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = os.cpu_count() or 1 # Can sometimes be better to let ORT manage
    
    print(f"[INFO] Attempting to load ONNX model: {args.onnx} with providers: {chosen_providers}")
    session = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=chosen_providers)
    print(f"[INFO] ONNX model loaded. Effective Provider(s): {session.get_providers()}")
    print(f"  Input: {session.get_inputs()[0].name}, Shape: {session.get_inputs()[0].shape}, Type: {session.get_inputs()[0].type}")
    print(f"  Outputs: {[o.name for o in session.get_outputs()]}")


    # -- evaluate --
    mean_iou_metric, time_per_image_sec, coco_eval_stats = evaluate(session, loader, coco_api_gt, coco_cat_id_to_cont_id)
    
    print("\n───────── Custom Mean IoU Metric ─────────")
    print(f"Validation images processed for IoU : {len(loader.dataset)}") # Or more specific counter if some skipped
    print(f"Mean IoU (preds vs GT, at IMG_SIZE) : {mean_iou_metric*100:5.2f} %") # Clarify scale
    
    print("\n───────── Performance ─────────")
    print(f"Avg inference time / image        : {time_per_image_sec*1e3:6.2f} ms "
          f"({1.0/time_per_image_sec if time_per_image_sec > 0 else 0:7.2f} img/s)")

    print("\n─── COCO Official mAP Results ───")
    for stat_name, stat_val in coco_eval_stats.items():
        # Clean up stat_name for printing if needed (e.g., replace underscores)
        print(f"  {stat_name.replace('_', ' '):<35} = {stat_val:.4f}")
    print("──────────────────────────────────")

    # -- visualise a few predictions --
    if args.plot and RAND_EXAMPLES > 0:
        print("\n[INFO] Preparing example predictions for plotting...")
        # Need to re-iterate or get specific items from dataset for plotting
        # For simplicity, grab first few from a new non-shuffled loader if subset was used.
        plot_loader = DataLoader(
            loader.dataset, # Use the same (potentially subsetted) dataset
            batch_size=1, shuffle=False, # No shuffle, take first few
            num_workers=NUM_WORKERS, collate_fn=collate_fn_eval
        )
        
        num_plotted = 0
        for plot_imgs_uint8, _, plot_coco_image_ids in plot_loader:
            if num_plotted >= RAND_EXAMPLES:
                break

            input_plot_np = plot_imgs_uint8.cpu().numpy().astype(DTYPE_EXPECTED)
            
            # Run inference for this single image
            ort_outputs_plot = session.run(None, {session.get_inputs()[0].name: input_plot_np})
            
            boxes_plot = ort_outputs_plot[actual_out_names.index("det_boxes")]
            scores_plot = ort_outputs_plot[actual_out_names.index("det_scores")]
            labels_contig_plot = ort_outputs_plot[actual_out_names.index("class_idx")]
            # batch_indices_plot = ort_outputs_plot[actual_out_names.index("batch_idx")] # Should all be 0

            # Predictions are at IMG_SIZE. Scale them to original for plotting.
            current_plot_coco_id = plot_coco_image_ids[0]
            img_meta_plot = coco_api_gt.loadImgs([current_plot_coco_id])[0]
            W_orig_plot, H_orig_plot = img_meta_plot['width'], img_meta_plot['height']

            sx_plot = W_orig_plot / IMG_SIZE
            sy_plot = H_orig_plot / IMG_SIZE
            
            boxes_plot_orig_scale = []
            for b_img_size in boxes_plot: # Assuming all predictions are for batch_idx 0
                x1, y1, x2, y2 = b_img_size
                boxes_plot_orig_scale.append([x1 * sx_plot, y1 * sy_plot, x2 * sx_plot, y2 * sy_plot])
            
            if boxes_plot_orig_scale:
                boxes_plot_orig_scale_np = np.array(boxes_plot_orig_scale)
            else: # Handle case with no detections
                 boxes_plot_orig_scale_np = np.empty((0,4))


            # The image fed to drawing should be the original image, not the one from input_plot_np if VAL_BATCH_SIZE > 1 used padding.
            # Fetch original PIL for plotting
            original_pil_image, _ = loader.dataset.dataset[loader.dataset.indices[num_plotted]] if isinstance(loader.dataset, Subset) else loader.dataset.dataset[num_plotted]
            original_img_tensor_chw = F_tv.pil_to_tensor(original_pil_image)


            fig = draw_image(original_img_tensor_chw, # Original image tensor CHW
                             boxes_plot_orig_scale_np,
                             labels_contig_plot,
                             scores_plot,
                             cont_id_to_name, score_th=0.1,
                             title=f"Image ID: {current_plot_coco_id}")
            num_plotted += 1
        plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.v2")
    warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable") # From ONNX Runtime
    main()
