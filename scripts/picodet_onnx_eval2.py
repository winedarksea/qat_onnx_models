# eval_onnx_picodet.py
import argparse
import time
import json
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torchvision.datasets import CocoDetection as OriginalCocoDetection
from torchvision.tv_tensors import BoundingBoxes
import torchvision.ops as tvops
from torch.utils.data import DataLoader

import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO as PyCOCO_COCO # Renamed to avoid class name collision
from pycocotools.cocoeval import COCOeval

# --- Constants and helpers from train_picodet_qat.py (or adapted) ---
IMG_SIZE = 256 # Should be consistent with the trained model

CANONICAL_COCO80_IDS: list[int] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

CANONICAL_COCO80_MAP: dict[int, int] = {
    coco_id: i for i, coco_id in enumerate(CANONICAL_COCO80_IDS)
}

def get_contiguous_id_to_name(coco_api: PyCOCO_COCO) -> dict[int, str]:
    """Return {0-79 -> class-name} using the official 80-class list."""
    return {
        i: coco_api.loadCats([coco_id])[0]["name"]
        for i, coco_id in enumerate(CANONICAL_COCO80_IDS)
    }

# This function is used by CocoDetectionV2, adapted from the training script
def _coco_to_tvt_for_dataset(annots, lb_map, canvas_size_pil):
    boxes, labels = [], []
    W, H = canvas_size_pil # PIL size is (W,H)
    for a in annots:
        if a.get("iscrowd", 0):
            continue
        cid = a["category_id"]
        if cid not in lb_map:
            continue
        x, y, w, h = a["bbox"] # COCO XYWH
        boxes.append([x, y, x + w, y + h]) # Store as XYXY for BoundingBoxes
        labels.append(lb_map[cid])

    if not boxes:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
    
    bbx = BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(H, W))
    return {"boxes": bbx, "labels": labels_tensor}


class CocoDetectionV2(OriginalCocoDetection):
    """COCO dataset for torchvision-v2 transforms, modified for evaluation."""
    def __init__(self, img_dir, ann_file, lb_map, transforms=None):
        super().__init__(img_dir, ann_file)
        self.lb_map = lb_map
        self._tf = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # Use super() to call OriginalCocoDetection.__getitem__ to get PIL image and annotations
        img_pil, anns_list = super().__getitem__(idx)


        # Create a target dict that transforms can operate on, even if not used directly for eval
        target_dict = _coco_to_tvt_for_dataset(anns_list, self.lb_map, img_pil.size)

        if self._tf is not None:
            # Apply transforms to image and dummy target
            # We only need the transformed image tensor for the model
            img_tensor, _ = self._tf(img_pil, target_dict) 
        else:
            # Default minimal transformation if none provided
            img_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.uint8, scale=True)])(img_pil)
            
        return img_tensor, img_id


def build_transforms_eval(img_size_tuple):
    # Transforms for evaluation, consistent with training's validation transforms
    return T.Compose([
        T.ToImage(), # PIL -> tv_tensors.Image
        T.Resize(img_size_tuple, antialias=True), # Resize to fixed size (e.g., (IMG_SIZE, IMG_SIZE))
        T.ToDtype(torch.uint8, scale=True), # Convert to uint8, model expects this
    ])

def collate_fn_eval(batch):
    imgs, img_ids = zip(*batch)
    return torch.stack(imgs, 0), list(img_ids)

def plot_detections_on_ax(ax, image_pil, boxes_xyxy, labels_contiguous, scores, class_names_map, score_thresh=0.3):
    ax.imshow(image_pil)
    ax.axis('off')
    
    for box, label_cont, score in zip(boxes_xyxy, labels_contiguous, scores):
        if score < score_thresh:
            continue
        
        x1, y1, x2, y2 = box
        w_box, h_box = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        class_name = class_names_map.get(label_cont, f"ID:{label_cont}")
        ax.text(x1, y1 - 5, f"{class_name} {score:.2f}", color='black', fontsize=8,
                bbox=dict(facecolor='lime', alpha=0.7, pad=1, edgecolor='none'))

def evaluate_onnx_model(onnx_model_path: str, coco_root_dir: str, batch_size: int, num_workers: int, device_str: str,
                        score_thresh_plot: float = 0.3, score_thresh_basic_acc: float = 0.3, iou_thresh_basic_acc: float = 0.5,
                        num_plot_samples: int = 5):
    
    print(f"Loading ONNX model from: {onnx_model_path}")
    providers = ['CPUExecutionProvider']
    if device_str.lower() == 'cuda':
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            try:
                providers = [('CUDAExecutionProvider', {'device_id': str(torch.cuda.current_device())}), 'CPUExecutionProvider']
                print(f"Attempting to use ONNX Runtime with CUDAExecutionProvider on device {torch.cuda.current_device()}.")
            except Exception as e:
                print(f"Failed to set CUDAExecutionProvider options: {e}. Using default CUDA provider.")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("Warning: Requested CUDA but ONNX Runtime CUDAExecutionProvider not available. Falling back to CPU.")
    else:
        print("Using ONNX Runtime with CPUExecutionProvider.")
            
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    # Get output names directly from the session; these are the true names in the ONNX model
    output_names_from_session = [o.name for o in ort_session.get_outputs()]
    print(f"ONNX Model Loaded. Input: '{input_name}', Outputs from session: {output_names_from_session}")

    val_annot_file = Path(coco_root_dir) / "annotations" / "instances_val2017.json"
    val_img_dir = Path(coco_root_dir) / "val2017"

    if not val_annot_file.exists() or not val_img_dir.exists():
        print(f"COCO validation data not found at {coco_root_dir} (Annotation: {val_annot_file}, Images: {val_img_dir}). Exiting.")
        return

    print("Loading COCO validation dataset API...")
    coco_gt = PyCOCO_COCO(str(val_annot_file))
    class_names_map = get_contiguous_id_to_name(coco_gt) # Maps 0-79 to string names

    print("Setting up DataLoader...")
    val_transforms = build_transforms_eval((IMG_SIZE, IMG_SIZE))
    val_dataset = CocoDetectionV2(
        str(val_img_dir), str(val_annot_file), CANONICAL_COCO80_MAP, val_transforms
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn_eval, pin_memory=(device_str.lower() == 'cuda' and 'CUDAExecutionProvider' in providers[0])
    )
    print(f"Dataset loaded: {len(val_dataset)} images. Batch size: {batch_size}.")

    all_coco_results = []
    total_inference_time_ms = 0
    num_processed_images = 0
    total_gt_boxes_for_basic_acc = 0
    correctly_detected_gt_for_basic_acc = 0
    samples_for_plotting = []

    # Define the expected output names (must match the ONNX model's actual output tensor names)
    # Based on your training script's append_nms_to_onnx and the previous error log.
    # The ONNX model's outputs are:
    # 1. Detected boxes: 'det_boxes'
    # 2. Detected scores: 'det_scores'
    # 3. Detected class indices (contiguous 0-79): 'class_idx'
    # 4. Batch index for each detection: 'batch_idx'
    
    expected_output_node_names = {
        "boxes": "det_boxes",
        "scores": "det_scores",
        "labels": "class_idx", # THIS IS THE CRUCIAL NAME
        "batch_indices": "batch_idx"
    }
    
    # Verify that the session's output names contain our expected names
    for key, name in expected_output_node_names.items():
        if name not in output_names_from_session:
            raise ValueError(f"Expected output tensor '{name}' for '{key}' not found in model outputs: {output_names_from_session}")


    print("Starting evaluation loop...")
    for batch_idx, (batch_img_tensors, batch_img_ids) in enumerate(tqdm(val_loader, desc="Evaluating")):
        batch_imgs_np = batch_img_tensors.numpy() # ONNX Runtime expects numpy

        start_time = time.perf_counter()
        ort_inputs = {input_name: batch_imgs_np}
        # ort_session.run returns a list of outputs in the order defined by output_names_from_session
        raw_ort_outputs_list = ort_session.run(None, ort_inputs)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Create a dictionary mapping the session's output names to the numpy arrays
        output_map = {name: data for name, data in zip(output_names_from_session, raw_ort_outputs_list)}

        det_boxes_batch = output_map[expected_output_node_names["boxes"]]    # (N_total_dets_in_batch, 4) XYXY
        det_scores_batch = output_map[expected_output_node_names["scores"]]  # (N_total_dets_in_batch,)
        cls_idx_batch = output_map[expected_output_node_names["labels"]]     # (N_total_dets_in_batch,) contiguous 0-79
        batch_idx_onnx = output_map[expected_output_node_names["batch_indices"]] # (N_total_dets_in_batch,) image index in batch

        # debug
        if batch_idx == 0: # Only for the first batch
            print("\nDEBUG: First batch raw outputs from ONNX:")
            print(f"  det_boxes_batch shape: {det_boxes_batch.shape}")
            print(f"  det_scores_batch shape: {det_scores_batch.shape}")
            print(f"  cls_idx_batch shape: {cls_idx_batch.shape}")
            print(f"  batch_idx_onnx shape: {batch_idx_onnx.shape}")
        
            # Show some values if there are any detections in the batch
            if det_scores_batch.shape[0] > 0:
                print(f"  Sample det_scores_batch: {det_scores_batch[:20]}") # First 20 scores
                print(f"  Min/Max scores: {np.min(det_scores_batch) if det_scores_batch.size > 0 else 'N/A'}, {np.max(det_scores_batch) if det_scores_batch.size > 0 else 'N/A'}")
                print(f"  Sample cls_idx_batch: {cls_idx_batch[:20]}")
                print(f"  Sample batch_idx_onnx: {batch_idx_onnx[:20]}")
                # Find detections for the first image in the batch (i=0)
                first_img_mask_debug = (batch_idx_onnx == 0)
                print(f"  Detections for first image in batch (count: {np.sum(first_img_mask_debug)}):")
                if np.sum(first_img_mask_debug) > 0:
                    print(f"    Boxes: {det_boxes_batch[first_img_mask_debug][:5]}")
                    print(f"    Scores: {det_scores_batch[first_img_mask_debug][:5]}")
                    print(f"    Labels: {cls_idx_batch[first_img_mask_debug][:5]}")
            else:
                print("  NO DETECTIONS IN THE FIRST BATCH OUTPUT FROM ONNX.")

        total_inference_time_ms += inference_time_ms
        num_processed_images += batch_imgs_np.shape[0]

        for i in range(batch_imgs_np.shape[0]): # Process each image in the batch
            img_id = batch_img_ids[i]
            img_info = coco_gt.loadImgs(img_id)[0]
            orig_w, orig_h = img_info['width'], img_info['height']
            scale_x = orig_w / IMG_SIZE
            scale_y = orig_h / IMG_SIZE

            current_img_mask = (batch_idx_onnx == i)
            img_boxes_pred_imgsize = det_boxes_batch[current_img_mask] # XYXY in IMG_SIZE space
            img_scores_pred = det_scores_batch[current_img_mask]
            img_labels_pred_cont = cls_idx_batch[current_img_mask] # These are 0-79

            # 1. Store results for COCOeval (scaled to original image dimensions)
            for box_pred, score_pred, label_pred_cont in zip(img_boxes_pred_imgsize, img_scores_pred, img_labels_pred_cont):
                x1, y1, x2, y2 = box_pred
                scaled_x1 = np.clip(x1 * scale_x, 0, orig_w)
                scaled_y1 = np.clip(y1 * scale_y, 0, orig_h)
                scaled_x2 = np.clip(x2 * scale_x, 0, orig_w)
                scaled_y2 = np.clip(y2 * scale_y, 0, orig_h)
                
                coco_w, coco_h = max(0, scaled_x2 - scaled_x1), max(0, scaled_y2 - scaled_y1)
                coco_box_xywh = [scaled_x1, scaled_y1, coco_w, coco_h]

                original_coco_cat_id = CANONICAL_COCO80_IDS[int(label_pred_cont)] # Ensure label_pred_cont is int for list indexing
                all_coco_results.append({
                    "image_id": img_id, "category_id": original_coco_cat_id,
                    "bbox": [round(float(c), 3) for c in coco_box_xywh],
                    "score": round(float(score_pred), 5),
                })
            
            # 2. Basic Accuracy Calculation
            gt_anns_for_img = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
            current_img_gt_boxes_xyxy_orig = []
            current_img_gt_labels_cont = [] # These will be 0-79
            for ann in gt_anns_for_img:
                if ann.get("iscrowd", 0): continue
                if ann['category_id'] not in CANONICAL_COCO80_MAP: continue # Skip if not in the 80 classes
                x, y, w, h = ann['bbox'] # Original space, XYWH
                current_img_gt_boxes_xyxy_orig.append([x, y, x + w, y + h])
                current_img_gt_labels_cont.append(CANONICAL_COCO80_MAP[ann['category_id']])
            
            num_gt_this_image = len(current_img_gt_boxes_xyxy_orig)
            total_gt_boxes_for_basic_acc += num_gt_this_image

            if num_gt_this_image > 0 and len(img_boxes_pred_imgsize) > 0:
                pred_boxes_orig_xyxy_filtered = []
                pred_labels_cont_filtered = [] # These will be 0-79
                for box_p, score_p, label_p_cont in zip(img_boxes_pred_imgsize, img_scores_pred, img_labels_pred_cont):
                    if score_p < score_thresh_basic_acc: continue
                    x1p, y1p, x2p, y2p = box_p
                    pred_boxes_orig_xyxy_filtered.append([x1p*scale_x, y1p*scale_y, x2p*scale_x, y2p*scale_y])
                    pred_labels_cont_filtered.append(int(label_p_cont)) # Ensure int
                
                if len(pred_boxes_orig_xyxy_filtered) > 0:
                    gt_boxes_tensor = torch.tensor(current_img_gt_boxes_xyxy_orig, dtype=torch.float32)
                    pred_boxes_tensor = torch.tensor(pred_boxes_orig_xyxy_filtered, dtype=torch.float32)
                    
                    # Ensure tensors are on the same device if using GPU for tvops
                    # For ONNX CPU, this is fine. If ONNX on GPU, might need .to(device)
                    iou_matrix = tvops.box_iou(gt_boxes_tensor, pred_boxes_tensor) # (Num_GT, Num_Pred)

                    # Simpler basic accuracy: one detection per GT box if IoU and class match
                    gt_matched_flags = [False] * num_gt_this_image
                    for pred_idx in range(pred_boxes_tensor.shape[0]):
                        pred_label_cont = pred_labels_cont_filtered[pred_idx]
                        best_iou_for_this_pred = 0
                        best_gt_idx_for_this_pred = -1
                        for gt_idx in range(gt_boxes_tensor.shape[0]):
                            if current_img_gt_labels_cont[gt_idx] == pred_label_cont and not gt_matched_flags[gt_idx]:
                                current_iou = iou_matrix[gt_idx, pred_idx].item()
                                if current_iou > best_iou_for_this_pred:
                                    best_iou_for_this_pred = current_iou
                                    best_gt_idx_for_this_pred = gt_idx
                        
                        if best_gt_idx_for_this_pred != -1 and best_iou_for_this_pred >= iou_thresh_basic_acc:
                            correctly_detected_gt_for_basic_acc += 1
                            gt_matched_flags[best_gt_idx_for_this_pred] = True # Mark GT as matched
            
            # 3. Store samples for plotting
            if len(samples_for_plotting) < num_plot_samples:
                img_tensor_for_plot = batch_img_tensors[i] # This is uint8 tensor
                pil_image_for_plot = T.ToPILImage()(img_tensor_for_plot)
                samples_for_plotting.append({
                    "pil_image": pil_image_for_plot, "boxes_xyxy": img_boxes_pred_imgsize.copy(), # Copy numpy arrays
                    "labels_contiguous": img_labels_pred_cont.copy(), "scores": img_scores_pred.copy(),
                })

    print("Evaluation loop finished.")
    avg_inference_time_ms = total_inference_time_ms / num_processed_images if num_processed_images > 0 else 0
    print(f"\n--- Metrics ---")
    print(f"Average inference time per image: {avg_inference_time_ms:.2f} ms ({1000/avg_inference_time_ms if avg_inference_time_ms > 0 else 0:.2f} FPS)")

    if not all_coco_results:
        print("No detections made, skipping COCO PYEVAL.")
    else:
        print("\nCalculating COCO AP metrics...")
        # Create a temporary JSON file for COCOeval
        temp_json_path = "temp_coco_results.json"
        with open(temp_json_path, 'w') as f:
            json.dump(all_coco_results, f)
        
        coco_dt = coco_gt.loadRes(temp_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        Path(temp_json_path).unlink() # Clean up temp file

    basic_accuracy_perc = (correctly_detected_gt_for_basic_acc / total_gt_boxes_for_basic_acc * 100) if total_gt_boxes_for_basic_acc > 0 else 0
    print(f"\nBasic Accuracy (Recall-like @IoU={iou_thresh_basic_acc}, Score>{score_thresh_basic_acc}):")
    print(f"  {basic_accuracy_perc:.2f}% ({correctly_detected_gt_for_basic_acc}/{total_gt_boxes_for_basic_acc} GT boxes correctly detected)")

    print(f"\nDisplaying {len(samples_for_plotting)} sample output images (score_thresh_plot={score_thresh_plot})...")
    if samples_for_plotting:
        num_cols = min(len(samples_for_plotting), 5)
        fig_height = 4 
        fig_width = num_cols * 4
        
        # Adjust subplot creation for single sample case
        if num_cols == 1:
            fig, ax_single = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            axes = [ax_single] # Make it a list to be compatible with loop
        else:
            fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))
            axes = axes.flatten() # Ensure axes is always a flat list

        for i, sample in enumerate(samples_for_plotting):
            plot_detections_on_ax(
                axes[i], sample["pil_image"], sample["boxes_xyxy"],
                sample["labels_contiguous"], sample["scores"],
                class_names_map, score_thresh=score_thresh_plot
            )
            axes[i].set_title(f"Sample {i+1}")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PicoDet ONNX model on COCO validation dataset.")
    parser.add_argument("--onnx_model_path", type=str, default="picodet_int8.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--coco_root", type=str, default="coco", help="Root directory of the COCO dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation. Default 1 for more accurate per-image time.")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device for ONNX Runtime ('cpu' or 'cuda').")
    parser.add_argument("--score_thresh_plot", type=float, default=0.02, help="Score threshold for plotting detections.")
    parser.add_argument("--num_plot_samples", type=int, default=5, help="Number of sample images to plot.")
    parser.add_argument("--basic_acc_score_thresh", type=float, default=0.02, help="Score threshold for basic accuracy calculation.")
    parser.add_argument("--basic_acc_iou_thresh", type=float, default=0.1, help="IoU threshold for basic accuracy calculation.")


    args = parser.parse_args()
    print(f"Using IMG_SIZE = {IMG_SIZE} for evaluation transforms.")
    
    evaluate_onnx_model(
        args.onnx_model_path, args.coco_root, args.batch_size, args.workers, args.device,
        score_thresh_plot=args.score_thresh_plot,
        score_thresh_basic_acc=args.basic_acc_score_thresh,
        iou_thresh_basic_acc=args.basic_acc_iou_thresh,
        num_plot_samples=args.num_plot_samples
    )