#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test an ONNX model exported from a PyTorch QAT pipeline.
Measures accuracy and average inference time per image using onnxruntime.
"""

import json
import os
import time
import random
from pathlib import Path

from tqdm import tqdm
import torch
import torchvision.transforms.v2 as T_v2
import torchvision.datasets as dsets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
import numpy as np
import onnx
import onnxruntime as ort

# Replicate necessary constants and data loading functions from the original script
SEED, IMG_SIZE, NUM_WORKERS = 42, 224, 0
random.seed(SEED) # Not strictly needed for ONNX testing but good practice for data loading consistency

# ─────────────────────────── transforms ──────────────────────────
# Replicate the validation transform logic
def _build_tf(is_train: bool):
    if is_train:
        # We only need the validation path for this script
        raise NotImplementedError("Training transforms are not needed for ONNX testing.")
    else: # Validation (train=False)
        return T_v2.Compose([
            T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T_v2.ToDtype(torch.uint8, scale=False) # Model expects uint8, preprocessing is inside
        ])

# ─────────────────────────── dataloader ──────────────────────────
def _rgb_loader(p):
    # Use torchvision.io.read_image to match the original script's loader
    # Ensure it's in RGB mode (3 channels)
    return read_image(p, mode=ImageReadMode.RGB)

def get_loader(root_dir: str, batch_size: int, is_train: bool):
    if is_train:
         raise NotImplementedError("Training loader is not needed for ONNX testing.")

    split = "val" # Only use validation data
    dataset = dsets.ImageFolder(
        root=os.path.join(root_dir, split),
        transform=_build_tf(is_train=False), # Pass is_train=False for validation
        loader=_rgb_loader # Use the custom RGB loader
    )
    # Pin memory is less relevant for CPU-based ONNX Runtime or if transferring to GPU provider
    # but we'll keep it for consistency if a CUDA provider is used.
    # Persistence workers/prefetch factor are mainly for speeding up data loading,
    # which might impact the overall script time but not the pure inference time measurement.
    pf = 4 if NUM_WORKERS else None
    pin_memory = False # Assume non-CUDA device for data loading before transfer
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=pin_memory, persistent_workers=bool(NUM_WORKERS),
                      prefetch_factor=pf, drop_last=False) # Don't drop last batch for evaluation

# ───────────────────────────── main ─────────────────────────────
onnx_path = "mobilenet_w1_0_mnv4s_pretrained_int8_fullpipe.onnx"  # "mobilenetv3_w0.75_int8.onnx"
data_dir = "filtered_imagenet2_native"
batch = 1
provider = "CPUExecutionProvider"  # CPUExecutionProvider, CUDAExecutionProvider, DmlExecutionProvider

if not os.path.exists(onnx_path):
    print(f"[ERROR] ONNX model not found at {onnx_path}")

if not os.path.exists(Path(data_dir) / "val"):
     print(f"[ERROR] Validation data not found in {data_dir}/val")

# Load class mapping to get number of classes (optional for accuracy, but good check)
try:
    with open(Path(data_dir) / "class_mapping.json") as f:
        ncls = len(json.load(f))
        print(f"[INFO] #classes = {ncls}")
except FileNotFoundError:
    print("[WARN] class_mapping.json not found. Cannot confirm number of classes.")


# Get the validation data loader
vl = get_loader(data_dir, batch, is_train=False)
print(f"[INFO] Loaded validation data with {len(vl.dataset)} images in {len(vl)} batches.")

# --- ONNX Graph Inspection ---
print(f"\n────────────────── Inspecting ONNX Graph ──────────────────")
try:
    onnx_model = onnx.load(onnx_path)
    # Optional: Check model validity (can sometimes fail on complex quantized models)
    # onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    # Define sets of operators we expect in a quantized graph
    # These are not necessarily "compute" ops but facilitate quantization
    quant_transition_ops = {"QuantizeLinear", "DequantizeLinear", "Cast", "Reshape", "Transpose", "Constant", "Shape", "Gather", "Unsqueeze", "Squeeze", "Flatten"}
    # Add common activation functions, pooling, and other non-linear/structural ops that
    # are often *not* represented as QLinear/Integer ops themselves, but operate on (de)quantized data
    common_non_quant_ops = {
        "Relu", "Clip", "Hardswish", "MaxPool", "GlobalAveragePool",
        "Add", "Mul", "Sub", "Div", # Basic arithmetic, might be from preprocessing or other layers
        "Resize", "ReduceMean", # Preprocessing or pooling related
        "Concat", "Gemm" # Gemm can be QLinearGemm, but standard Gemm is float
    }

    # Combine into a set of ops we will *not* report as "non-INT8 compute"
    known_quant_related_ops = quant_transition_ops.union(common_non_quant_ops)

    non_int8_compute_ops_found = set()

    print(f"[INFO] Analyzing {len(graph.node)} nodes in the ONNX graph...")
    for i, node in enumerate(graph.node):
        op_type = node.op_type

        # Check if the operator is a known INT8 compute op (e.g., QLinearConv, MatMulInteger)
        # These are the primary operations that should be quantized
        is_clearly_int8_compute = op_type.startswith("QLinear") or op_type.endswith("Integer") or op_type in {"MatMulInteger", "ConvInteger"} # Add other specific ones if known

        # If the op is NOT a clearly INT8 compute op AND NOT in our list of expected
        # non-quantized/transition ops, report it.
        # This heuristic aims to catch core compute ops that *should* have been quantized
        # but weren't, while ignoring expected float/transition ops.
        if not is_clearly_int8_compute and op_type not in known_quant_related_ops:
             non_int8_compute_ops_found.add(op_type)

    print("\n────────────────── Inspection Results ──────────────────")
    if not non_int8_compute_ops_found:
        print("[INFO] No unexpected non-INT8 compute operators found based on common patterns.")
        print("The graph appears to consist mainly of expected quantized, transition, and common non-quantizable operations.")
    else:
        print("[WARN] The following operator types were found which are typically not INT8 quantized compute operations")
        print("and are not common non-quantizable operations or transitions:")
        for op_type in sorted(list(non_int8_compute_ops_found)):
            print(f"- {op_type}")
        print("\nNote: This is a heuristic. Some operators might have INT8 support not covered by this check,")
        print("or might be expected FP32 layers in your specific model.")

    print("─────────────────────────────────────────────────────────────")

except onnx.onnx_cpp2py_export.checker.ValidationError as e:
     print(f"[ERROR] ONNX model check failed: {e}")
     # Decide whether to continue or exit if inspection fails
     # return # Uncomment to exit if check fails
except Exception as e:
    print(f"[ERROR] Error during ONNX graph inspection: {e}")
    import traceback
    traceback.print_exc()
    # Decide whether to continue or exit if inspection fails
    # return # Uncomment to exit if inspection fails

# Load the ONNX model
print(f"[INFO] Loading ONNX model from {onnx_path} with provider {provider}...")
try:
    sess_options = ort.SessionOptions()
    # Optional: Add optimization configurations if needed
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # Check available providers and use the specified one, fallback to CPU
    available_providers = ort.get_available_providers()
    print(f"[INFO] Available ONNX Runtime providers: {available_providers}")
    providers_to_use = [provider]
    if provider not in available_providers:
         print(f"[WARN] Specified provider '{provider}' not available. Falling back to CPUExecutionProvider.")
         providers_to_use = ["CPUExecutionProvider"]
    elif provider != "CPUExecutionProvider" and "CPUExecutionProvider" not in providers_to_use:
         # Ensure CPU is a fallback if a non-CPU provider is specified
         providers_to_use.append("CPUExecutionProvider")


    ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers_to_use)

    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"[INFO] Model loaded successfully. Input name: '{input_name}', Output name: '{output_name}'")
    print(f"[INFO] Model Input shape: {ort_session.get_inputs()[0].shape}, dtype: {ort_session.get_inputs()[0].type}")

except Exception as e:
    print(f"[ERROR] Failed to load ONNX model or initialize ONNX Runtime session: {e}")
    import traceback
    traceback.print_exc()

# Run inference and evaluate
correct_predictions = 0
total_samples = 0
total_inference_time = 0.0

print("[INFO] Starting inference...")
# Set the session to use the main device if possible (for GPU providers)
# ort_session.set_device(cfg.provider) # This method might not always be necessary or available depending on provider setup

for i, (img_batch, lab_batch) in tqdm(enumerate(vl), total=len(vl), desc="Running Inference"):
    # DataLoader provides torch.uint8 in NCHW format.
    # Convert Torch tensor to NumPy array with the correct dtype.
    # The ONNX model expects uint8 input (0-255), which is handled by the PreprocNorm layer internally.
    onnx_input = {input_name: img_batch.numpy()}

    # --- Measure Inference Time ---
    start_time = time.time()
    # Run the ONNX model inference
    ort_outputs = ort_session.run([output_name], onnx_input)
    end_time = time.time()
    # -----------------------------

    batch_inference_time = end_time - start_time
    total_inference_time += batch_inference_time
    total_samples += img_batch.size(0)

    # Process output logits
    # ort_outputs is a list, access the first element (logits)
    logits = ort_outputs[0]

    # Get predictions (index of the max logit for each sample in the batch)
    # Using numpy argmax directly on the numpy output
    predictions = np.argmax(logits, axis=1)

    # Compare with ground truth labels (convert torch tensor labels to numpy)
    correct_predictions += (predictions == lab_batch.numpy()).sum()

    # if (i + 1) % 1000 == 0 or (i + 1) == len(vl):
    #     print(f"Processed batch {i+1}/{len(vl)}")

# Calculate metrics
accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
average_time_per_image = total_inference_time / total_samples if total_samples > 0 else 0.0

print("\n─────────────────────────── Results ───────────────────────────")
print(f"Total images tested: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Total inference time: {total_inference_time:.4f} seconds")
print(f"Average inference time per image: {average_time_per_image * 1000:.4f} ms")
print("─────────────────────────────────────────────────────────────")
