#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test an ONNX model exported from a PyTorch QAT pipeline.
Measures accuracy and average inference time per image using onnxruntime.
Provides enhanced ONNX graph inspection for INT8 quantization assessment.
"""

import json
import os
import time
import random
from pathlib import Path
from collections import Counter

from tqdm import tqdm
import torch
import torchvision.transforms.v2 as T_v2
import torchvision.datasets as dsets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto # For checking tensor data types

# Replicate necessary constants and data loading functions
SEED, IMG_SIZE, NUM_WORKERS = 42, 224, 0
random.seed(SEED)

# ─────────────────────────── transforms ──────────────────────────
def _build_tf(is_train: bool):
    if is_train:
        raise NotImplementedError("Training transforms are not needed for ONNX testing.")
    else:
        return T_v2.Compose([
            T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T_v2.ToDtype(torch.uint8, scale=False)
        ])

# ─────────────────────────── dataloader ──────────────────────────
def _rgb_loader(p):
    return read_image(p, mode=ImageReadMode.RGB)

def get_loader(root_dir: str, batch_size: int, is_train: bool):
    if is_train:
         raise NotImplementedError("Training loader is not needed for ONNX testing.")
    split = "val"
    dataset = dsets.ImageFolder(
        root=os.path.join(root_dir, split),
        transform=_build_tf(is_train=False),
        loader=_rgb_loader
    )
    pf = 4 if NUM_WORKERS else None
    pin_memory = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=pin_memory, persistent_workers=bool(NUM_WORKERS),
                      prefetch_factor=pf, drop_last=False)

# ───────────────────────────── ONNX Graph Inspection Helper ─────────────────────────────
def get_tensor_type_name(tensor_proto_type):
    """Returns human-readable type name from TensorProto type enum."""
    for name, value in TensorProto.DataType.items():
        if value == tensor_proto_type:
            return name
    return "UNKNOWN_TYPE"

def inspect_onnx_graph(onnx_model_path):
    print(f"\n────────────────── Inspecting ONNX Graph: {onnx_model_path} ──────────────────")
    try:
        model = onnx.load(onnx_model_path)
        graph = model.graph

        # --- Build a map of tensor names to their data types ---
        tensor_types = {}
        # Inputs
        for inp in graph.input:
            tensor_types[inp.name] = inp.type.tensor_type.elem_type
        # Outputs
        for outp in graph.output:
            tensor_types[outp.name] = outp.type.tensor_type.elem_type
        # Initializers (weights, biases)
        for initializer in graph.initializer:
            tensor_types[initializer.name] = initializer.data_type
        # ValueInfo (intermediate tensors)
        for vi in graph.value_info:
            tensor_types[vi.name] = vi.type.tensor_type.elem_type


        # --- Operator Categorization ---
        # Ops that are explicitly INT8 compute (or part of its direct path)
        INT8_COMPUTE_OPS = {"QLinearConv", "ConvInteger", "QLinearMatMul", "MatMulInteger",
                            "QLinearAdd", "QLinearMul", "QLinearAveragePool", "QLinearGlobalAveragePool",
                            "MaxPoolInt8", "AveragePoolInt8"} # Add more as needed
        # Ops that are for quantization/dequantization
        QUANT_DEQUANT_OPS = {"QuantizeLinear", "DequantizeLinear"}
        # Common FP32 compute ops that *should ideally be* INT8 in a quantized model
        POTENTIAL_FP32_COMPUTE_FALLBACKS = {"Conv", "MatMul", "Gemm",
                                            "Add", "Mul", "Div", "Sub", # If operating on FP32 feature maps
                                            "AveragePool", "GlobalAveragePool", "MaxPool"} # If not QLinear* versions
        # Ops that should ideally be fused or are problematic if present
        SHOULD_BE_FUSED_OR_ABSENT = {"BatchNormalization"}
        # Common ops that are often part of the graph structure, activations, etc.
        # Not typically the main compute bottlenecks if the heavy ops are quantized.
        SUPPORT_OPS = {
            "Constant", "ConstantOfShape", "Identity", "Cast", "Reshape", "Transpose",
            "Shape", "Gather", "Unsqueeze", "Squeeze", "Flatten", "Concat", "Split",
            "Relu", "Clip", "Sigmoid", "HardSigmoid", "HardSwish", "Swish",
            "Resize", "ReduceMean", "Softmax", "LogSoftmax"
        }

        op_counts = Counter()
        problematic_fp32_ops = Counter()
        unfused_ops_found = Counter()
        int8_compute_ops_found = Counter()
        quant_dequant_ops_found = Counter()
        other_ops_found = Counter()

        print(f"[INFO] Analyzing {len(graph.node)} nodes in the ONNX graph...")
        for node in graph.node:
            op_type = node.op_type
            op_counts[op_type] += 1

            is_fp32_compute_fallback = False
            if op_type in POTENTIAL_FP32_COMPUTE_FALLBACKS:
                # Check input/output types for this node.
                # A simple heuristic: if any non-parameter input is FP32, or output is FP32,
                # it's likely an FP32 operation.
                # Note: Weights (initializers) can be INT8 for Q/DQ nodes.
                # This check is primarily for activation paths.
                node_is_fp32 = False
                for input_name in node.input:
                    # Ignore initializers for this check as they might be INT8 for QDQ
                    if input_name in tensor_types and input_name not in {init.name for init in graph.initializer}:
                        if tensor_types[input_name] == TensorProto.FLOAT:
                            node_is_fp32 = True
                            break
                if not node_is_fp32: # If inputs look okay, check outputs
                    for output_name in node.output:
                        if output_name in tensor_types and tensor_types[output_name] == TensorProto.FLOAT:
                            node_is_fp32 = True
                            break
                
                if node_is_fp32:
                    problematic_fp32_ops[op_type] += 1
                    is_fp32_compute_fallback = True


            if op_type in INT8_COMPUTE_OPS:
                int8_compute_ops_found[op_type] += 1
            elif op_type in QUANT_DEQUANT_OPS:
                quant_dequant_ops_found[op_type] += 1
            elif op_type in SHOULD_BE_FUSED_OR_ABSENT:
                unfused_ops_found[op_type] += 1
            elif not is_fp32_compute_fallback and op_type not in SUPPORT_OPS: # If not already categorized
                 other_ops_found[op_type] += 1


        print("\n────────────────── Graph Inspection Summary ──────────────────")
        print(f"[INFO] Total nodes: {len(graph.node)}")
        print(f"[INFO] Unique operator types found: {len(op_counts)}")

        print("\n--- Explicit INT8 Compute Ops ---")
        if int8_compute_ops_found:
            for op, count in int8_compute_ops_found.items(): print(f"- {op}: {count}")
        else:
            print("  None explicitly identified (e.g., QLinearConv, ConvInteger). Model might be using QDQ format.")

        print("\n--- Quantization/Dequantization Ops ---")
        if quant_dequant_ops_found:
            for op, count in quant_dequant_ops_found.items(): print(f"- {op}: {count}")
            print(f"  Total Q/DQ ops: {sum(quant_dequant_ops_found.values())} (High numbers can indicate performance overhead)")
        else:
            print("  No QuantizeLinear/DequantizeLinear ops found. This is unusual for typical INT8 models unless it's QOperator format without any FP32 parts.")

        print("\n--- Potential FP32 Compute Fallbacks (Problematic if INT8 expected) ---")
        if problematic_fp32_ops:
            for op, count in problematic_fp32_ops.items(): print(f"- {op}: {count}")
            print("  Note: These are compute-intensive ops that appear to be running in FP32.")
        else:
            print("  No obvious FP32 compute fallbacks detected among common op types.")

        print("\n--- Ops That Should Ideally Be Fused/Absent in Quantized Model ---")
        if unfused_ops_found:
            for op, count in unfused_ops_found.items(): print(f"- {op}: {count} (e.g., BatchNormalization should typically be fused)")
        else:
            print("  No common unfused ops like BatchNormalization detected.")
        
        if other_ops_found:
            print("\n--- Other Operator Types Found (Review if unexpected) ---")
            for op, count in sorted(other_ops_found.items()):
                print(f"- {op}: {count}")

        print("\n--- Overall Operator Counts (Top 10) ---")
        for op, count in op_counts.most_common(10):
            print(f"- {op}: {count}")
        
        print("\n[RECOMMENDATION] For detailed analysis:")
        print("1. Visualize the .onnx model with Netron to inspect node properties and connections.")
        print("2. Enable ONNX Runtime profiling (see script comments) to get per-operator timings.")
        print("─────────────────────────────────────────────────────────────")
        return True

    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
         print(f"[ERROR] ONNX model check failed: {e}")
         return False
    except Exception as e:
        print(f"[ERROR] Error during ONNX graph inspection: {e}")
        import traceback
        traceback.print_exc()
        return False

# ───────────────────────────── main ─────────────────────────────
# onnx_path = "mobilenet_w1_0_mnv3_pretrained_int8_fullpipe.onnx"
onnx_path = "mobilenet_w1_0_mnv4s_pretrained_int8_fullpipe.onnx"
data_dir = "filtered_imagenet2_native" # Make sure this path is correct
batch = 1 # Keep batch=1 for per-image timing, but can increase for throughput tests
provider = "CPUExecutionProvider"  # CPUExecutionProvider, CUDAExecutionProvider

if not os.path.exists(onnx_path):
    print(f"[ERROR] ONNX model not found at {onnx_path}")
    exit()

if not os.path.exists(Path(data_dir) / "val"):
     print(f"[ERROR] Validation data not found in {data_dir}/val. Please check the path.")
     exit()

# Load class mapping
try:
    with open(Path(data_dir) / "class_mapping.json") as f:
        ncls = len(json.load(f))
        print(f"[INFO] #classes = {ncls}")
except FileNotFoundError:
    print("[WARN] class_mapping.json not found. Cannot confirm number of classes.")

# Perform detailed graph inspection
inspection_successful = inspect_onnx_graph(onnx_path)
if not inspection_successful:
    print("[WARN] Proceeding with inference despite inspection issues.")

# Get the validation data loader
vl = get_loader(data_dir, batch, is_train=False)
print(f"[INFO] Loaded validation data with {len(vl.dataset)} images in {len(vl)} batches.")


# Load the ONNX model
print(f"[INFO] Loading ONNX model from {onnx_path} with provider {provider}...")
try:
    sess_options = ort.SessionOptions()

    # ** FOR DETAILED PERFORMANCE ANALYSIS, ENABLE PROFILING **
    if True:
        sess_options.enable_profiling = True
        profile_file_name = f"{Path(onnx_path).stem}_profile.json"
        sess_options.profile_file_prefix = Path(profile_file_name).stem # ORT will append suffix
        print(f"[INFO] ONNX Runtime profiling enabled. Profile will be saved to a file starting with '{Path(profile_file_name).stem}'.")
        print("         You can view this profile in chrome://tracing or with tools like 'pyprof2flame.py'")


    available_providers = ort.get_available_providers()
    print(f"[INFO] Available ONNX Runtime providers: {available_providers}")
    providers_to_use = [provider]
    if provider not in available_providers:
         print(f"[WARN] Specified provider '{provider}' not available. Falling back to CPUExecutionProvider.")
         providers_to_use = ["CPUExecutionProvider"]
    elif provider != "CPUExecutionProvider" and "CPUExecutionProvider" not in providers_to_use:
         providers_to_use.append("CPUExecutionProvider") # Ensure CPU fallback

    ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers_to_use)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"[INFO] Model loaded successfully. Input name: '{input_name}', Output name: '{output_name}'")
    model_input_shape = ort_session.get_inputs()[0].shape
    model_input_type_enum = ort_session.get_inputs()[0].type
    print(f"[INFO] Model Input shape: {model_input_shape}, dtype: {model_input_type_enum}")
    if model_input_type_enum != 'tensor(uint8)':
         print(f"[WARN] Model input type is {model_input_type_enum}, not tensor(uint8). This might indicate issues with full quantization if uint8 input was expected.")


except Exception as e:
    print(f"[ERROR] Failed to load ONNX model or initialize ONNX Runtime session: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Run inference and evaluate
correct_predictions = 0
total_samples = 0
total_inference_time = 0.0

print("[INFO] Starting inference...")
for i, (img_batch, lab_batch) in tqdm(enumerate(vl), total=len(vl), desc="Running Inference"):
    onnx_input = {input_name: img_batch.numpy()} # Dataloader provides uint8 NCHW

    start_time = time.perf_counter() # Use perf_counter for more precise timing
    ort_outputs = ort_session.run([output_name], onnx_input)
    end_time = time.perf_counter()

    batch_inference_time = end_time - start_time
    total_inference_time += batch_inference_time
    total_samples += img_batch.size(0)

    logits = ort_outputs[0]
    predictions = np.argmax(logits, axis=1)
    correct_predictions += (predictions == lab_batch.numpy()).sum()

# After loop, if profiling was enabled:
if sess_options.enable_profiling:
    prof_file = ort_session.end_profiling() # This is the actual file name
    print(f"[INFO] Profiling complete. Profile saved to: {prof_file}")
    print(f"         You can view this JSON profile in chrome://tracing")


# Calculate metrics
accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
average_time_per_image = total_inference_time / total_samples if total_samples > 0 else 0.0

print("\n─────────────────────────── Results ───────────────────────────")
print(f"Total images tested: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Total inference time: {total_inference_time:.4f} seconds")
print(f"Average inference time per image: {average_time_per_image * 1000:.4f} ms")
print(f"Images per second (throughput): {1.0 / average_time_per_image if average_time_per_image > 0 else 0:.2f} IPS")
print("─────────────────────────────────────────────────────────────")