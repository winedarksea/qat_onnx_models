# ONNX Model Design Guidelines for Edge Execution

This document outlines the optimized input/output specifications for final ONNX models created here. The goal is to maximize execution efficiency on edge devices (WebNN, WebGPU, WASM) by moving preprocessing and post-processing from host code into the ONNX graph.

## 1. Input Specifications

### Primary Image Input
- **Type**: `uint8` (Integer 0-255)
- **Shape**: `[BatchSize, 3, None, None]`
    - **Note on Batching**: For detection, `BatchSize=1` is typically recommended to manage memory safely. For classification, a dynamic batch dimension is helpful to process multiple detections at once.
    - **Uniformity**: All items in a single tensor batch must be the same dimensions before input [BatchSize, 3, H, W].
    - **Colorspace**: Colorspace is RGB.
- **Format**: NCHW (N, Channel, Height, Width)
- **Design Requirement**: 
    - Include a `Cast` node to `float32` immediately after input.
    - Include a `Resize` node to downsample to the model's training resolution (e.g., 256x256).
    - Include normalization logic (usually `Sub` and `Mul`) inside the graph to handle scaling.

---

## 2. Output Specifications

### Detection Models
- **Integrated NMS**: Models should include an integrated `NonMaxSuppression` node if NMS is used by the model, for deployment simplicity.
- **Consolidated Output**: Return a single tensor of shape `[N, 7]`. For simplicity these will be all float32.
    - **Indices**: `[x1, y1, x2, y2, score, class_id, batch_idx]` although most use cases are expected to just be a batch size of 1 with constant `batch_idx=0.0`.
- **Coordinate Scaling**: Output boxes must be in the coordinate space of the ONNX input tensor, not the model’s internal resized resolution. If the ONNX input tensor has shape [B,3,H_in,W_in] (for example 540x540), returned boxes must be scaled to H_in x W_in (for example 540x540), even if internal inference runs at 320x320.
- **Max Detections**: Limit output to top 40-100 detections to minimize IPC overhead.

### Classification Models
- **Dynamic Batching**: Use a dynamic batch dimension `[BatchSize, 3, H, W]`.
- **Integrated Logic**: Include `Softmax` node as the final layer.
- **Output**: Return probabilities with shape [B, C] where C is number of classes.

---

## 3. General Optimization Principles

- **Quantization**: Use **INT8 Quantization** and, where feasible, **Quantization Aware Training (QAT)**
- **Batching Strategy**:
    - **Detectors**: Optimize for `BatchSize=1`. Full-resolution images on edge devices consume significant VRAM/RAM. Processing one-by-one prevents memory crashes.
    - **Classifiers**: Use dynamic `BatchSize`. Because all items in a batch must share the same input resolution, standard practice is to resize all crops to a fixed square in host code before stacking them into the tensor if images of mixed sizes are to be passed in a single batch.
- **Avoid Downstream Loops**: Generally aim for pixel and bounding box math to be in the onnx graph.
- **WASM Compatibility**: Avoid operators not supported by standard ONNX opsets (Opset 17 recommended) to ensure broad compatibility across inference devices.
- **Operator Fusion**: Structure the graph to allow the runtime to fuse operations (e.g., keep normalization immediately before the first convolution).

---

## 4. Summary for LLM Prompting
When generating or modifying model export scripts, use the following directive:
> "Export the ONNX model with uint8 NCHW inputs, internal normalization/resizing, an integrated NMS node for detections, and a consolidated [N, 7] output tensor containing rescaled coordinates, scores, class IDs, and batch index."
