#!/usr/bin/env python3
"""
Test an ONNX model for conformance with scripts/ONNX_GUIDELINES.md.

Default model:
  model_saves/yolo26_coco_320_int8.onnx

What this script does:
1. Runs static ONNX graph checks for guideline conformance.
2. Runs inference on scripts/test_image.jpg.
3. Displays the test image with box overlays if detections can be parsed.
4. If output is non-compliant, reports output shape/format details.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    import onnx
    from onnx import TensorProto
except Exception:
    onnx = None
    TensorProto = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = REPO_ROOT / "model_saves" / "picodet_v5_int8.onnx"   # "yolo26_coco_320_int8.onnx" maip_vertexai_2nodehrs_medium_guideline
DEFAULT_IMAGE = REPO_ROOT / "scripts" / "test_image_2.jpg"


@dataclass
class CheckResult:
    name: str
    passed: bool | None
    detail: str
    required: bool = True


@dataclass
class DetectionOutput:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    batch_idx: np.ndarray
    source_format: str


@dataclass
class PrecisionAudit:
    total_nodes: int
    total_initializers: int
    total_initializer_bytes: int
    int8_initializers: int
    int8_initializer_bytes: int
    qdq_nodes: int
    total_compute_nodes: int
    quantized_compute_nodes: int
    float_compute_nodes: int
    prepost_float_nodes: int
    advisory_level: str
    advisory_message: str
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test ONNX model conformance and visualize detections.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path to ONNX model.")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to input image.")
    p.add_argument("--score-threshold", type=float, default=0.25, help="Overlay score threshold.")
    p.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime provider (default: CPUExecutionProvider).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive matplotlib window (useful for headless runs).",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit non-zero if required guideline checks fail.",
    )
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before timing.",
    )
    p.add_argument(
        "--timing-runs",
        type=int,
        default=10,
        help="Number of timed runs for single-image inference.",
    )
    p.add_argument(
        "--fallback-topk",
        type=int,
        default=20,
        help="If no boxes pass threshold, overlay top-K boxes by score.",
    )
    return p.parse_args()


def _dim_to_str(dim: Any) -> str:
    if isinstance(dim, (int, np.integer)):
        return str(int(dim))
    return str(dim)


def _shape_to_str(shape: list[Any]) -> str:
    return "[" + ", ".join(_dim_to_str(d) for d in shape) + "]"


def _shape_from_value_info(vi: Any) -> list[Any]:
    dims: list[Any] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def _onnx_dtype_name(elem_type: int) -> str:
    if TensorProto is None:
        return f"TensorProto({elem_type})"
    for n, v in TensorProto.DataType.items():
        if v == elem_type:
            return n
    return f"TensorProto({elem_type})"


def _collect_graph_checks(onnx_path: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    ctx: dict[str, Any] = {}

    if onnx is None or TensorProto is None:
        checks.append(
            CheckResult(
                name="ONNX Graph Checks Available",
                passed=None,
                detail="`onnx` package not installed, graph-level checks skipped.",
                required=False,
            )
        )
        return checks, ctx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    graph = model.graph
    op_counts = Counter(n.op_type for n in graph.node)
    ctx["op_counts"] = op_counts
    ctx["precision_audit"] = _collect_precision_audit(model, op_counts)

    if not graph.input:
        checks.append(CheckResult("Model Has Input", False, "No graph inputs found."))
        return checks, ctx

    in0 = graph.input[0]
    in_shape = _shape_from_value_info(in0)
    in_dtype = in0.type.tensor_type.elem_type
    ctx["input_shape"] = in_shape
    ctx["input_dtype"] = in_dtype

    checks.append(
        CheckResult(
            "Input DType",
            in_dtype == TensorProto.UINT8,
            f"found={_onnx_dtype_name(in_dtype)}, expected=UINT8",
        )
    )

    rank_ok = len(in_shape) == 4
    checks.append(
        CheckResult(
            "Input Rank",
            rank_ok,
            f"found={_shape_to_str(in_shape)}, expected rank-4 NCHW [B,3,H,W]",
        )
    )

    ch_ok = rank_ok and isinstance(in_shape[1], int) and in_shape[1] == 3
    checks.append(
        CheckResult(
            "Input Channel Dimension",
            ch_ok,
            f"found channel dim={in_shape[1] if rank_ok else 'n/a'}, expected=3",
        )
    )

    checks.append(
        CheckResult(
            "Has Cast Node",
            op_counts.get("Cast", 0) > 0,
            f"Cast count={op_counts.get('Cast', 0)}",
        )
    )
    checks.append(
        CheckResult(
            "Has Resize Node",
            op_counts.get("Resize", 0) > 0,
            f"Resize count={op_counts.get('Resize', 0)}",
        )
    )
    norm_count = op_counts.get("Sub", 0) + op_counts.get("Mul", 0) + op_counts.get("Div", 0)
    checks.append(
        CheckResult(
            "Has Normalization Arithmetic (Sub/Mul/Div)",
            norm_count > 0,
            f"Sub={op_counts.get('Sub', 0)}, Mul={op_counts.get('Mul', 0)}, Div={op_counts.get('Div', 0)}",
        )
    )

    checks.append(
        CheckResult(
            "Integrated NMS Node Present",
            op_counts.get("NonMaxSuppression", 0) > 0,
            f"NonMaxSuppression count={op_counts.get('NonMaxSuppression', 0)}",
            required=False,
        )
    )

    outputs = list(graph.output)
    checks.append(
        CheckResult(
            "Single Output Tensor",
            len(outputs) == 1,
            f"output_count={len(outputs)}",
        )
    )

    if outputs:
        out0 = outputs[0]
        out_shape = _shape_from_value_info(out0)
        out_dtype = out0.type.tensor_type.elem_type
        ctx["output_shape"] = out_shape
        ctx["output_dtype"] = out_dtype

        checks.append(
            CheckResult(
                "Output DType",
                out_dtype == TensorProto.FLOAT,
                f"found={_onnx_dtype_name(out_dtype)}, expected=FLOAT",
            )
        )

        out_shape_ok = len(out_shape) == 2 and isinstance(out_shape[1], int) and out_shape[1] == 7
        checks.append(
            CheckResult(
                "Output Shape",
                out_shape_ok,
                f"found={_shape_to_str(out_shape)}, expected=[N,7]",
            )
        )

    return checks, ctx


def _initializer_nbytes(init: Any) -> int:
    if hasattr(init, "raw_data") and init.raw_data:
        return int(len(init.raw_data))

    # Fall back to array materialization for non-raw tensor encodings.
    try:
        arr = onnx.numpy_helper.to_array(init)
        return int(arr.nbytes)
    except Exception:
        return 0


def _collect_precision_audit(model: Any, op_counts: Counter[str]) -> PrecisionAudit:
    graph = model.graph
    int8_types = {TensorProto.INT8, TensorProto.UINT8}
    explicit_qcompute_ops = {"QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger", "QGemm"}
    base_compute_ops = {"Conv", "MatMul", "Gemm", "ConvTranspose"}

    total_initializer_bytes = 0
    int8_initializer_bytes = 0
    int8_initializers = 0
    for init in graph.initializer:
        nbytes = _initializer_nbytes(init)
        total_initializer_bytes += nbytes
        if init.data_type in int8_types:
            int8_initializers += 1
            int8_initializer_bytes += nbytes

    producer_by_output: dict[str, Any] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer_by_output[out] = node
    initializer_dtype = {i.name: i.data_type for i in graph.initializer}

    def _input_from_int8_qdq(name: str) -> bool:
        prod = producer_by_output.get(name)
        if prod is None or prod.op_type != "DequantizeLinear" or not prod.input:
            return False

        source = prod.input[0]
        if initializer_dtype.get(source) in int8_types:
            return True

        src_prod = producer_by_output.get(source)
        return bool(src_prod is not None and src_prod.op_type == "QuantizeLinear")

    total_compute_nodes = 0
    quantized_compute_nodes = 0
    for node in graph.node:
        op = node.op_type
        if op in explicit_qcompute_ops:
            total_compute_nodes += 1
            quantized_compute_nodes += 1
            continue
        if op in base_compute_ops:
            total_compute_nodes += 1
            if any(_input_from_int8_qdq(inp_name) for inp_name in node.input if inp_name):
                quantized_compute_nodes += 1

    float_compute_nodes = max(0, total_compute_nodes - quantized_compute_nodes)
    qdq_nodes = int(op_counts.get("QuantizeLinear", 0) + op_counts.get("DequantizeLinear", 0))
    prepost_float_nodes = int(
        op_counts.get("Cast", 0) + op_counts.get("Resize", 0) + op_counts.get("NonMaxSuppression", 0)
    )

    int8_param_ratio = (
        float(int8_initializer_bytes) / float(total_initializer_bytes)
        if total_initializer_bytes > 0
        else 0.0
    )
    quantized_compute_ratio = (
        float(quantized_compute_nodes) / float(total_compute_nodes)
        if total_compute_nodes > 0
        else 0.0
    )

    reasons: list[str] = []
    reasons.append(
        "int8_param_ratio={:.1%}, quantized_compute_ratio={:.1%}".format(
            int8_param_ratio, quantized_compute_ratio
        )
    )
    if prepost_float_nodes > 0:
        reasons.append(
            "Found {} float-leaning pre/post nodes (Cast/Resize/NMS), often left on CPU/GPU.".format(
                prepost_float_nodes
            )
        )

    if total_compute_nodes == 0:
        advisory_level = "UNKNOWN"
        advisory_message = "No Conv/MatMul/Gemm-style compute nodes found; offload estimate is inconclusive."
    elif int8_param_ratio >= 0.70 and quantized_compute_ratio >= 0.80:
        advisory_level = "LIKELY_MOSTLY_NPU"
        advisory_message = "Model core looks strongly quantized; many INT8 NPUs should offload most core compute."
    elif int8_param_ratio >= 0.40 or quantized_compute_ratio >= 0.40:
        advisory_level = "MIXED_OFFLOAD"
        advisory_message = (
            "Model appears partially quantized; likely mixed execution with some CPU/GPU fallback on many NPUs."
        )
    else:
        advisory_level = "LOW_NPU_OFFLOAD"
        advisory_message = "Model core looks mostly float; many INT8 NPUs may offload little or none of inference."

    return PrecisionAudit(
        total_nodes=len(graph.node),
        total_initializers=len(graph.initializer),
        total_initializer_bytes=total_initializer_bytes,
        int8_initializers=int8_initializers,
        int8_initializer_bytes=int8_initializer_bytes,
        qdq_nodes=qdq_nodes,
        total_compute_nodes=total_compute_nodes,
        quantized_compute_nodes=quantized_compute_nodes,
        float_compute_nodes=float_compute_nodes,
        prepost_float_nodes=prepost_float_nodes,
        advisory_level=advisory_level,
        advisory_message=advisory_message,
        reasons=reasons,
    )


def _print_precision_audit(audit: PrecisionAudit | None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if audit is None:
        print("\n=== Quantization / NPU Audit ===")
        print("- Not available (ONNX graph inspection disabled).")
        checks.append(
            CheckResult(
                name="Quantization / NPU Audit Available",
                passed=None,
                detail="ONNX package not available for graph precision analysis.",
                required=False,
            )
        )
        return checks

    int8_param_ratio = (
        float(audit.int8_initializer_bytes) / float(audit.total_initializer_bytes)
        if audit.total_initializer_bytes > 0
        else 0.0
    )
    quantized_compute_ratio = (
        float(audit.quantized_compute_nodes) / float(audit.total_compute_nodes)
        if audit.total_compute_nodes > 0
        else 0.0
    )

    print("\n=== Quantization / NPU Audit ===")
    print(
        "- Initializers: total={}, int8={} | int8 bytes={}/{} ({:.1%})".format(
            audit.total_initializers,
            audit.int8_initializers,
            audit.int8_initializer_bytes,
            audit.total_initializer_bytes,
            int8_param_ratio,
        )
    )
    print(
        "- Compute nodes (Conv/MatMul/Gemm family): total={}, estimated quantized={}, float-leaning={} | quantized ratio={:.1%}".format(
            audit.total_compute_nodes,
            audit.quantized_compute_nodes,
            audit.float_compute_nodes,
            quantized_compute_ratio,
        )
    )
    print(
        "- Quantization plumbing nodes (Q/DQ): {} | float pre/post nodes (Cast/Resize/NMS): {}".format(
            audit.qdq_nodes, audit.prepost_float_nodes
        )
    )
    print(f"- NPU advisory: {audit.advisory_level} | {audit.advisory_message}")
    for reason in audit.reasons:
        print(f"  {reason}")

    checks.append(
        CheckResult(
            name="INT8 Initializer Byte Coverage",
            passed=int8_param_ratio >= 0.70,
            detail="int8_initializer_bytes_ratio={:.1%}".format(int8_param_ratio),
            required=False,
        )
    )
    if audit.total_compute_nodes > 0:
        checks.append(
            CheckResult(
                name="Estimated Quantized Compute Coverage",
                passed=quantized_compute_ratio >= 0.80,
                detail="estimated_quantized_compute_ratio={:.1%}".format(quantized_compute_ratio),
                required=False,
            )
        )
    checks.append(
        CheckResult(
            name="NPU Offload Risk",
            passed=audit.advisory_level not in {"LOW_NPU_OFFLOAD"},
            detail=f"advisory_level={audit.advisory_level}",
            required=False,
        )
    )

    return checks


def _select_runtime_provider(provider: str) -> list[str]:
    available = ort.get_available_providers()
    if provider in available:
        if provider == "CPUExecutionProvider":
            return [provider]
        if "CPUExecutionProvider" in available:
            return [provider, "CPUExecutionProvider"]
        return [provider]
    if "CPUExecutionProvider" in available:
        print(f"[WARN] Provider '{provider}' unavailable, using CPUExecutionProvider.")
        return ["CPUExecutionProvider"]
    print(f"[WARN] Provider '{provider}' unavailable; falling back to ONNX Runtime defaults.")
    return []


def _is_static_dim(dim: Any) -> bool:
    return isinstance(dim, (int, np.integer)) and int(dim) > 0


def _prepare_input_tensor(img_rgb: np.ndarray, input_meta: Any) -> tuple[np.ndarray, np.ndarray, str]:
    model_shape = list(input_meta.shape)
    input_type = input_meta.type

    if len(model_shape) != 4:
        raise ValueError(f"Expected rank-4 input tensor, got shape={model_shape}")

    # Infer layout from shape hints.
    layout = "NCHW"
    if _is_static_dim(model_shape[1]) and int(model_shape[1]) == 3:
        layout = "NCHW"
    elif _is_static_dim(model_shape[3]) and int(model_shape[3]) == 3:
        layout = "NHWC"

    fed_img = img_rgb
    h_model = int(model_shape[2]) if _is_static_dim(model_shape[2]) else None
    w_model = int(model_shape[3]) if _is_static_dim(model_shape[3]) else None

    if layout == "NHWC":
        h_model = int(model_shape[1]) if _is_static_dim(model_shape[1]) else None
        w_model = int(model_shape[2]) if _is_static_dim(model_shape[2]) else None

    if h_model is not None and w_model is not None:
        fed_img = np.array(Image.fromarray(img_rgb).resize((w_model, h_model), Image.BILINEAR))

    if layout == "NCHW":
        tensor = np.transpose(fed_img, (2, 0, 1))[None, ...]
    else:
        tensor = fed_img[None, ...]

    if input_type == "tensor(uint8)":
        tensor = tensor.astype(np.uint8, copy=False)
    elif input_type == "tensor(float)":
        tensor = (tensor.astype(np.float32) / 255.0).astype(np.float32)
    elif input_type == "tensor(float16)":
        tensor = (tensor.astype(np.float32) / 255.0).astype(np.float16)
    elif input_type == "tensor(int8)":
        tensor = np.clip(tensor.astype(np.int16) - 128, -128, 127).astype(np.int8)
    else:
        tensor = tensor.astype(np.float32)

    return tensor, fed_img, layout


def _summarize_outputs(output_names: list[str], outputs: list[np.ndarray]) -> list[str]:
    lines: list[str] = []
    for i, (name, out) in enumerate(zip(output_names, outputs)):
        arr = np.asarray(out)
        detail = f"[{i}] {name}: shape={arr.shape}, dtype={arr.dtype}"
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            detail += f", min={arr_min:.5f}, max={arr_max:.5f}"
        lines.append(detail)
    return lines


def _extract_detections(output_names: list[str], outputs: list[np.ndarray]) -> DetectionOutput | None:
    arrays = [np.asarray(o) for o in outputs]
    lowered = [n.lower() for n in output_names]

    # Case A: single consolidated output [N,7] or [1,N,7] or [N,6].
    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[1] in (6, 7):
            arr = np.transpose(arr[0], (1, 0))

        if arr.ndim == 2 and arr.shape[1] >= 6:
            boxes = arr[:, 0:4].astype(np.float32, copy=False)
            scores = arr[:, 4].astype(np.float32, copy=False)
            classes = arr[:, 5].astype(np.float32, copy=False)
            if arr.shape[1] >= 7:
                batch = arr[:, 6].astype(np.float32, copy=False)
            else:
                batch = np.zeros((arr.shape[0],), dtype=np.float32)
            return DetectionOutput(boxes, scores, classes, batch, f"single_tensor_{arr.shape}")

    # Case B: separate tensors like det_boxes/det_scores/class_idx/batch_idx.
    box_idx = None
    for i, a in enumerate(arrays):
        if a.ndim == 2 and a.shape[1] == 4:
            box_idx = i
            if "box" in lowered[i]:
                break
    if box_idx is None:
        return None

    boxes = arrays[box_idx].astype(np.float32, copy=False)
    n = boxes.shape[0]

    score_idx = None
    class_idx = None
    batch_idx = None
    for i, (name, a) in enumerate(zip(lowered, arrays)):
        if i == box_idx:
            continue
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
            arrays[i] = a
        if a.ndim != 1 or a.shape[0] != n:
            continue
        if score_idx is None and ("score" in name or "conf" in name):
            score_idx = i
            continue
        if class_idx is None and ("class" in name or "cls" in name or "label" in name):
            class_idx = i
            continue
        if batch_idx is None and "batch" in name:
            batch_idx = i

    if score_idx is None:
        for i, a in enumerate(arrays):
            if i == box_idx or i == class_idx or i == batch_idx:
                continue
            if a.ndim == 1 and a.shape[0] == n:
                score_idx = i
                break
    if class_idx is None:
        for i, a in enumerate(arrays):
            if i in (box_idx, score_idx, batch_idx):
                continue
            if a.ndim == 1 and a.shape[0] == n:
                class_idx = i
                break

    if score_idx is None:
        return None
    if class_idx is None:
        classes = np.zeros((n,), dtype=np.float32)
    else:
        classes = arrays[class_idx].astype(np.float32, copy=False)

    if batch_idx is None:
        batch = np.zeros((n,), dtype=np.float32)
    else:
        batch = arrays[batch_idx].astype(np.float32, copy=False)

    scores = arrays[score_idx].astype(np.float32, copy=False)
    return DetectionOutput(boxes, scores, classes, batch, "separate_tensors")


def _plot_detections(
    img_rgb: np.ndarray,
    detections: DetectionOutput | None,
    score_threshold: float,
    input_size_hw: tuple[int, int],
    model_path: Path,
    fallback_topk: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.imshow(img_rgb)
    ax.axis("off")

    title = f"{model_path.name}"
    if detections is None:
        ax.set_title(f"{title} | no parseable detection output")
        plt.show(block=True)
        return

    h, w = input_size_hw
    valid_base = np.isfinite(detections.scores)
    valid_base &= np.all(np.isfinite(detections.boxes_xyxy), axis=1)
    valid_base &= detections.batch_idx == 0

    valid = valid_base & (detections.scores >= float(score_threshold))
    boxes = detections.boxes_xyxy[valid]
    scores = detections.scores[valid]
    class_ids = detections.class_ids[valid]
    fallback_used = False

    if boxes.shape[0] == 0 and int(fallback_topk) > 0 and np.any(valid_base):
        fallback_used = True
        cand_idx = np.where(valid_base)[0]
        cand_scores = detections.scores[cand_idx]
        order = np.argsort(-cand_scores)
        top_idx = cand_idx[order[: int(fallback_topk)]]
        boxes = detections.boxes_xyxy[top_idx]
        scores = detections.scores[top_idx]
        class_ids = detections.class_ids[top_idx]

    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(w - 1.0, x1))
        y1 = max(0.0, min(h - 1.0, y1))
        x2 = max(0.0, min(w - 1.0, x2))
        y2 = max(0.0, min(h - 1.0, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0.0 or bh <= 0.0:
            continue

        color = plt.cm.tab20(int(cls) % 20)
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0.0, y1 - 4.0),
            f"cls {int(cls)} | {score:.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 1, "edgecolor": "none"},
        )

    mode_text = f"threshold={score_threshold:.2f}"
    if fallback_used:
        mode_text = f"fallback_topk={int(fallback_topk)} (no boxes met threshold)"
    ax.set_title(f"{title} | detections shown={len(scores)} | source={detections.source_format} | {mode_text}")
    plt.show(block=True)


def _print_check_table(checks: list[CheckResult]) -> tuple[bool, bool]:
    required_pass = True
    any_unknown = False

    print("\n=== Guideline Compliance Checks ===")
    for c in checks:
        if c.passed is True:
            state = "PASS"
        elif c.passed is False:
            state = "FAIL"
        else:
            state = "SKIP"
            any_unknown = True
        req = "required" if c.required else "advisory"
        print(f"- {state:<4} | {req:<8} | {c.name}: {c.detail}")
        if c.required and c.passed is not True:
            required_pass = False
    return required_pass, any_unknown


def _runtime_box_bounds_check(
    det: DetectionOutput | None, input_hw: tuple[int, int]
) -> CheckResult:
    if det is None or det.boxes_xyxy.size == 0:
        return CheckResult(
            name="Runtime Box Coordinate Bounds",
            passed=None,
            detail="No parseable boxes found at runtime.",
            required=False,
        )

    h, w = input_hw
    b = det.boxes_xyxy
    finite = np.all(np.isfinite(b), axis=1)
    if not finite.any():
        return CheckResult(
            name="Runtime Box Coordinate Bounds",
            passed=False,
            detail="All predicted boxes are non-finite.",
            required=False,
        )

    b = b[finite]
    within = (b[:, 0] >= -1.0) & (b[:, 1] >= -1.0) & (b[:, 2] <= (w + 1.0)) & (b[:, 3] <= (h + 1.0))
    ratio = float(np.mean(within)) if b.shape[0] else 0.0
    return CheckResult(
        name="Runtime Box Coordinate Bounds",
        passed=ratio > 0.80,
        detail=f"within_bounds_ratio={ratio:.3f} (input_w={w}, input_h={h})",
        required=False,
    )


def _runtime_box_scale_coverage_check(
    det: DetectionOutput | None, input_hw: tuple[int, int]
) -> CheckResult:
    if det is None or det.boxes_xyxy.size == 0:
        return CheckResult(
            name="Runtime Box Scale Coverage",
            passed=None,
            detail="No parseable boxes found at runtime.",
            required=False,
        )

    h, w = input_hw
    b = det.boxes_xyxy
    finite = np.all(np.isfinite(b), axis=1)
    b = b[finite]
    if b.shape[0] == 0:
        return CheckResult(
            name="Runtime Box Scale Coverage",
            passed=False,
            detail="No finite boxes found.",
            required=False,
        )

    x2_p95 = float(np.quantile(b[:, 2], 0.95))
    y2_p95 = float(np.quantile(b[:, 3], 0.95))
    cov_x = x2_p95 / max(1.0, float(w))
    cov_y = y2_p95 / max(1.0, float(h))
    # Heuristic: if both are tiny, boxes likely still in internal feature scale.
    likely_scaled = (cov_x > 0.20) or (cov_y > 0.20)
    return CheckResult(
        name="Runtime Box Scale Coverage",
        passed=likely_scaled,
        detail=(
            "x2_p95/input_w={cx:.4f}, y2_p95/input_h={cy:.4f} "
            "(low values suggest boxes not rescaled to input space)"
        ).format(cx=cov_x, cy=cov_y),
        required=False,
    )


def _tensor_column_stats(arr: np.ndarray) -> list[str]:
    if arr.ndim != 2:
        return []
    lines: list[str] = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        if col.size == 0:
            lines.append(f"col[{c}]: empty")
            continue
        lines.append(
            "col[{c}]: min={mn:.5f}, max={mx:.5f}, mean={me:.5f}, p95={p95:.5f}".format(
                c=c,
                mn=float(np.nanmin(col)),
                mx=float(np.nanmax(col)),
                me=float(np.nanmean(col)),
                p95=float(np.nanquantile(col, 0.95)),
            )
        )
    return lines


def _inference_timing_ms(
    session: ort.InferenceSession,
    input_name: str,
    inp_tensor: np.ndarray,
    warmup_runs: int,
    timing_runs: int,
) -> tuple[list[float], list[np.ndarray]]:
    warmup_runs = max(0, int(warmup_runs))
    timing_runs = max(1, int(timing_runs))

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: inp_tensor})

    times_ms: list[float] = []
    last_outputs: list[np.ndarray] = []
    for _ in range(timing_runs):
        t0 = time.perf_counter()
        last_outputs = session.run(None, {input_name: inp_tensor})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
    return times_ms, last_outputs


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    image_path = args.image.resolve()

    if not onnx_path.exists():
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return 2
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        return 2

    print(f"Model: {onnx_path}")
    print(f"Image: {image_path}")

    checks, graph_ctx = _collect_graph_checks(onnx_path)
    precision_audit = graph_ctx.get("precision_audit")

    providers = _select_runtime_provider(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers if providers else None)
    inp = session.get_inputs()[0]
    out_names = [o.name for o in session.get_outputs()]

    print("\n=== Runtime IO ===")
    print(f"- Input name: {inp.name}")
    print(f"- Input type: {inp.type}")
    print(f"- Input shape: {_shape_to_str(list(inp.shape))}")
    print(f"- Output names: {out_names}")

    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    inp_tensor, fed_img, layout = _prepare_input_tensor(img_rgb, inp)
    resized_by_script = (fed_img.shape[0] != img_rgb.shape[0]) or (fed_img.shape[1] != img_rgb.shape[1])
    print(
        "- Image HxW: original={}x{}, fed={}x{}, resized_by_script={}".format(
            img_rgb.shape[0],
            img_rgb.shape[1],
            fed_img.shape[0],
            fed_img.shape[1],
            resized_by_script,
        )
    )
    if layout != "NCHW":
        checks.append(
            CheckResult(
                name="Input Layout",
                passed=False,
                detail=f"runtime interpreted input layout as {layout}, expected NCHW",
                required=True,
            )
        )
    else:
        checks.append(
            CheckResult(
                name="Input Layout",
                passed=True,
                detail="runtime interpreted input layout as NCHW",
                required=True,
            )
        )

    times_ms, outputs = _inference_timing_ms(
        session=session,
        input_name=inp.name,
        inp_tensor=inp_tensor,
        warmup_runs=args.warmup_runs,
        timing_runs=args.timing_runs,
    )
    output_summary = _summarize_outputs(out_names, outputs)

    checks.append(
        CheckResult(
            name="Runtime Single Output Tensor",
            passed=len(outputs) == 1,
            detail=f"runtime_output_count={len(outputs)}",
            required=True,
        )
    )
    if outputs:
        out0 = np.asarray(outputs[0])
        checks.append(
            CheckResult(
                name="Runtime Output DType",
                passed=out0.dtype == np.float32,
                detail=f"found={out0.dtype}, expected=float32",
                required=True,
            )
        )
        checks.append(
            CheckResult(
                name="Runtime Output Shape",
                passed=(out0.ndim == 2 and out0.shape[1] == 7),
                detail=f"found={out0.shape}, expected=[N,7]",
                required=True,
            )
        )

    print("\n=== Runtime Output Summary ===")
    for line in output_summary:
        print(f"- {line}")
    checks.extend(_print_precision_audit(precision_audit))
    if times_ms:
        print("\n=== Single-Image Inference Timing ===")
        print(f"- Runs: warmup={max(0, int(args.warmup_runs))}, timed={len(times_ms)}")
        print(
            "- Time (ms): avg={avg:.3f}, min={mn:.3f}, max={mx:.3f}, p95={p95:.3f}".format(
                avg=float(np.mean(times_ms)),
                mn=float(np.min(times_ms)),
                mx=float(np.max(times_ms)),
                p95=float(np.quantile(times_ms, 0.95)),
            )
        )

    detections = _extract_detections(out_names, outputs)
    if detections is not None:
        score = detections.scores
        print("\n=== Parsed Detection Stats ===")
        print(
            "- rows={rows}, batch0_rows={b0}, score_min={mn:.5f}, score_max={mx:.5f}, score_mean={me:.5f}, >thr({thr:.2f})={gt}".format(
                rows=int(detections.boxes_xyxy.shape[0]),
                b0=int(np.sum(detections.batch_idx == 0)),
                mn=float(np.min(score)) if score.size else 0.0,
                mx=float(np.max(score)) if score.size else 0.0,
                me=float(np.mean(score)) if score.size else 0.0,
                thr=float(args.score_threshold),
                gt=int(np.sum(score > float(args.score_threshold))),
            )
        )
        if len(outputs) == 1:
            arr0 = np.asarray(outputs[0])
            arr2 = arr0 if arr0.ndim == 2 else (arr0[0] if arr0.ndim == 3 and arr0.shape[0] == 1 else None)
            if isinstance(arr2, np.ndarray):
                print("- Column stats for consolidated output:")
                for line in _tensor_column_stats(arr2):
                    print(f"  {line}")

        nonzero_scores = bool(np.any(score > 0.0))
        checks.append(
            CheckResult(
                name="Runtime Non-Zero Detection Scores",
                passed=nonzero_scores,
                detail="any(score>0)={}".format(nonzero_scores),
                required=False,
            )
        )
        score01_ok = bool(np.all((score >= -1e-6) & (score <= 1.0 + 1e-6)))
        checks.append(
            CheckResult(
                name="Runtime Score Range [0,1]",
                passed=score01_ok,
                detail=(
                    "score_min={:.5f}, score_max={:.5f} "
                    "(detector scores are usually probabilities after sigmoid/postprocess)"
                ).format(float(np.min(score)), float(np.max(score))),
                required=False,
            )
        )

        cls = detections.class_ids
        if cls.size > 0:
            frac = np.abs(cls - np.round(cls))
            cls_int_like = float(np.mean(frac <= 1e-3)) > 0.9
            checks.append(
                CheckResult(
                    name="Runtime Class IDs Integer-Like",
                    passed=cls_int_like,
                    detail="integer_like_ratio={:.3f}".format(float(np.mean(frac <= 1e-3))),
                    required=False,
                )
            )

    bounds_check = _runtime_box_bounds_check(detections, (fed_img.shape[0], fed_img.shape[1]))
    checks.append(bounds_check)
    scale_check = _runtime_box_scale_coverage_check(detections, (fed_img.shape[0], fed_img.shape[1]))
    checks.append(scale_check)

    required_pass, unknowns = _print_check_table(checks)
    compliant = required_pass

    if compliant:
        print("\nResult: COMPLIANT with required checks.")
    else:
        print("\nResult: NON-COMPLIANT with required checks.")
        print("Detected output format:")
        if detections is not None:
            print(
                f"- Parseable detection format: {detections.source_format}, "
                f"boxes={detections.boxes_xyxy.shape}, scores={detections.scores.shape}, "
                f"class_ids={detections.class_ids.shape}, batch_idx={detections.batch_idx.shape}"
            )
        else:
            print("- Could not parse outputs into detection boxes.")
        print("Raw runtime outputs (shape/format):")
        for line in output_summary:
            print(f"  {line}")

    if unknowns:
        print("\nNote: Some advisory checks were skipped or inconclusive.")

    if args.no_show:
        print("\nInteractive image display skipped (--no-show).")
    else:
        _plot_detections(
            img_rgb=fed_img,
            detections=detections,
            score_threshold=args.score_threshold,
            input_size_hw=(fed_img.shape[0], fed_img.shape[1]),
            model_path=onnx_path,
            fallback_topk=args.fallback_topk,
        )

    if args.strict_exit and not compliant:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
