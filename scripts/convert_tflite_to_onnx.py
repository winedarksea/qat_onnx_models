#!/usr/bin/env python3
"""
Convert the Vertex AI TFLite detector to a guideline-compatible ONNX model.

Target contract (scripts/ONNX_GUIDELINES.md + scripts/onnx_guideline_model_test.py):
- Input: uint8 NCHW [1, 3, H, W]
- Preprocess in-graph: Cast + Resize + normalization arithmetic
- Output: single float32 [N, 7] tensor with:
  [x1, y1, x2, y2, score, class_id, batch_idx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Convert maip Vertex AI TFLite model to guideline ONNX.")
    p.add_argument(
        "--tflite",
        type=Path,
        default=repo_root / "model_saves" / "maip_vertexai_2nodehrs_medium.tflite",
        help="Path to input TFLite model.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "model_saves" / "maip_vertexai_2nodehrs_medium_guideline.onnx",
        help="Path to final ONNX model.",
    )
    p.add_argument(
        "--raw-onnx",
        type=Path,
        default=repo_root / "model_saves" / "maip_vertexai_2nodehrs_medium_raw_tf2onnx.onnx",
        help="Intermediate ONNX path produced by tf2onnx.",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset to use for tf2onnx conversion.")
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the intermediate tf2onnx model instead of deleting it.",
    )
    return p.parse_args()


def _shape_from_vi(vi) -> list[int | str]:
    dims: list[int | str] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def _find_detection_output_names(graph) -> tuple[str, str, str]:
    outputs = list(graph.output)
    if len(outputs) < 3:
        raise RuntimeError(f"Expected at least 3 outputs from raw model, found {len(outputs)}")

    boxes_name = None
    vec_names: list[str] = []
    boxes_len = None

    for out in outputs:
        shape = _shape_from_vi(out)
        if len(shape) == 3 and isinstance(shape[2], int) and shape[2] == 4 and boxes_name is None:
            boxes_name = out.name
            boxes_len = shape[1]

    if boxes_name is None:
        raise RuntimeError("Could not find box output [1, N, 4] in raw ONNX outputs.")

    for out in outputs:
        shape = _shape_from_vi(out)
        if len(shape) == 2 and boxes_len is not None and shape[1] == boxes_len:
            vec_names.append(out.name)

    if len(vec_names) < 2:
        raise RuntimeError(
            "Could not find class/score outputs with shape [1, N]. "
            f"Found candidates: {vec_names}"
        )

    class_name = next((n for n in vec_names if n.endswith(":1")), vec_names[0])
    score_name = next((n for n in vec_names if n.endswith(":2")), None)
    if score_name is None or score_name == class_name:
        score_name = vec_names[1] if vec_names[0] == class_name else vec_names[0]

    return boxes_name, class_name, score_name


def _inject_guideline_io(raw_onnx: Path, output_onnx: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(raw_onnx))
    graph = model.graph

    if len(graph.input) != 1:
        raise RuntimeError(f"Expected exactly 1 input in raw ONNX, found {len(graph.input)}")

    orig_input = graph.input[0]
    orig_in_name = orig_input.name
    orig_in_shape = _shape_from_vi(orig_input)
    if len(orig_in_shape) != 4:
        raise RuntimeError(f"Expected NHWC rank-4 raw input, found shape {orig_in_shape}")
    if not isinstance(orig_in_shape[1], int) or not isinstance(orig_in_shape[2], int):
        raise RuntimeError(f"Expected static internal size from raw input, found shape {orig_in_shape}")
    target_h = int(orig_in_shape[1])
    target_w = int(orig_in_shape[2])

    boxes_name, class_name, score_name = _find_detection_output_names(graph)

    new_input_name = "images_uint8"
    new_input = helper.make_tensor_value_info(
        new_input_name, TensorProto.UINT8, [1, 3, "input_h", "input_w"]
    )

    pp = "maip_pp"
    initializers = [
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), name=f"{pp}_zero"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), name=f"{pp}_one"),
        numpy_helper.from_array(np.array([], dtype=np.float32), name=f"{pp}_resize_roi"),
        numpy_helper.from_array(np.array([], dtype=np.float32), name=f"{pp}_resize_scales"),
        numpy_helper.from_array(
            np.array([1, 3, target_h, target_w], dtype=np.int64),
            name=f"{pp}_resize_sizes",
        ),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), name=f"{pp}_clip_min"),
        numpy_helper.from_array(np.array([255.0], dtype=np.float32), name=f"{pp}_clip_max"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name=f"{pp}_axes0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{pp}_axes1"),
        numpy_helper.from_array(np.array([1, 0, 3, 2], dtype=np.int64), name=f"{pp}_xyxy_gather"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name=f"{pp}_shape_idx_h"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name=f"{pp}_shape_idx_w"),
    ]
    graph.initializer.extend(initializers)

    pre_nodes = [
        helper.make_node(
            "Cast",
            [new_input_name],
            [f"{pp}_cast_f32"],
            name=f"{pp}_CastUint8ToFloat",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Sub",
            [f"{pp}_cast_f32", f"{pp}_zero"],
            [f"{pp}_norm_sub"],
            name=f"{pp}_NormSub",
        ),
        helper.make_node(
            "Mul",
            [f"{pp}_norm_sub", f"{pp}_one"],
            [f"{pp}_norm_mul"],
            name=f"{pp}_NormMul",
        ),
        helper.make_node(
            "Resize",
            [f"{pp}_norm_mul", f"{pp}_resize_roi", f"{pp}_resize_scales", f"{pp}_resize_sizes"],
            [f"{pp}_resized_nchw_f32"],
            name=f"{pp}_ResizeToModelInput",
            mode="linear",
            nearest_mode="floor",
        ),
        helper.make_node(
            "Clip",
            [f"{pp}_resized_nchw_f32", f"{pp}_clip_min", f"{pp}_clip_max"],
            [f"{pp}_resized_nchw_u8_clip"],
            name=f"{pp}_ClipU8Range",
        ),
        helper.make_node(
            "Cast",
            [f"{pp}_resized_nchw_u8_clip"],
            [f"{pp}_resized_nchw_u8"],
            name=f"{pp}_CastToUint8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "Transpose",
            [f"{pp}_resized_nchw_u8"],
            [orig_in_name],
            name=f"{pp}_TransposeNCHWToNHWC",
            perm=[0, 2, 3, 1],
        ),
    ]

    post_nodes = [
        helper.make_node(
            "Squeeze",
            [boxes_name, f"{pp}_axes0"],
            [f"{pp}_boxes_yxyx"],
            name=f"{pp}_SqueezeBoxes",
        ),
        helper.make_node(
            "Gather",
            [f"{pp}_boxes_yxyx", f"{pp}_xyxy_gather"],
            [f"{pp}_boxes_xyxy_norm"],
            name=f"{pp}_ReorderBoxesToXYXY",
            axis=1,
        ),
        helper.make_node(
            "Squeeze",
            [score_name, f"{pp}_axes0"],
            [f"{pp}_scores"],
            name=f"{pp}_SqueezeScores",
        ),
        helper.make_node(
            "Squeeze",
            [class_name, f"{pp}_axes0"],
            [f"{pp}_classes_raw"],
            name=f"{pp}_SqueezeClasses",
        ),
        helper.make_node(
            "Cast",
            [f"{pp}_classes_raw"],
            [f"{pp}_classes"],
            name=f"{pp}_CastClassesFloat",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Shape",
            [new_input_name],
            [f"{pp}_in_shape"],
            name=f"{pp}_InputShape",
        ),
        helper.make_node(
            "Gather",
            [f"{pp}_in_shape", f"{pp}_shape_idx_h"],
            [f"{pp}_h_i64"],
            name=f"{pp}_GatherH",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            [f"{pp}_in_shape", f"{pp}_shape_idx_w"],
            [f"{pp}_w_i64"],
            name=f"{pp}_GatherW",
            axis=0,
        ),
        helper.make_node(
            "Cast",
            [f"{pp}_h_i64"],
            [f"{pp}_h_f"],
            name=f"{pp}_CastHFloat",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Cast",
            [f"{pp}_w_i64"],
            [f"{pp}_w_f"],
            name=f"{pp}_CastWFloat",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Concat",
            [f"{pp}_w_f", f"{pp}_h_f", f"{pp}_w_f", f"{pp}_h_f"],
            [f"{pp}_box_scale_1d"],
            name=f"{pp}_ConcatScale",
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{pp}_box_scale_1d", f"{pp}_axes0"],
            [f"{pp}_box_scale_2d"],
            name=f"{pp}_UnsqueezeScale",
        ),
        helper.make_node(
            "Mul",
            [f"{pp}_boxes_xyxy_norm", f"{pp}_box_scale_2d"],
            [f"{pp}_boxes_xyxy"],
            name=f"{pp}_ScaleBoxesToInputSpace",
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{pp}_scores", f"{pp}_axes1"],
            [f"{pp}_score_col"],
            name=f"{pp}_UnsqueezeScores",
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{pp}_classes", f"{pp}_axes1"],
            [f"{pp}_class_col"],
            name=f"{pp}_UnsqueezeClasses",
        ),
        helper.make_node(
            "Shape",
            [f"{pp}_scores"],
            [f"{pp}_score_shape"],
            name=f"{pp}_ScoreShape",
        ),
        helper.make_node(
            "ConstantOfShape",
            [f"{pp}_score_shape"],
            [f"{pp}_batch_idx"],
            name=f"{pp}_BatchIdxZeros",
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{pp}_batch_idx", f"{pp}_axes1"],
            [f"{pp}_batch_col"],
            name=f"{pp}_UnsqueezeBatchIdx",
        ),
        helper.make_node(
            "Concat",
            [f"{pp}_boxes_xyxy", f"{pp}_score_col", f"{pp}_class_col", f"{pp}_batch_col"],
            ["detections"],
            name=f"{pp}_ConcatDetections",
            axis=1,
        ),
    ]

    del graph.input[:]
    graph.input.append(new_input)
    for node in reversed(pre_nodes):
        graph.node.insert(0, node)
    graph.node.extend(post_nodes)

    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("detections", TensorProto.FLOAT, ["num_detections", 7])
    )

    onnx.checker.check_model(model)
    onnx.save(model, str(output_onnx))


def main() -> int:
    args = parse_args()
    tflite_path = args.tflite.resolve()
    output_path = args.output.resolve()
    raw_onnx_path = args.raw_onnx.resolve()

    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    from tf2onnx import convert

    print(f"[INFO] Converting TFLite -> raw ONNX (opset {args.opset})")
    print(f"       input : {tflite_path}")
    print(f"       raw   : {raw_onnx_path}")
    convert.from_tflite(str(tflite_path), opset=int(args.opset), output_path=str(raw_onnx_path))

    print("[INFO] Injecting guideline-compatible IO wrapper")
    _inject_guideline_io(raw_onnx_path, output_path)
    print(f"[INFO] Saved final ONNX: {output_path}")

    if not args.keep_raw and raw_onnx_path.exists():
        raw_onnx_path.unlink()
        print(f"[INFO] Removed intermediate ONNX: {raw_onnx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
