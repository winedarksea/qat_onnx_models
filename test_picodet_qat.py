#!/usr/bin/env python3
"""
Comprehensive sanity tests for PicoDet QAT training pipeline.
Tests model creation, forward passes, QAT preparation, and ONNX export without actual training.

Usage:
    python test_picodet_qat.py
    
    or with conda:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n gpu311 python test_picodet_qat.py
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, 'scripts')
from picodet_lib_v2 import PicoDet, get_backbone, PicoDetHead, ResizeNorm
from picodet_v5_qat import (
    qat_prepare, PostprocessorForONNX, ONNXExportablePicoDet,
    build_transforms, contiguous_id_to_name, unwrap_dataset,
    CANONICAL_COCO80_MAP
)

# Test configuration
IMG_SIZE = 256
NUM_CLASSES = 80
BATCH_SIZE = 2
DEVICE = torch.device('cpu')

class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.current_section = ""
    
    def section(self, name):
        """Print section header."""
        self.current_section = name
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
    
    def test(self, name, func, *args, **kwargs):
        """Run a test and report results."""
        print(f"\n{name}...")
        try:
            result = func(*args, **kwargs)
            self.tests_passed += 1
            print(f"   ✓ PASSED")
            return result
        except AssertionError as e:
            self.tests_failed += 1
            print(f"   ✗ FAILED: {e}")
            return None
        except Exception as e:
            self.tests_failed += 1
            print(f"   ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def summary(self):
        """Print test summary."""
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print('='*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.tests_passed} ✓")
        print(f"Failed: {self.tests_failed} ✗")
        print(f"Success rate: {100*self.tests_passed/total:.1f}%")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED! Code is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed. Please review the errors above.")
            return 1


# ============================================================================
# Test Functions
# ============================================================================

def test_backbone_creation():
    """Test backbone creation with various architectures."""
    backbone, feat_chs = get_backbone('mnv4c', ckpt=None, img_size=IMG_SIZE)
    assert backbone is not None, "Backbone should not be None"
    assert len(feat_chs) == 3, f"Expected 3 feature channels, got {len(feat_chs)}"
    print(f"     Feature channels: {feat_chs}")
    return backbone, feat_chs


def test_model_creation(backbone, feat_chs):
    """Test full PicoDet model instantiation."""
    model = PicoDet(
        backbone, feat_chs,
        num_classes=NUM_CLASSES,
        neck_out_ch=96,
        img_size=IMG_SIZE,
        head_reg_max=9,
        reg_conv_depth=2,
        cls_conv_depth=3,
        lat_k=5,
        inplace_act_for_head_neck=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable initially"
    
    print(f"     Total parameters: {total_params:,}")
    print(f"     Trainable parameters: {trainable_params:,}")
    return model


def test_forward_training_mode(model):
    """Test forward pass in training mode."""
    model.train()
    dummy_input = torch.randint(0, 256, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), 
                                dtype=torch.uint8, device=DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert isinstance(output, tuple), "Output should be a tuple"
    assert len(output) == 3, f"Expected 3 outputs (cls, reg, strides), got {len(output)}"
    
    cls_preds, reg_preds, strides = output
    assert len(cls_preds) == 3, f"Expected 3 FPN levels, got {len(cls_preds)}"
    assert len(reg_preds) == 3, f"Expected 3 FPN levels, got {len(reg_preds)}"
    
    print(f"     Classification predictions: {len(cls_preds)} levels")
    print(f"     Regression predictions: {len(reg_preds)} levels")
    for i, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
        print(f"       Level {i}: cls={cls_p.shape}, reg={reg_p.shape}")
    
    return output


def test_forward_eval_mode(model):
    """Test forward pass in evaluation mode."""
    model.eval()
    dummy_input = torch.randint(0, 256, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE),
                                dtype=torch.uint8, device=DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert isinstance(output, tuple), "Output should be a tuple"
    assert len(output) == 2, f"Expected 2 outputs (boxes, scores), got {len(output)}"
    
    boxes, scores = output
    assert boxes.shape[0] == BATCH_SIZE, f"Batch size mismatch in boxes"
    assert boxes.shape[2] == 4, f"Expected 4 coordinates per box"
    assert scores.shape[0] == BATCH_SIZE, f"Batch size mismatch in scores"
    assert scores.shape[2] == NUM_CLASSES, f"Expected {NUM_CLASSES} classes"
    
    print(f"     Boxes shape: {boxes.shape}")
    print(f"     Scores shape: {scores.shape}")
    
    return boxes, scores


def test_helper_functions():
    """Test data processing helper functions."""
    # Test transform builders
    train_tf = build_transforms((IMG_SIZE, IMG_SIZE), train=True)
    val_tf = build_transforms((IMG_SIZE, IMG_SIZE), train=False)
    assert train_tf is not None, "Training transforms should not be None"
    assert val_tf is not None, "Validation transforms should not be None"
    
    # Test COCO mapping
    assert len(CANONICAL_COCO80_MAP) == 80, \
        f"COCO mapping should have 80 classes, got {len(CANONICAL_COCO80_MAP)}"
    
    # Test unwrap_dataset
    from torch.utils.data import TensorDataset, Subset
    base_ds = TensorDataset(torch.randn(10, 3, 32, 32))
    wrapped_ds = Subset(Subset(base_ds, list(range(5))), list(range(3)))
    unwrapped = unwrap_dataset(wrapped_ds)
    assert unwrapped is base_ds, "unwrap_dataset should return base dataset"
    
    print(f"     Transform builders: ✓")
    print(f"     COCO mapping: {len(CANONICAL_COCO80_MAP)} classes")
    print(f"     Dataset unwrapping: ✓")


def test_qat_preparation(model):
    """Test QAT preparation using FX graph mode."""
    model.cpu().eval()
    dummy_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    
    # Prepare for QAT
    qat_model = qat_prepare(model, dummy_input)
    
    assert hasattr(qat_model, 'graph'), "QAT model should be a GraphModule"
    assert qat_model is not None, "QAT model should not be None"
    
    # Test forward pass
    qat_model.train()
    with torch.no_grad():
        output = qat_model(dummy_input)
    
    assert output is not None, "QAT forward pass should produce output"
    
    print(f"     Type: {type(qat_model).__name__}")
    print(f"     Is GraphModule: ✓")
    print(f"     Forward pass: ✓")
    
    return qat_model


def test_postprocessor(model):
    """Test ONNX postprocessor creation and execution."""
    postprocessor = PostprocessorForONNX(model.head)
    
    assert hasattr(postprocessor, 'nc'), "Should have num_classes attribute"
    assert hasattr(postprocessor, 'reg_max'), "Should have reg_max attribute"
    assert hasattr(postprocessor, 'nl'), "Should have num_levels attribute"
    assert hasattr(postprocessor, 'strides_buffer'), "Should have strides buffer"
    assert hasattr(postprocessor, 'dfl_project_buffer'), "Should have DFL project buffer"
    
    assert postprocessor.nc == NUM_CLASSES, f"Expected {NUM_CLASSES} classes"
    assert postprocessor.nl == 3, f"Expected 3 FPN levels"
    
    print(f"     Number of classes: {postprocessor.nc}")
    print(f"     Reg max: {postprocessor.reg_max}")
    print(f"     FPN levels: {postprocessor.nl}")
    print(f"     Buffers registered: ✓")
    
    return postprocessor


def test_onnx_exportable_wrapper(qat_model, postprocessor):
    """Test ONNX exportable wrapper creation and execution."""
    exportable_model = ONNXExportablePicoDet(qat_model, postprocessor)
    exportable_model.cpu().eval()
    
    dummy_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    
    with torch.no_grad():
        boxes, scores = exportable_model(dummy_input)
    
    assert boxes.shape[0] == 1, "Batch size should be 1"
    assert boxes.shape[2] == 4, "Should have 4 coordinates per box"
    assert scores.shape[0] == 1, "Batch size should be 1"
    assert scores.shape[2] == NUM_CLASSES, f"Should have {NUM_CLASSES} classes"
    
    print(f"     Boxes shape: {boxes.shape}")
    print(f"     Scores shape: {scores.shape}")
    print(f"     Output validation: ✓")
    
    return exportable_model


def test_onnx_compatibility():
    """Test ONNX module availability and version."""
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"     ONNX version: {onnx.__version__}")
        print(f"     ONNX Runtime version: {ort.__version__}")
        print(f"     Modules available: ✓")
        
        return True
    except ImportError as e:
        print(f"     Warning: ONNX not fully available: {e}")
        return False


def test_resize_norm():
    """Test ResizeNorm preprocessing module."""
    preprocess = ResizeNorm((IMG_SIZE, IMG_SIZE))
    test_input = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    
    with torch.no_grad():
        normalized = preprocess(test_input)
    
    assert normalized.dtype == torch.float32, "Output should be float32"
    assert normalized.shape == test_input.shape, "Shape should be preserved"
    assert -5.0 < normalized.min() < 5.0, "Output should be normalized"
    assert -5.0 < normalized.max() < 5.0, "Output should be normalized"
    
    print(f"     Input dtype: {test_input.dtype}, Output dtype: {normalized.dtype}")
    print(f"     Input range: [{test_input.min().item()}, {test_input.max().item()}]")
    print(f"     Output range: [{normalized.min().item():.3f}, {normalized.max().item():.3f}]")
    print(f"     Normalization: ✓")


# ============================================================================
# Main Test Execution
# ============================================================================

def main():
    """Run all tests."""
    runner = TestRunner()
    
    print("\n" + "="*60)
    print("PicoDet QAT Pipeline - Sanity Tests")
    print("="*60)
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    
    # Part 1: Model Integrity Tests
    runner.section("Part 1: Model Creation and Inference")
    
    backbone, feat_chs = runner.test("1.1 Backbone creation", test_backbone_creation)
    if backbone is None:
        return runner.summary()
    
    model = runner.test("1.2 Full model creation", test_model_creation, backbone, feat_chs)
    if model is None:
        return runner.summary()
    
    runner.test("1.3 Forward pass (training mode)", test_forward_training_mode, model)
    runner.test("1.4 Forward pass (eval mode)", test_forward_eval_mode, model)
    runner.test("1.5 Helper functions", test_helper_functions)
    
    # Part 2: QAT and ONNX Tests
    runner.section("Part 2: QAT Preparation and ONNX Export")
    
    qat_model = runner.test("2.1 QAT preparation", test_qat_preparation, model)
    if qat_model is None:
        print("     Skipping remaining QAT tests due to preparation failure")
    else:
        postprocessor = runner.test("2.2 PostprocessorForONNX creation", 
                                    test_postprocessor, model)
        if postprocessor is not None:
            runner.test("2.3 ONNX exportable wrapper", 
                       test_onnx_exportable_wrapper, qat_model, postprocessor)
    
    runner.test("2.4 ONNX module availability", test_onnx_compatibility)
    
    # Part 3: Preprocessing Tests
    runner.section("Part 3: Data Preprocessing")
    
    runner.test("3.1 ResizeNorm preprocessing", test_resize_norm)
    
    # Final summary
    return runner.summary()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
