#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3-Small → QAT → INT8 ONNX (opset-17, full pipeline).

Generally tested on pytorch 2.7.0 (pip installed), CUDA 12.8 (mamba installed first with onnxruntime)
"""

from __future__ import annotations
import json, os, random, warnings
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.v2 as T_v2
import torchvision.models as tvm
import torchvision.datasets as dsets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
from torch.amp import autocast

# FX QAT imports
import torch.nn.functional as F
from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfig
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx as prepare_qat_fx_torch, convert_fx as convert_fx_torch

# onnx imports
import onnx
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
import onnxruntime as ort

warnings.filterwarnings(
    "ignore", message="'.*has_(cuda|cudnn|mps|mkldnn).*is deprecated", module="torch.overrides"
)

SEED, IMG_SIZE, NUM_WORKERS = 42, 224, 0
DUMMY_H, DUMMY_W = IMG_SIZE + 32, IMG_SIZE + 64 # Used for QAT example input
random.seed(SEED); torch.manual_seed(SEED)

_CNL = lambda d: (d.type == "cuda")

# ─────────────────────────── transforms ──────────────────────────
def _build_tf(train: bool):
    if train:
        return T_v2.Compose([
            T_v2.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.67, 1.0), antialias=True),
            T_v2.RandomHorizontalFlip(),
            T_v2.TrivialAugmentWide(),
            T_v2.ToDtype(torch.uint8, scale=False)
        ])
    else: # Validation (train=False)
        # Always resize to a fixed size for batching by the DataLoader
        return T_v2.Compose([
            T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T_v2.ToDtype(torch.uint8, scale=False)
        ])

# ─────────────────────────── dataloader ──────────────────────────
def _rgb_loader(p): return read_image(p, mode=ImageReadMode.RGB)

def get_loader(root_dir: str, batch_size: int, is_train: bool, device: torch.device):
    split = "train" if is_train else "val"
    dataset = dsets.ImageFolder(
        root=os.path.join(root_dir, split),
        transform=_build_tf(is_train), # Pass only is_train flag
        loader=_rgb_loader
    )
    pf = 4 if NUM_WORKERS else None
    pin_memory = device.type == "cuda"
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=NUM_WORKERS,
                    pin_memory=pin_memory, persistent_workers=bool(NUM_WORKERS),
                    prefetch_factor=pf, drop_last=is_train)

# ───────────────────────── modules ───────────────────────────────
class PreprocNorm(nn.Module):
    def __init__(self):
        super().__init__()
        m = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        s = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.register_buffer("m", m, persistent=False)
        self.register_buffer("s", s, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div_(255)
        return (x - self.m) / self.s

def _mnv4_id(kind: str, width: float) -> str:
    base = f"mobilenetv4_conv_{'small' if kind=='mnv4s' else 'medium'}"
    return base if abs(width-1.0) < 1e-6 else f"{base}_{int(width*100):03d}"

def remove_dropout(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Dropout):
            setattr(m, name, nn.Identity())
        else:
            remove_dropout(child)

def get_backbone(arch: str, ncls: int, width: float,
                 pretrained: bool, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0) -> nn.Module:
    """Return backbone ready for train/QAT pipeline."""
    if arch == "mnv3":
        model = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            width_mult=width, dropout=drop_rate)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, ncls)
    elif arch == "mnv2":
        model = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None,
            width_mult=width_mult,
        )
        # 1. Get the input features for the final linear layer
        # The default classifier is Sequential(Dropout, Linear)
        # The in_features for the Linear layer is at classifier[1].in_features
        in_features = model.classifier[1].in_features
        # 2. Reconstruct the classification head with inplace=False for Dropout
        # This is the most robust way when replacing the head for fine-tuning
        new_classifier_layers = []
        if drop_rate is not None and drop_rate > 0.0:
            # Use inplace=False explicitly for better tracing compatibility
            new_classifier_layers.append(nn.Dropout(p=drop_rate, inplace=False))
        
        new_classifier_layers.append(nn.Linear(in_features, ncls))
        
        model.classifier = nn.Sequential(*new_classifier_layers)
    elif arch == "mnv2_old":
        # build the backbone
        model = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None,
            width_mult=width,                  # width multiplier, same idea as v3
        )
    
        # MobileNet V2’s classifier is:  [0] Dropout(0.2)  [1] Linear(last_chan, 1000)
        model.classifier[0] = nn.Dropout(p=drop_rate if drop_rate is not None else 0.2,
                                         inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, ncls)
    elif arch in {"mnv4s", "mnv4m"}:
        import timm

        timm_name = _mnv4_id(arch, width)
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=ncls,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )  # timm provides official-reproduction weights :contentReference[oaicite:0]{index=0}
    else:
        raise ValueError(f"Unknown arch '{arch}'")
    return model

def build_model(
        ncls, width_mult_val, dev, arch="mnv3",
        pretrained: bool=False, drop_rate: float = 0.2,  # 0.20
        drop_path_rate: float = 0.0,  # 0.075
    ):
    backbone = get_backbone(arch, ncls, width=width_mult_val,
                        pretrained=pretrained,
                        drop_rate=drop_rate,
                        drop_path_rate=drop_path_rate)
    # Model's first layer is a Resize transform.
    # This will operate on batches of images already resized to IMG_SIZE by the DataLoader's transforms.
    model = nn.Sequential(
        T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        PreprocNorm(),
        backbone
    ).to(
        dev, memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format
    )
    return model

# ───────────────────── training helpers ─────────────────────────
def train_epoch(model, loader, crit, opt, scaler, dev, ep, qat_mode_active:bool = False):
    model.train(); tot = loss_sum = 0
    for i, (img, lab) in enumerate(loader):
        img = img.to(dev, non_blocking=True,
                     memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format)
        lab = lab.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if not qat_mode_active and dev.type == "cuda":
            with autocast(device_type=dev.type, enabled=True):
                out = model(img); loss = crit(out, lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            out = model(img); loss = crit(out, lab)
            loss.backward(); opt.step()

        tot += img.size(0); loss_sum += loss.item() * img.size(0)
        if i == 0 and ep == 0:
            print(f"[{'QAT ' if qat_mode_active else ''}Batch-0] input shape {img.shape} input dtype {img.dtype} loss {loss.item():.4f}")
    return loss_sum / tot

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); corr = tot = 0
    for img, lab in loader: # img is now guaranteed to be (B, C, IMG_SIZE, IMG_SIZE)
        img = img.to(dev, memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format)
        lab = lab.to(dev); corr += (model(img).argmax(1) == lab).sum().item(); tot += lab.size(0)
    return corr / tot

# ───────────────────────── QAT PREPARATION ─────────────────────────
def get_qat_model_fx(model_fp32_cpu: nn.Module, example_inputs_cpu: tuple):
    model_fp32_cpu.eval()
    qconfig_mapping = get_default_qat_qconfig_mapping("x86") # "x86" can be "fbgemm" or "qnnpack" too
    weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    global_qat_qconfig = QConfig(activation=qconfig_mapping.global_qconfig.activation, weight=weight_observer)
    qconfig_mapping = qconfig_mapping.set_global(global_qat_qconfig)
    prepared_model = prepare_qat_fx_torch(model_fp32_cpu, qconfig_mapping, example_inputs_cpu)
    return prepared_model

data_dir: str = "filtered_imagenet2_native"
epochs: int = 2
qat_epochs: int = 2
batch: int = 64
lr: float = 0.025
qat_lr_factor: float = 0.05
width_mult: float = 1.0
device = None
compile_model: bool = False
arch: str = "mnv2"  # mnv4m, mnv3, mnv2, mnv4s
pretrained: bool = True
drop_rate = 0.0

if device is None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if device == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
dev = torch.device(device)
print(f"[INFO] device: {dev}")

pretrain_str = "_pretrained" if pretrained else ""
base_path = f"mobilenet_w{str(width_mult).replace('.', '_')}_{arch}{pretrain_str}_drp{str(drop_rate).replace('.', '_')}_int8_fullpipe"
pt_path = base_path+ ".pt"
onnx_fp32_path = f"{base_path}_fp32.onnx"


with open(Path(data_dir) / "class_mapping.json") as f:
    ncls = len(json.load(f)); print(f"[INFO] #classes = {ncls}")
print(f"[INFO] #classes = {ncls}")

tr = get_loader(data_dir, batch, True, dev)
vl = get_loader(data_dir, batch, False, dev)

model = build_model(ncls, width_mult, dev, arch=arch, pretrained=pretrained, drop_rate=drop_rate)

crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler() if dev.type == "cuda" else torch.amp.GradScaler(enabled=False)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

print("[INFO] Starting FP32 training...")
for ep in range(epochs):
    l = train_epoch(model, tr, crit, opt, scaler, dev, ep, qat_mode_active=False)
    a = evaluate(model, vl, dev)
    sched.step()
    print(f"Epoch {ep+1}/{epochs}  loss {l:.4f}  val@1 {a*100:.2f}%  lr {opt.param_groups[0]['lr']:.5f}")

print("[INFO] Extracting FP32 backbone state_dict...")
try:
    fp32_backbone_state_dict = {}
    model.cpu()
    full_sd = model.state_dict()

    key_prefix = "2.classifier." if arch in ["mnv3", "mnv2"] else "2.head." if arch in ["mnv4s", "mnv4m"] else None
    backbone_prefix = "2."

    if key_prefix:
        for k, v in full_sd.items():
            if k.startswith(backbone_prefix) and not k.startswith(key_prefix):
                fp32_backbone_state_dict[k[len(backbone_prefix):]] = v
        if fp32_backbone_state_dict:
            bpath = f"{pt_path}_fp32_backbone.pt"
            Path(bpath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(fp32_backbone_state_dict, bpath)
            print(f"[SAVE] FP32 PyTorch backbone state_dict → {bpath}")
except Exception as e:
    print("failed to export fp32 backbone: " + repr(e))

print("[INFO] Preparing model for QAT...")
qat_train_device = torch.device("cpu") if dev.type == "mps" else dev
# patch to remove dropout
_orig_dropout = F.dropout
F.dropout = lambda x, p=0.0, training=False, inplace=False: x
model_fp32_cpu = model.cpu()
remove_dropout(model_fp32_cpu)
example_inputs_cpu = (torch.randint(0, 256, (1, 3, DUMMY_H, DUMMY_W), dtype=torch.uint8),)
qat_prepared_model = get_qat_model_fx(model_fp32_cpu, example_inputs_cpu)
# ✅ Restore F.dropout so that actual QAT training uses real dropout (if needed)
F.dropout = _orig_dropout
qat_model = qat_prepared_model.to(qat_train_device)
opt_q = optim.SGD(qat_model.parameters(), lr=lr * qat_lr_factor, momentum=0.9)

print(f"[INFO] Starting QAT fine-tuning on {qat_train_device}")
for qep in range(qat_epochs):
    l_q = train_epoch(qat_model, tr, crit, opt_q, scaler, qat_train_device, qep, qat_mode_active=True)
    val_acc_q = evaluate(qat_model, vl, qat_train_device)
    print(f"[QAT] Epoch {qep+1}/{qat_epochs} loss {l_q:.4f} val@1 {val_acc_q*100:.2f}%")

qat_model.cpu().eval()
qat_model_for_export = qat_model.cpu().eval()
qat_model_for_export.apply(torch.ao.quantization.disable_fake_quant)
qat_model_for_export.apply(torch.ao.quantization.disable_observer)
torch.save(qat_model_for_export.state_dict(), "qat_model.pth") # Save the state_dict
print("[INFO] QAT model state_dict saved to qat_model.pth")
fp32_model_for_onnx_export = build_model(ncls, width_mult, torch.device("cpu"), arch=arch, pretrained=False, drop_rate=drop_rate).cpu().eval()
fp32_model_for_onnx_export.load_state_dict(torch.load("qat_model.pth")) # Load the state_dict
print("[INFO] QAT model state_dict loaded.")

int8_model_final = convert_fx_torch(qat_model)

if compile_model:
    print("[INFO] Compiling INT8 model with torch.compile...")
    try:
        base_model = int8_model_final.module() if hasattr(int8_model_final, 'module') else int8_model_final
        int8_model_final = torch.compile(base_model.cpu(), mode="reduce-overhead")
    except Exception as e:
        print(f"[WARN] torch.compile() failed – {e}")

final_model_state = int8_model_final.module() if hasattr(int8_model_final, 'module') else int8_model_final
torch.save(final_model_state.cpu().state_dict(), pt_path)
print(f"[SAVE] INT8 PyTorch model state_dict → {pt_path}")

onnx_path = pt_path.replace(".pt", ".onnx")
print(f"[INFO] Exporting INT8 model to ONNX: {onnx_path}")
dummy_input_onnx_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
model_for_export = int8_model_final.cpu().eval()

try:
    torch.onnx.export(
        model_for_export,
        (dummy_input_onnx_cpu,),
        onnx_path,
        input_names=["input_u8"],
        output_names=["logits"],
        dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[SAVE] INT8 ONNX model → {onnx_path}")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {repr(e)}")
    import traceback
    traceback.print_exc()

try:
    path = "fp32_onnx.onnx"
    torch.onnx.export(
        model,
        (dummy_input_onnx_cpu,),
        path,
        input_names=["input_u8"],
        output_names=["logits"],
        dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[SAVE] ONNX model → {path}")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {repr(e)}")
    import traceback
    traceback.print_exc()

# STATIC QUANTIZATION FROM ONNX to ONNX

# --- Configuration ---
data_dir = "filtered_imagenet2_native" # Make sure this path is correct
batch_size = 32 # Use a larger batch size for calibration for efficiency
num_calibration_batches = 100 # Number of batches to use for calibration
onnx_int8_path = "mobilenet_v2_int8_ort_quantized.onnx"
provider = "CPUExecutionProvider" # Or "CUDAExecutionProvider"
# ... (after INT8 ONNX export from PyTorch QAT model)

# --- Configuration for ONNX Runtime PTQ ---
# data_dir, batch_size, num_calibration_batches, onnx_int8_path, provider are already defined

print("[INFO] Preparing calibration data for ONNX Runtime quantization...")
# Calibration loader can run on CPU, ensure it yields uint8 tensors
calibration_loader = get_loader(data_dir, batch_size, is_train=False, device=torch.device("cpu"))

# Define a data reader for ONNX Runtime quantizer
class DataReader:
    def __init__(self, dataloader, num_batches, input_name: str): # input_name is still good for flexibility if this assumption is wrong
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.iterator = iter(dataloader)
        self.count = 0
        self.input_name = input_name
        print(f"[DataReader INFO] Initialized to provide data for input key: '{self.input_name}'")


    def get_next(self):
        if self.count < self.num_batches:
            try:
                batch, _ = next(self.iterator) # batch should be uint8 from get_loader
                self.count += 1
                # ONNX Runtime expects input as a dictionary of numpy arrays
                return {self.input_name: batch.numpy()}
            except StopIteration:
                print(f"[WARN] Calibration data exhausted after {self.count} batches. Using available batches.")
                return None
        else:
            return None

# FORCED INPUT NAME FOR CALIBRATOR:
# The error message strongly suggests that the ORT Calibrator, when quantizing
# for INT8/UINT8 activations, expects the calibration data feed to use the key "input_u8".
calibrator_expected_input_key = "input_u8"
print(f"[INFO] Using input key '{calibrator_expected_input_key}' for ONNX Runtime calibration data reader.")

calibration_data_reader = DataReader(calibration_loader, num_calibration_batches, input_name=calibrator_expected_input_key)
print(f"[INFO] Using {num_calibration_batches} batches for calibration (batch size {batch_size}).")

# Also, add a debug print to confirm the input name of the onnx_fp32_path model itself
try:
    loaded_onnx_fp32_model = onnx.load(path)
    if loaded_onnx_fp32_model.graph.input:
        actual_model_input_name = loaded_onnx_fp32_model.graph.input[0].name
        print(f"[DEBUG] The FP32 ONNX model at '{onnx_fp32_path}' has input node named: '{actual_model_input_name}'.")
    else:
        print(f"[DEBUG] The FP32 ONNX model at '{onnx_fp32_path}' has no graph.input defined.")
except Exception as e_load:
    print(f"[DEBUG] Could not load or inspect '{onnx_fp32_path}': {e_load}")


print(f"[INFO] Performing ONNX Runtime static quantization on {onnx_fp32_path}...")
try:
    onnx_model_optimized_path = path.replace(".onnx", "_optimized.onnx")
    sess_options = ort.SessionOptions()
    # Set graph optimization level
    # ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Try ALL for more aggressive opts
    
    # Set up the output path for the optimized model
    sess_options.optimized_model_filepath = onnx_model_optimized_path
    
    # Create a session with the model and options to trigger optimization and save.
    # We don't actually need to run inference here.
    _ = ort.InferenceSession(path, sess_options, providers=['CPUExecutionProvider'])
    
    if os.path.exists(onnx_model_optimized_path):
        print(f"[INFO] Optimized ONNX model saved to: {onnx_model_optimized_path}")
        # Update the path to be used for quantization
        onnx_model_path_for_quantization = onnx_model_optimized_path
    else:
        print(f"[WARN] Optimized model was not saved to {onnx_model_optimized_path}. Using previous model path.")

    quantize_static(
        onnx_model_optimized_path,
        onnx_int8_path,
        calibration_data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8 # This likely triggers the "input_u8" expectation
    )
    print(f"[INFO] ONNX Runtime INT8 quantized model saved to {onnx_int8_path}")

    onnx_quant_model = onnx.load(onnx_int8_path)
    onnx.checker.check_model(onnx_quant_model)
    print("[INFO] INT8 ONNX model check passed.")

except Exception as e:
    print(f"[ERROR] Failed to quantize ONNX model with ONNX Runtime: {e}")
    import traceback
    traceback.print_exc()
    # exit()

print("[DONE]")