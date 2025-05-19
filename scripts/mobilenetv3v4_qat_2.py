
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3-Small → QAT → INT8 ONNX (opset-17, full pipeline).

Generally tested on pytorch 2.7.0 (pip installed), CUDA 12.8 (mamba installed first with onnxruntime)
"""

from __future__ import annotations
import argparse, json, os, random, warnings
from pathlib import Path
from typing import List

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
        transform=_build_tf(is_train),
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

def get_backbone(arch: str, ncls: int, width: float,
                 pretrained: bool, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0) -> nn.Module:
    if arch == "mnv3":
        model = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            width_mult=width, dropout=drop_rate)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, ncls)
    elif arch == "mnv2":
        model = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None,
            width_mult=width,
        )
        model.classifier[0] = nn.Dropout(p=drop_rate if drop_rate is not None else 0.2, inplace=True)
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
        )
    else:
        raise ValueError(f"Unknown arch '{arch}'")
    return model

def build_model(
        ncls, width_mult_val, dev, arch="mnv3",
        pretrained: bool=False, drop_rate: float = 0.2,
        drop_path_rate: float = 0.0,
    ):
    backbone = get_backbone(arch, ncls, width=width_mult_val,
                        pretrained=pretrained,
                        drop_rate=drop_rate,
                        drop_path_rate=drop_path_rate)
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

        if not qat_mode_active and dev.type == "cuda": # Standard FP32/AMP training
            with autocast(device_type=dev.type, enabled=True):
                out = model(img); loss = crit(out, lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: # QAT training (or CPU FP32 training)
              # AMP is generally not recommended or as beneficial with QAT on CPU
              # For QAT on CUDA, some setups might use it, but often it's disabled
              # to avoid interference with quantization numerics.
            out = model(img); loss = crit(out, lab)
            loss.backward(); opt.step()

        tot += img.size(0); loss_sum += loss.item() * img.size(0)
        if i == 0 and ep == 0:
            print(f"[{'QAT ' if qat_mode_active else ''}Batch-0] input shape {img.shape} input dtype {img.dtype} loss {loss.item():.4f}")
    return loss_sum / tot

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); corr = tot = 0
    for img, lab in loader:
        img = img.to(dev, memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format)
        lab = lab.to(dev); corr += (model(img).argmax(1) == lab).sum().item(); tot += lab.size(0)
    return corr / tot

# ───────────────────────── QAT PREPARATION ─────────────────────────
def get_qat_model_fx_old(model_fp32_eval_cpu: nn.Module, example_inputs_cpu: tuple):
    # model_fp32_eval_cpu is assumed to be in .eval() mode
    qconfig_mapping = get_default_qat_qconfig_mapping("x86") # "x86" (FBGEMM) or "qnnpack"
    
    # The original global QConfig setting for per-channel weights is good.
    weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    # Ensure activation observer is appropriate for static quantization (e.g., MinMax based)
    # get_default_qat_qconfig_mapping usually provides this.
    # Default for "x86" uses MovingAverageMinMaxObserver for activations.
    global_qat_qconfig = QConfig(
        activation=qconfig_mapping.global_qconfig.activation, # Use default activation observer
        weight=weight_observer
    )
    qconfig_mapping = qconfig_mapping.set_global(global_qat_qconfig)
    
    prepared_model = prepare_qat_fx_torch(model_fp32_eval_cpu, qconfig_mapping, example_inputs_cpu)
    return prepared_model

def get_qat_model_fx(model_fp32_eval_cpu: nn.Module, example_inputs_cpu: tuple):
    # model_fp32_eval_cpu is ALREADY in .eval() mode when passed here
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")

    # Customize global config (mainly for weight observer here)
    weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    custom_global_qconfig = QConfig(
        activation=qconfig_mapping.global_qconfig.activation,
        weight=weight_observer
    )
    qconfig_mapping = qconfig_mapping.set_global(custom_global_qconfig)

    # Explicitly ensure Dropout is NOT quantized (default mapping should do this)
    qconfig_mapping = qconfig_mapping.set_module_name_qconfig(torch.nn.Dropout, None)
    # Also ensure BN is set for fusion
    qconfig_mapping = qconfig_mapping.set_module_name_qconfig(torch.nn.BatchNorm2d, None)
    qconfig_mapping = qconfig_mapping.set_module_name_qconfig(torch.nn.BatchNorm3d, None)

    # For debugging QConfig for Dropout:
    # print(f"[INFO] QConfig for nn.Dropout in get_qat_model_fx: {qconfig_mapping.module_name_qconfigs.get(torch.nn.Dropout, 'Not Set (should be None)')}")

    # model_fp32_eval_cpu is already .eval()
    # F.dropout is patched to identity by the caller of this function
    prepared_model = prepare_qat_fx_torch(model_fp32_eval_cpu, qconfig_mapping, example_inputs_cpu)
    return prepared_model

# --- Main script execution ---
data_dir: str = "filtered_imagenet2_native"
epochs: int = 5
qat_epochs: int = 3
batch: int = 64
lr: float = 0.025
qat_lr_factor: float = 0.05
width_mult: float = 1.0
device_arg = None # Keep variable name different from torch.device
compile_model: bool = False
arch: str = "mnv4s"
pretrained: bool = True

if device_arg is None:
    device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device_name = device_arg

if device_name == "mps": # PyTorch anmed it device, not device_arg
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
dev = torch.device(device_name)
print(f"[INFO] device: {dev}")

pretrain_str = "_pretrained" if pretrained else ""
pt_path = f"mobilenet_w{str(width_mult).replace('.', '_')}_{arch}{pretrain_str}_int8_fullpipe.pt"

with open(Path(data_dir) / "class_mapping.json") as f:
    ncls = len(json.load(f));
print(f"[INFO] #classes = {ncls}")

tr = get_loader(data_dir, batch, True, dev)
vl = get_loader(data_dir, batch, False, dev)

model = build_model(ncls, width_mult, dev, arch=arch, pretrained=pretrained)

crit = nn.CrossEntropyLoss(label_smoothing=0.1)
# For FP32 training on CUDA, scaler is used. For QAT, it's typically not (or handled carefully).
# The train_epoch function handles this logic.
scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda"))


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
    # model is already on 'dev', move to CPU for state_dict manipulation if preferred, though not strictly necessary here
    full_sd = model.cpu().state_dict() # Ensure it's on CPU for consistent key access

    key_prefix = "2.classifier." if arch in ["mnv3", "mnv2"] else "2.head." if arch in ["mnv4s", "mnv4m"] else None
    backbone_prefix = "2." # Corresponds to the 'backbone' module in nn.Sequential

    if key_prefix:
        for k, v in full_sd.items():
            # We want keys from the backbone, which is the 3rd module (index 2)
            if k.startswith(backbone_prefix) and not k.startswith(key_prefix): # Exclude classifier/head keys
                fp32_backbone_state_dict[k[len(backbone_prefix):]] = v # Strip the "2." prefix
        if fp32_backbone_state_dict:
            bpath = f"{Path(pt_path).stem}_fp32_backbone.pt" # Save in same dir as main model
            Path(bpath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(fp32_backbone_state_dict, bpath)
            print(f"[SAVE] FP32 PyTorch backbone state_dict → {bpath}")
except Exception as e:
    print(f"Failed to export fp32 backbone: {repr(e)}")

# Move model back to original device if it was moved to CPU for state_dict
model.to(dev)


print("[INFO] Preparing model for QAT...")
# QAT is often done on CPU or specific device. If MPS, fallback to CPU for QAT.
qat_train_device = torch.device("cpu") if dev.type == "mps" else dev
print(f"[INFO] QAT will run on: {qat_train_device}")

# Store original F.dropout, patch it if necessary for tracing (user mentioned it was for MNV4)
# _original_F_dropout = F.dropout
# This lambda makes F.dropout an identity function during prepare_qat_fx tracing.
# F.dropout = lambda x, p=0.0, training=False, inplace=False: x

model_fp32_for_qat_prep = model.cpu() # Get a CPU copy of the trained model
# CRITICAL: Set to eval mode for prepare_qat_fx.
model_fp32_for_qat_prep.eval()

example_inputs_cpu = (torch.randint(0, 256, (1, 3, DUMMY_H, DUMMY_W), dtype=torch.uint8),)
qat_prepared_model = get_qat_model_fx(model_fp32_for_qat_prep, example_inputs_cpu)

# Restore original F.dropout behavior for QAT fine-tuning phase
# F.dropout = _original_F_dropout

qat_model = qat_prepared_model.to(qat_train_device) # Move prepared model to QAT device
# Optimizer for QAT model
opt_q = optim.SGD(qat_model.parameters(), lr=lr * qat_lr_factor, momentum=0.9, weight_decay=1e-4) # Added weight_decay

print(f"[INFO] Starting QAT fine-tuning on {qat_train_device}")
# Note: The 'scaler' for AMP is not typically used in the QAT loop in this setup.
# train_epoch handles disabling AMP for QAT.
for qep in range(qat_epochs):
    l_q = train_epoch(qat_model, tr, crit, opt_q, scaler, qat_train_device, qep, qat_mode_active=True)
    # It's good practice to evaluate on the same device as training for consistency during QAT.
    val_acc_q = evaluate(qat_model, vl, qat_train_device)
    print(f"[QAT] Epoch {qep+1}/{qat_epochs} loss {l_q:.4f} val@1 {val_acc_q*100:.2f}%")

print("[INFO] Preparing QAT model for INT8 conversion (CPU, eval mode)...")
qat_model_for_conversion = qat_model.cpu()       # Move to CPU
qat_model_for_conversion.eval()                  # Set to eval mode (disables dropout, uses learned BN stats, fixes observer ranges)
print("[INFO] Converting QAT model to an INT8 GraphModule...")
# The qat_model_for_conversion (which is on CPU and in .eval()) is passed.
int8_model_final = convert_fx_torch(qat_model_for_conversion)

# Optional: Print the int8_graph_module structure to verify quantization and fusion
# print("\n--- INT8 GraphModule Structure (Post convert_fx_torch) ---")
# int8_graph_module.print_readable(print_output=True)
# print("--- End of INT8 GraphModule Structure ---\n")

if compile_model:
    print("[INFO] Compiling INT8 model with torch.compile...")
    try:
        # Ensure we are compiling the actual module, not a wrapper if any
        base_model_to_compile = int8_model_final.module_ if hasattr(int8_model_final, 'module_') else int8_model_final # common for DataParallel remnants
        # torch.compile works best on CPU or CUDA
        compiled_model = torch.compile(base_model_to_compile.cpu(), mode="reduce-overhead", fullgraph=True) # try fullgraph
        int8_model_final = compiled_model
    except Exception as e:
        print(f"[WARN] torch.compile() failed – {e}")

# Ensure the model for saving/export is the core module and on CPU
final_model_for_export = int8_model_final.module_ if hasattr(int8_model_final, 'module_') else int8_model_final
final_model_for_export = final_model_for_export.cpu().eval()


torch.save(final_model_for_export.state_dict(), pt_path)
print(f"[SAVE] INT8 PyTorch model state_dict → {pt_path}")

onnx_path = Path(pt_path).with_suffix(".onnx")
print(f"[INFO] Exporting INT8 model to ONNX: {onnx_path}")
dummy_input_onnx_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)

try:
    torch.onnx.export(
        final_model_for_export, # Use the CPU, eval mode model
        (dummy_input_onnx_cpu,),
        str(onnx_path), # Ensure onnx_path is a string
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

print("[DONE]")
