#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3-Small/V2/V4s → QAT → INT8 ONNX (opset-17, full pipeline).
Corrected QAT workflow for PyTorch FX.
Torch 2.7, cuda 12.8, onnx 1.17.0

main thing to try that is not here is diffusion
"""

from __future__ import annotations
import json, os, random, warnings, copy, sys # Added copy
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision.transforms.v2 as T_v2
from torchvision.transforms.v2 import MixUp, CutMix
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
from torch.ao.quantization.backend_config import get_native_backend_config # Correct import for backend_config

warnings.filterwarnings(
    "ignore", message="'.*has_(cuda|cudnn|mps|mkldnn).*is deprecated", module="torch.overrides"
)


folder_to_add = r"C:\Users\Colin\qat_onnx_models\scripts" # Use 'r' for raw string to handle backslashes
sys.path.append(folder_to_add)

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

from timm.layers.norm_act import BatchNormAct2d # For isinstance check
from timm.layers.conv_bn_act import ConvBnAct # Another common timm layer


def replace_timm_bn_act_with_torch_bn_and_act(module: nn.Module, module_name_prefix=""):
    for child_name, child_module in module.named_children():
        full_child_name = f"{module_name_prefix}.{child_name}" if module_name_prefix else child_name
        
        if isinstance(child_module, BatchNormAct2d):
            new_bn = nn.BatchNorm2d(
                num_features=child_module.num_features,
                eps=child_module.eps,
                momentum=child_module.momentum,
                affine=child_module.affine,
                track_running_stats=child_module.track_running_stats
            )
            # Transfer state for the BN part
            # This assumes child_module.bn.state_dict() gives the pure BN state if it has an internal .bn
            # Or if child_module itself holds the BN params directly.
            # For BatchNormAct2d, it seems it directly holds BN params.
            bn_state = {k: v for k, v in child_module.state_dict().items() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k or k == 'weight' or k == 'bias'}
            # Filter further if activation has its own weight/bias
            bn_param_names = {'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'}
            bn_state_filtered = {k: v for k, v in child_module.state_dict().items() if k in bn_param_names}

            if bn_state_filtered : new_bn.load_state_dict(bn_state_filtered, strict=True) # strict=True now for BN part

            layers_to_replace_with = [new_bn]
            
            # Handle the activation part
            if not isinstance(child_module.act, nn.Identity):
                print(f"[INFO] {full_child_name} had activation: {type(child_module.act)}. Adding it after new nn.BatchNorm2d.")
                layers_to_replace_with.append(child_module.act) # Add the original activation instance
            
            # Handle the drop_path part (usually nn.Identity if drop_path=0)
            if not isinstance(child_module.drop, nn.Identity):
                print(f"[INFO] {full_child_name} had drop_path: {type(child_module.drop)}. Adding it.")
                layers_to_replace_with.append(child_module.drop)

            if len(layers_to_replace_with) > 1:
                replacement_module = nn.Sequential(*layers_to_replace_with)
            else:
                replacement_module = new_bn
            
            print(f"[INFO] Replacing {full_child_name} (timm.BatchNormAct2d) with {type(replacement_module)}")
            setattr(module, child_name, replacement_module)

        elif isinstance(child_module, ConvBnAct):
            if hasattr(child_module, 'bn') and isinstance(child_module.bn, BatchNormAct2d):
                timm_bn_act_inner = child_module.bn
                
                new_inner_bn = nn.BatchNorm2d(
                    num_features=timm_bn_act_inner.num_features, eps=timm_bn_act_inner.eps, 
                    momentum=timm_bn_act_inner.momentum, affine=timm_bn_act_inner.affine,
                    track_running_stats=timm_bn_act_inner.track_running_stats
                )
                inner_bn_param_names = {'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'}
                inner_bn_state_filtered = {k: v for k, v in timm_bn_act_inner.state_dict().items() if k in inner_bn_param_names}

                if inner_bn_state_filtered : new_inner_bn.load_state_dict(inner_bn_state_filtered, strict=True)

                inner_layers_to_replace_with = [new_inner_bn]
                if not isinstance(timm_bn_act_inner.act, nn.Identity):
                    inner_layers_to_replace_with.append(timm_bn_act_inner.act)
                if not isinstance(timm_bn_act_inner.drop, nn.Identity):
                    inner_layers_to_replace_with.append(timm_bn_act_inner.drop)
                
                if len(inner_layers_to_replace_with) > 1:
                    child_module.bn = nn.Sequential(*inner_layers_to_replace_with)
                else:
                    child_module.bn = new_inner_bn
                print(f"[INFO] Replacing {full_child_name}.bn (BatchNormAct2d within ConvBnAct) with {type(child_module.bn)}")
            
            replace_timm_bn_act_with_torch_bn_and_act(child_module, full_child_name) # Recurse into ConvBnAct's children
        else:
            replace_timm_bn_act_with_torch_bn_and_act(child_module, full_child_name)

def get_backbone(arch: str, ncls: int, width: float,
                 pretrained: bool, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 replace_bn_for_qat: bool = False) -> nn.Module:
    if arch == "mnv3":
        model = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            width_mult=width, dropout=drop_rate)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, ncls)
    elif arch == "mnv3l":
        model = tvm.mobilenet_v3_large(
            weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None,
            width_mult=width, dropout=drop_rate)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, ncls)
    elif arch == "mnv2":
        model = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None,
            width_mult=width,
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
    elif arch == "mnv4c":
        from customMobilenetNetv4 import MobileNetV4ConvSmallPico
        # For PicoDet, you might want to set num_classes=0 and specify out_indices
        # For classification as in this script, num_classes=ncls
        model = MobileNetV4ConvSmallPico(
            width_multiplier=width,
            num_classes=ncls,
            drop_rate=drop_rate, # For classifier
            drop_path_rate=drop_path_rate # For stochastic depth in blocks
            # out_features_indices can be set if used for detection later
        )
        # The model already has its own classifier if ncls > 0
        # No need to replace model.classifier like for mnv3/mnv2
    elif arch in {"mnv4s", "mnv4m"}:
        try:
            import timm
            from timm.layers.norm_act import BatchNormAct2d # For isinstance checks later
            from timm.layers.conv_bn_act import ConvBnAct
        except ImportError:
            raise ImportError("Please install timm to use MobileNetV4: pip install timm")
        
        timm_name = _mnv4_id(arch, width)
        model = timm.create_model(
            timm_name,
            pretrained=pretrained, # Load pretrained weights first if applicable
            num_classes=ncls,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        if (arch in {"mnv4s", "mnv4m"}):
            print(f"[INFO] Attempting to replace timm BatchNorm layers in '{timm_name}' with torch.nn.BatchNorm2d for QAT compatibility...")
            # If pretrained, the state_dict keys for BN might differ slightly.
            # This replacement is simpler if done BEFORE loading state_dict,
            # or if the state_dict loading is robust to minor differences / done carefully.
            # If you load pretrained weights for the timm model, those weights are for
            # timm.BatchNormAct2d. When you replace with nn.BatchNorm2d, you'd ideally
            # want to transfer these weights.
            
            # If model is already pretrained and loaded:
            # Create a temporary fresh model, replace BNs, then load state_dict carefully.
            # Or, replace BNs and then try to load state_dict from the original timm model,
            # possibly with strict=False and manual key mapping if needed.

            # For simplicity here, assuming replacement happens on a model structure that will
            # either be trained from scratch or where state_dict loading can handle this.
            
            # --- Perform replacement ---
            # We need to preserve the pretrained weights if `pretrained=True`.
            # The most robust way if weights are loaded *before* replacement:
            if pretrained:
                # 1. Store state_dict of the original timm model
                original_state_dict = model.state_dict()
                
                # 2. Create a NEW instance of the model WITHOUT pretrained weights (to modify its structure)
                temp_model_for_structure = timm.create_model(timm_name, pretrained=False, num_classes=ncls, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
                replace_timm_bn_act_with_torch_bn_and_act(temp_model_for_structure) # Modify structure
                
                # 3. Load original state_dict into the modified structure.
                # This might require careful key mapping if timm.BatchNormAct2d has different keys
                # than nn.BatchNorm2d or if it includes activation parameters.
                # For standard BN params (weight, bias, running_mean, running_var), keys are often the same.
                
                # Simplified approach: If keys mostly match for BN parts
                # We need to handle cases where BatchNormAct2d might have extra parameters
                # related to its activation, which nn.BatchNorm2d won't have.
                # So, load with strict=False.
                
                # Create a new model instance that we will modify and then load state into
                model_with_torch_bn = timm.create_model(timm_name, pretrained=False, num_classes=ncls, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
                replace_timm_bn_act_with_torch_bn_and_act(model_with_torch_bn) # Replace BNs in this new instance
                
                # Now, load the state from the original pretrained model into this new one
                # We need to filter the original_state_dict to only include keys present
                # in model_with_torch_bn, or handle mismatches.
                
                # A common pattern for state_dict loading with structural changes:
                new_state_dict = model_with_torch_bn.state_dict()
                for k_orig, v_orig in original_state_dict.items():
                    if k_orig in new_state_dict and new_state_dict[k_orig].shape == v_orig.shape:
                        new_state_dict[k_orig] = v_orig
                    else:
                        # This part needs careful debugging if keys don't match for BN layers.
                        # E.g. timm's BNAct might be 'module.bn.weight' vs 'module.weight'
                        print(f"[WARN] Key mismatch or shape mismatch during state_dict transfer for QAT BN replacement: {k_orig}")
                model_with_torch_bn.load_state_dict(new_state_dict)
                model = model_with_torch_bn # Use the model with replaced BNs
                print("[INFO] Transferred state_dict to model with replaced BNs.")

            else: # Not pretrained, just replace in the current model
                replace_timm_bn_act_with_torch_bn_and_act(model)
            print(f"[INFO] Finished BatchNorm replacement for '{timm_name}'.")
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
        T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True), # 0: Resize, antialias=False would be faster, antialias requires opset 18
        PreprocNorm(),                                   # 1: Preprocessing
        backbone                                         # 2: Backbone
    ).to(
        dev, memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format
    )
    return model

# ───────────────────── training helpers ─────────────────────────
def train_epoch(model, loader, crit, opt, scaler, dev, ep, qat_mode_active:bool = False, mixup_fn=None):
    model.train(); tot = loss_sum = 0
    for i, (img, lab) in enumerate(loader): # Consider adding tqdm here for progress
        img = img.to(dev, non_blocking=True,
                     memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format)
        lab = lab.to(dev, non_blocking=True)
        # Apply MixUp/CutMix if provided
        if mixup_fn and not qat_mode_active: # Usually not applied during QAT fine-tuning
            # T_v2 MixUp/CutMix expect labels to be one-hotted if not already
            img, lab = mixup_fn(img, lab) # lab will now be soft labels
        opt.zero_grad(set_to_none=True)

        if not qat_mode_active and dev.type == "cuda":
            with autocast(device_type=dev.type, enabled=True):
                out = model(img); loss = crit(out, lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            out = model(img); loss = crit(out, lab)
            loss.backward(); opt.step()

        tot += img.size(0); loss_sum += loss.item() * img.size(0)
        if i == 0 and ep == 0: # First batch of first epoch
            print(f"[{'QAT ' if qat_mode_active else ''}Batch-0] input shape {img.shape} input dtype {img.dtype} loss {loss.item():.4f}")
    return loss_sum / tot

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); corr = tot = 0
    for img, lab in loader: # Consider adding tqdm here for progress
        img = img.to(dev, memory_format=torch.channels_last if _CNL(dev) else torch.contiguous_format)
        lab = lab.to(dev); corr += (model(img).argmax(1) == lab).sum().item(); tot += lab.size(0)
    return corr / tot

# --- Main script execution ---
data_dir: str = "filtered_imagenet2_native" # Example, replace with your actual path
epochs: int = 750
qat_epochs: int = 10
batch: int = 64
lr: float = 0.025
qat_lr_factor: float = 0.05
width_mult: float = 1.0
device_arg = None
compile_model: bool = False
arch: str = "mnv4s" # Change to "mnv2", "mnv3", "mnv3l", "mnv4s", "mnv4m" as needed
pretrained: bool = True # Using pretrained weights for FP32 start
drop_rate: float = 0.2

if device_arg is None:
    device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device_name = device_arg

if device_name == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
dev = torch.device(device_name)
print(f"[INFO] Device: {dev}")

# Ensure data_dir exists
if not Path(data_dir).exists() or not (Path(data_dir) / "train").exists() or not (Path(data_dir) / "val").exists():
    print(f"[ERROR] Data directory '{data_dir}' or its 'train'/'val' subdirectories not found.")
    print("Please create dummy ImageFolder structure or point to existing data.")
    print("Example: data_dir/train/classA/img1.jpg, data_dir/val/classA/img2.jpg")
    exit(1)

pretrain_str = "_pretrained" if pretrained else ""
# Using a more descriptive name that includes QAT status
output_base_name = f"mobilenet_w{str(width_mult).replace('.', '_')}_{arch}{pretrain_str}_drp{str(drop_rate).replace('.', '_')}"
pt_path = f"{output_base_name}_qat_int8.pt" # For PyTorch INT8 state_dict
onnx_fp32_path = f"{output_base_name}_fp32_from_qat.onnx"
onnx_path = f"{output_base_name}_qat_int8.onnx" # For ONNX INT8 model

# Create class_mapping.json if it doesn't exist for dummy runs
class_map_path = Path(data_dir) / "class_mapping.json"
if not class_map_path.exists():
    # Try to infer from 'val' directory structure
    try:
        class_names = sorted([d.name for d in (Path(data_dir) / "val").iterdir() if d.is_dir()])
        if not class_names: raise FileNotFoundError # force except if no classes found
        class_mapping = {name: i for i, name in enumerate(class_names)}
        with open(class_map_path, 'w') as f:
            json.dump(class_mapping, f)
        print(f"[INFO] Created dummy class_mapping.json with {len(class_names)} classes based on '{data_dir}/val' subdirectories.")
    except Exception as e:
        print(f"[WARN] Could not auto-create class_mapping.json from '{data_dir}/val': {e}")
        print("[WARN] Please create class_mapping.json or ensure data directory is correct.")
        # Create a truly dummy one if all else fails for script to run
        default_dummy_classes = {"class0": 0, "class1": 1}
        with open(class_map_path, 'w') as f:
            json.dump(default_dummy_classes, f)
        print(f"[INFO] Created a default dummy class_mapping.json with {len(default_dummy_classes)} classes.")


with open(class_map_path) as f:
    class_map = json.load(f)
    class_names = list(class_map.keys())
    ncls = len(class_map)
    
print(f"[INFO] #classes = {ncls}")


tr = get_loader(data_dir, batch, True, dev)
vl = get_loader(data_dir, batch, False, dev)

model = build_model(ncls, width_mult, dev, arch=arch, pretrained=pretrained, drop_rate=drop_rate)

crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda" and not compile_model)) # torch.compile may not like scaler with fullgraph

stock_trainer = False
if stock_trainer:
    # used for faster, simpler testing
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
else:
    # Define the new hyperparameters
    peak_lr = 0.002
    adamw_weight_decay = 0.01
    adamw_beta1 = 0.6
    adamw_beta2 = 0.999
    adamw_epsilon = 1e-6
    total_training_epochs = epochs if epochs > 10 else 10
    warmup_epochs = 5
    cosine_decay_alpha = 0.0 # This usually maps to eta_min_ratio or eta_min directly
    
    # 1. New Optimizer: AdamW
    opt = optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_epsilon,
        weight_decay=adamw_weight_decay
    )
    
    # Cosine Annealing with Warm-up
    # Warm-up scheduler: Linear increase from a very small LR to peak_lr
    # The start_factor can be chosen to be very small, e.g., 1e-6, so LR effectively starts from 0
    warmup_scheduler = LinearLR(
        opt,
        start_factor=1e-6, # Start from near zero
        end_factor=1.0,    # End at the peak_lr
        total_iters=warmup_epochs
    )
    
    # Cosine Annealing scheduler: Decays from peak_lr to peak_lr * cosine_decay_alpha
    # T_max is the total number of epochs for the cosine decay phase.
    # eta_min is the minimum learning rate. Since cosine_decay_alpha is 0.0, eta_min will be 0.
    cosine_scheduler = CosineAnnealingLR(
        opt,
        T_max=total_training_epochs - warmup_epochs, # Remaining epochs after warm-up
        eta_min=peak_lr * cosine_decay_alpha # This will be 0 if cosine_decay_alpha is 0.0
    )
    
    # Combine them using SequentialLR
    # The schedulers will be applied sequentially.
    # The `milestones` define when the next scheduler in the list takes over.
    sched = SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

print("[INFO] Starting FP32 training...")
# if used, operate on batches, pass to mixup_fn
mixup_or_cutmix = T_v2.RandomChoice([
    MixUp(alpha=0.2, num_classes=ncls),
    CutMix(alpha=1.0, num_classes=ncls)
])
for ep in range(epochs):
    l = train_epoch(model, tr, crit, opt, scaler, dev, ep, qat_mode_active=False, mixup_fn=None)
    a = evaluate(model, vl, dev)
    sched.step()
    print(f"FP32 Epoch {ep+1}/{epochs}  loss {l:.4f}  val@1 {a*100:.2f}%  lr {opt.param_groups[0]['lr']:.5f}")

# save raw training for loading into object detector on same backbone
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

try:
    path = output_base_name + "fp32_onnx.onnx"
    dummy_input_onnx_cpu = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    torch.onnx.export(
        model.cpu().eval(),
        (dummy_input_onnx_cpu,),
        path,
        input_names=["input_u8"],
        output_names=["logits"],
        dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        # dynamo=False,
    )
    print(f"[SAVE] ONNX model → {path}")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {repr(e)}")
    import traceback
    traceback.print_exc()

# ───────────────────────── QAT Process Start ─────────────────────────
print("\n[INFO] Preparing model for QAT...")
qat_train_device = torch.device("cpu") if dev.type == "mps" else dev
print(f"[INFO] QAT will run on: {qat_train_device}")

# 1. Create a CPU copy of the FP32-trained model FOR QAT PREPARATION
#    Make sure it's a deepcopy to avoid altering the original FP32 model.
model_for_qat_prep = copy.deepcopy(model).cpu()
model_for_qat_prep.train() # IMPORTANT: Set to TRAIN mode for prepare_qat_fx

# 2. Define example inputs for tracing.
#    The model takes uint8 and PreprocNorm converts to FP32.
#    The example input should be a tuple.
example_inputs_cpu_for_prepare = (torch.randint(0, 256, (1, 3, DUMMY_H, DUMMY_W), dtype=torch.uint8),)

# 3. Get QConfig mapping and Backend Config
#    Using "x86" for CPU, implies FBGEMM. For server-side.
#    For mobile, "qnnpack" might be an alternative if targeting Android.
qconfig_mapping = get_default_qat_qconfig_mapping("x86")

# Customize QConfig for per-channel weights (optional, but often good)
# The default "x86" QAT mapping already uses good observers.
# If you want specific per-channel symmetric for weights globally:
default_activation_factory = qconfig_mapping.global_qconfig.activation
per_channel_weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)
custom_global_qconfig = QConfig(
    activation=default_activation_factory,
    weight=per_channel_weight_observer
)
qconfig_mapping = qconfig_mapping.set_global(custom_global_qconfig)
# Note: The default "x86" map already sets BN/Dropout QConfig to None, so they won't be quantized.

backend_config = get_native_backend_config() # For fusion patterns

# 4. Patch F.dropout for tracing if it causes issues (especially with complex models or control flow)
#    This makes F.dropout an identity function ONLY during prepare_qat_fx tracing.
_original_F_dropout = F.dropout
# For MobileNetV4 particularly, or if tracing fails, this can be helpful.
# If tracing works without it, you can comment this patch out.
F.dropout = lambda x, p=0.0, training=False, inplace=False: x
print("[INFO] F.dropout patched for prepare_qat_fx tracing.")

# 5. Call prepare_qat_fx
print("[INFO] Calling prepare_qat_fx_torch...")
qat_prepared_model_cpu = prepare_qat_fx_torch(
    model_for_qat_prep, # This model MUST be in .train() mode
    qconfig_mapping,
    example_inputs_cpu_for_prepare,
    backend_config=backend_config # Crucial for enabling fusion patterns
)

# 6. Restore original F.dropout behavior for QAT fine-tuning phase
F.dropout = _original_F_dropout
print("[INFO] F.dropout restored.")

# 7. Move prepared model to QAT training device and ensure it's in train mode for fine-tuning
qat_model = qat_prepared_model_cpu.to(qat_train_device)
qat_model.train() # Ensure it's in train mode for the QAT fine-tuning loop

# Optimizer for QAT model (typically with a lower learning rate)
opt_q = optim.SGD(qat_model.parameters(), lr=lr * qat_lr_factor, momentum=0.9, weight_decay=1e-4)
# Note: Scheduler for QAT fine-tuning might also be useful, e.g., CosineAnnealingLR for qat_epochs
sched_q = CosineAnnealingLR(opt_q, T_max=qat_epochs, eta_min=(lr * qat_lr_factor) * 0.01)


print(f"[INFO] Starting QAT fine-tuning on {qat_train_device} for {qat_epochs} epochs...")
# The 'scaler' for AMP is not used here as train_epoch handles it via qat_mode_active.
for qep in range(qat_epochs):
    l_q = train_epoch(qat_model, tr, crit, opt_q, scaler, qat_train_device, qep, qat_mode_active=True)
    # It's good practice to evaluate on the same device as training for consistency during QAT.
    val_acc_q = evaluate(qat_model, vl, qat_train_device)
    sched_q.step()
    print(f"QAT Epoch {qep+1}/{qat_epochs} loss {l_q:.4f} val@1 {val_acc_q*100:.2f}% lr {opt_q.param_groups[0]['lr']:.6f}")

# ───────────────────────── INT8 Conversion and Export ─────────────────────────
print("\n[INFO] Preparing QAT model for INT8 conversion (CPU, eval mode)...")
qat_model.eval() # Set to eval mode (disables dropout, uses learned BN stats, fixes observer ranges)
qat_model_for_conversion = qat_model.cpu() # Move to CPU

print("[INFO] Converting QAT model to an INT8 GraphModule using convert_fx_torch...")
# Pass the backend_config to convert_fx as well to ensure fusions are correctly applied
int8_model_final = convert_fx_torch(
    qat_model_for_conversion,
    backend_config=backend_config # Pass backend_config here too!
)

# Optional: Print the int8_model_final structure to verify quantization and fusion
# print("\n--- INT8 GraphModule Structure (Post convert_fx_torch) ---")
# int8_model_final.print_readable(print_output=True) # May be very verbose
# print("--- End of INT8 GraphModule Structure ---\n")

if compile_model:
    print("[INFO] Compiling INT8 model with torch.compile...")
    try:
        # Ensure we are compiling the actual module
        # torch.compile works best on CPU or CUDA
        compiled_model = torch.compile(int8_model_final.cpu(), mode="reduce-overhead", fullgraph=True)
        int8_model_final = compiled_model
        print("[INFO] torch.compile successful.")
    except Exception as e:
        print(f"[WARN] torch.compile() failed – {e}. Using uncompiled model.")
        # Ensure int8_model_final is still the CPU model if compile failed
        int8_model_final = int8_model_final.cpu()


# Ensure the model for saving/export is the core module and on CPU, in eval mode
final_model_for_export = int8_model_final.cpu().eval()

# Save PyTorch INT8 state_dict
torch.save(final_model_for_export.state_dict(), pt_path)
print(f"[SAVE] INT8 PyTorch model state_dict → {pt_path}")

# Export to ONNX
print(f"[INFO] Exporting INT8 model to ONNX: {onnx_path}")
# Dummy input for ONNX export should match the expected input of the final ONNX model
# Your model has an internal T_v2.Resize((IMG_SIZE, IMG_SIZE)), so input can be IMG_SIZE.
dummy_input_onnx_cpu = (torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),)

try:
    torch.onnx.export(
        final_model_for_export,
        dummy_input_onnx_cpu, # Must be a tuple of inputs
        str(onnx_path),
        input_names=["input_u8"],
        output_names=["logits"],
        dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=18, # Using a recent opset, 18 has antialias resize
        do_constant_folding=True
    )
    print(f"[SAVE] INT8 ONNX model → {onnx_path}")
    print(f"[INFO] Use the ONNX inspection and inference script to test '{Path(onnx_path).name}'")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {repr(e)}")
    import traceback
    traceback.print_exc()

try:
    float_model = copy.deepcopy(model).train()
    float_model.meta = {
        "task":  "classify",
        "nc":    ncls,
        "names": {i: n for i, n in enumerate(class_names)},
        # optional extras:
        "input_size": [3, IMG_SIZE, IMG_SIZE],
        "mean":  [0.485, 0.456, 0.406],
        "std":   [0.229, 0.224, 0.225],
    }

    torch.onnx.export(
        float_model.cpu().eval(),
        dummy_input_onnx_cpu, # Must be a tuple of inputs
        str(onnx_path).replace("int8.onnx", "fp32_dynamo.onnx"),
        input_names=["input_u8"],
        output_names=["logits"],
        # dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        # dynamic for N, H, W — C stays fixed at 3
        dynamic_shapes={
            "input": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=18, # Using a recent opset, 18 has antialias resize
        dynamo=True,
        keep_initializers_as_inputs=False,
    )
    print(f"[SAVE] INT8 ONNX model → {onnx_path} dynamo")
    print(f"[INFO] Use the ONNX inspection and inference script to test '{Path(onnx_path).name}'")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {repr(e)}")
    import traceback
    traceback.print_exc()

print("\n[DONE]")



# onnx imports
import onnx
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
import onnxruntime as ort


# STATIC QUANTIZATION FROM ONNX to ONNX

# --- Configuration ---
num_calibration_batches = 100 # Number of batches to use for calibration
onnx_int8_path = output_base_name + "_ort_quantized.onnx"
provider = "CPUExecutionProvider" # Or "CUDAExecutionProvider"

# --- Configuration for ONNX Runtime PTQ ---
# data_dir, batch_size, num_calibration_batches, onnx_int8_path, provider are already defined

print("[INFO] Preparing calibration data for ONNX Runtime quantization...")
# Calibration loader can run on CPU, ensure it yields uint8 tensors
calibration_loader = get_loader(data_dir, batch, is_train=False, device=torch.device("cpu"))

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
print(f"[INFO] Using {num_calibration_batches} batches for calibration (batch size {batch}).")

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

onnx_model_optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
sess_options = ort.SessionOptions()
# Set graph optimization level
# ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Try ALL for more aggressive opts

# Set up the output path for the optimized model
sess_options.optimized_model_filepath = onnx_model_optimized_path

# Create a session with the model and options to trigger optimization and save.
# We don't actually need to run inference here.
_ = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

if os.path.exists(onnx_model_optimized_path):
    print(f"[INFO] Optimized ONNX model saved to: {onnx_model_optimized_path}")
    # Update the path to be used for quantization
    onnx_model_path_for_quantization = onnx_model_optimized_path
else:
    print(f"[WARN] Optimized model was not saved to {onnx_model_optimized_path}. Using previous model path.")

try:
    if True:
        quantize_static(
            path,
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

try:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.backend_config import get_native_backend_config
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    
    import torch.nn.functional as F
    
    # Step 1: Patch dropout to no-op for tracing during prepare_fx
    _original_dropout = F.dropout
    F.dropout = lambda x, *args, **kwargs: x  # Patch it to be identity
    
    # Step 2: Create CPU + eval copy of the FP32 model
    fp32_model_for_ptq = copy.deepcopy(model).cpu().eval()
    
    # Step 3: Define input and configs
    example_input = (torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),)
    qconfig_mapping = get_default_qconfig_mapping("x86")  # or 'qnnpack' for ARM/mobile
    backend_config = get_native_backend_config()
    
    # Step 4: Prepare for quantization (tracing will now succeed)
    prepared_model = prepare_fx(
        fp32_model_for_ptq,
        qconfig_mapping=qconfig_mapping,
        example_inputs=example_input,
        backend_config=backend_config
    )
    
    # Step 5: Restore original dropout
    F.dropout = _original_dropout
    print("[INFO] Restored F.dropout after prepare_fx.")

    # Use a subset of validation data for calibration
    prepared_model.eval()
    with torch.inference_mode():
        for i, (images, _) in enumerate(vl):  # `vl` is the val DataLoader
            if i >= 50: break  # 50 batches for calibration
            prepared_model(images.to(torch.uint8))
    int8_ptq_model = convert_fx(prepared_model, backend_config=backend_config)
    # ONNX export dummy input
    dummy_input = (torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),)
    
    torch.onnx.export(
        int8_ptq_model,
        dummy_input,
        output_base_name + "fp32_model_ptq_int8.onnx",
        input_names=["input_u8"],
        output_names=["logits"],
        dynamic_axes={"input_u8": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True
    )
    print("[SAVE] Exported static PTQ INT8 ONNX model → fp32_model_ptq_int8.onnx")

except Exception as e:
    print(repr(e))

print("[DONE]")