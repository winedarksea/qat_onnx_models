#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV2 → FP32 warm-up → QAT → INT8 → ONNX
PyTorch 2.7  (eager-mode quantisation)
"""

from __future__ import annotations
import argparse, json, os, random, time, warnings
from pathlib import Path
from typing import Tuple

import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torch.ao.quantization import (
    prepare_qat, convert, get_default_qat_qconfig, QuantStub, DeQuantStub,
    fuse_modules_qat as fuse_modules,  # eager-mode fusion util that keeps observers
)

# ────────────────────────────── misc ─────────────────────────────
SEED = 42
IMG_SIZE = 224          # network input size
NUM_WORKERS = 0         # keep 0 for Windows / debug
BATCH = 64
EPOCHS_FP32 = 5
EPOCHS_QAT = 3

random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────── transforms ──────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def _build_tf(is_train: bool):
    aug = []
    if is_train:
        aug += [
            T_v2.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.67, 1.0), antialias=True),
            T_v2.RandomHorizontalFlip(),
            T_v2.TrivialAugmentWide(),
        ]
    else:
        aug += [T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)]
    aug += [
        T_v2.ToDtype(torch.float32, scale=True),       # uint8 → [0,1] float
        T_v2.Normalize(MEAN, STD),
    ]
    return T_v2.Compose(aug)

# ─────────────────────────── dataloader ──────────────────────────
def get_loader(root: str, batch: int, train: bool, device: torch.device):
    split = "train" if train else "val"
    ds = dsets.ImageFolder(Path(root) / split, transform=_build_tf(train))
    pf = 4 if NUM_WORKERS else None
    return DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda", persistent_workers=bool(NUM_WORKERS),
        prefetch_factor=pf, drop_last=train,
    )

# ─────────────────────── MobileNet V2 (quant-ready) ───────────────────────
def _make_divisible(v, d, min_value=None):
    if min_value is None: min_value = d
    new_v = max(min_value, int(v + d / 2) // d * d)
    if new_v < 0.9 * v: new_v += d
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(
            self, c_in, c_out, kernel_size: int = 3,
            stride: int = 1, groups: int = 1
    ):
        p = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size, stride, p,
            groups=groups, bias=False),
            nn.BatchNorm2d(c_out, momentum=0.1),
            nn.ReLU(inplace=False)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, exp):
        super().__init__(); self.stride = stride
        hid = int(round(inp * exp))
        self.use_res = self.stride == 1 and inp == oup
        layers = []
        if exp != 1:
            layers.append(ConvBNReLU(inp, hid, k=1))
        layers.extend([
            ConvBNReLU(hid, hid, stride=stride, groups=hid),
            nn.Conv2d(hid, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res:
            return self.skip_add.add(x, self.conv(x))
        return self.conv(x)

class MobileNetV2Q(nn.Module):
    def __init__(self, ncls=1000, width=1.0, inv_set=None, round_nearest=8):
        super().__init__()
        if inv_set is None:
            inv_set = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        inp_ch = _make_divisible(32 * width, round_nearest)
        last_ch = _make_divisible(1280 * max(1.0, width), round_nearest)

        features = [ConvBNReLU(3, inp_ch, stride=2)]
        for t, c, n, s in inv_set:
            out_ch = _make_divisible(c * width, round_nearest)
            for i in range(n):
                features.append(InvertedResidual(inp_ch, out_ch, s if i == 0 else 1, t))
                inp_ch = out_ch
        features.append(ConvBNReLU(inp_ch, last_ch, k=1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_ch, ncls))

        self.quant = QuantStub(); self.dequant = DeQuantStub()
        self.last_ch = last_ch
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvBNReLU):
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if isinstance(m, InvertedResidual):
                for idx in range(len(m.conv) - 1):
                    if isinstance(m.conv[idx], nn.Conv2d):
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# ──────────────────── training / evaluation helpers ────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval(); corr = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        corr += (pred == y).sum().item(); tot += y.size(0)
    return corr / tot

def train_epoch(model, loader, crit, opt, device):
    model.train(); tot = loss_sum = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0); tot += x.size(0)
    return loss_sum / tot

# ─────────────────────────────── main ────────────────────────────────
def main(cfg):
    data_dir = "filtered_imagenet2_native"
    # device selection ----------------------------------------------------
    dev = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available()
                                       else "mps" if torch.backends.mps.is_available()
                                       else "cpu")
    )
    if dev.type == "mps": os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"[device] {dev}")

    # data ----------------------------------------------------------------
    with open(Path(data_dir) / "class_mapping.json") as f: ncls = len(json.load(f))
    tr_loader = get_loader(data_dir, BATCH, True,  dev)
    vl_loader = get_loader(data_dir, BATCH, False, dev)

    # fp32 model ----------------------------------------------------------
    model = MobileNetV2Q(ncls=ncls).to(dev)
    model.fuse_model()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS_FP32, eta_min=1e-3)

    print("[FP32] training …")
    for ep in range(EPOCHS_FP32):
        l = train_epoch(model, tr_loader, crit, opt, dev)
        acc = evaluate(model, vl_loader, dev)
        sched.step()
        print(f"  ep {ep+1}/{EPOCHS_FP32}  loss {l:.4f}  val@1 {acc*100:.2f}%")

    # qat preparation -----------------------------------------------------
    print("[QAT] preparing … (moving to CPU)")
    model.cpu().eval()
    model.qconfig = get_default_qat_qconfig('x86')
    prepare_qat(model, inplace=True)

    # fine-tune -----------------------------------------------------------
    opt_q = optim.SGD(model.parameters(), lr=0.025 * 0.05, momentum=0.9, weight_decay=1e-4)
    print("[QAT] fine-tuning …")
    for ep in range(EPOCHS_QAT):
        l = train_epoch(model, tr_loader, crit, opt_q, torch.device('cpu'))
        if ep > 1: model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)  # freeze BN stats
        if ep > 2: model.apply(torch.ao.quantization.disable_observer)  # freeze observers
        acc = evaluate(model, vl_loader, torch.device('cpu'))
        print(f"  ep {ep+1}/{EPOCHS_QAT}  loss {l:.4f}  val@1 {acc*100:.2f}%")

    # convert to int8 -----------------------------------------------------
    print("[INT8] converting …")
    int8_model = convert(model.eval(), inplace=False)

    # save / export -------------------------------------------------------
    stem = f"mobilenetv2_int8"
    pt_path   = Path(stem + ".pt")
    onnx_path = Path(stem + ".onnx")

    torch.save(int8_model.state_dict(), pt_path)
    print(f"[SAVE] PyTorch state_dict → {pt_path}")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        int8_model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"[SAVE] INT8 ONNX → {onnx_path}")

# ──────────────────────────── cli entry ───────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", help="'cuda' | 'mps' | 'cpu' (default: auto)")
    cfg = p.parse_args()
    main(cfg)
