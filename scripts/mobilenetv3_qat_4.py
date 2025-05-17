#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3-Small → QAT → INT8/ONNX (opset-18), runnable on
CUDA, Apple Metal (MPS), or pure CPU.

Key points
──────────
• ImageFolder + read_image(mode=RGB) ➜ uint8 CHW tensors
• Light aug (RandomResizedCrop, H-flip, TrivialAugmentWide)
• GPU-side normalisation
• SGD + cosine LR (no warm restarts, no EMA, no MixUp)
• 3 epoch QAT with per-channel observers
• optional torch.compile flag (ignored on unsupported back-ends)
"""

from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import List

import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.v2 as T_v2
import torchvision.models as tvm
import torchvision.datasets as dsets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
from torch.amp import autocast                              # new AMP API

# ───────────────────────────── setup ─────────────────────────────
SEED, IMG_SIZE, NUM_WORKERS = 42, 224, 0          # change workers as desired
random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────────── transforms ──────────────────────────
def _build_tf(train:bool):
    if train:
        return T_v2.Compose([
            T_v2.RandomResizedCrop((IMG_SIZE, IMG_SIZE),
                                   scale=(0.67, 1.0), antialias=True),
            T_v2.RandomHorizontalFlip(),
            T_v2.TrivialAugmentWide(),
        ])
    return T_v2.Compose([T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])

# ─────────────────────────── dataloader ──────────────────────────
def _rgb_loader(path:str):
    """Load image as 3-channel uint8 tensor via libjpeg-turbo."""
    return read_image(path, mode=ImageReadMode.RGB)         # CHW uint8 C=3

def get_loader(root:str, batch:int, train:bool,
               device:torch.device)->DataLoader:
    split = "train" if train else "val"
    dataset = dsets.ImageFolder(
        root=os.path.join(root, split),
        transform=_build_tf(train),
        loader=_rgb_loader
    )

    pf    = 4 if NUM_WORKERS>0 else None
    pwork = True if NUM_WORKERS>0 else False
    pin   = device.type == "cuda"                           # pin memory on CUDA

    print(f"[DL] {split}: {len(dataset)} imgs  workers={NUM_WORKERS}")
    return DataLoader(dataset,
                      batch_size=batch,
                      shuffle=train,
                      num_workers=NUM_WORKERS,
                      pin_memory=pin,
                      persistent_workers=pwork,
                      prefetch_factor=pf,
                      drop_last=train)

# ───────────────────────── helper modules ────────────────────────
class PreprocNorm(nn.Module):
    """uint8 ➜ float32 & ImageNet normalise (runs on chosen device)."""
    def __init__(self):
        super().__init__()
        m = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
        s = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
        self.register_buffer("m", m); self.register_buffer("s", s)
    def forward(self,x): return (x.float().div_(255) - self.m)/self.s

def _fuse_mobilenetv3(model:nn.Module):
    from torch.ao.quantization import fuse_modules
    backbone = model[1]
    for blk in backbone.features:
        if isinstance(blk, nn.Sequential):
            try: fuse_modules(blk, ['0','1','2'], inplace=True)
            except (AttributeError, KeyError): pass
        if hasattr(blk,'block'):
            for m in blk.block.children():
                if isinstance(m, nn.Sequential):
                    try: fuse_modules(m, ['0','1','2'], inplace=True)
                    except (AttributeError, KeyError): pass

def build_model(ncls:int, width:float, device:torch.device)->nn.Module:
    backbone = (tvm.mobilenet_v3_small(
                    weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                if width==1.0 else
                tvm.mobilenet_v3_small(weights=None, width_mult=width))
    backbone.classifier[3] = nn.Linear(backbone.classifier[3].in_features, ncls)
    model = nn.Sequential(PreprocNorm(), backbone).to(device)
    if device.type == "cuda":                                # channels_last only on CUDA
        model = model.to(memory_format=torch.channels_last)
    return model

# ─────────────────────── training utilities ─────────────────────
def train_epoch(model, loader, crit, opt, scaler, device, epoch):
    model.train(); tot=loss_sum=0
    for i,(img,lab) in enumerate(loader):
        img = img.to(device, non_blocking=True,
                     memory_format=torch.channels_last if device.type=="cuda"
                     else torch.contiguous_format)
        lab = lab.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type=="cuda"):
            out = model(img); loss = crit(out, lab)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        tot += img.size(0); loss_sum += loss.item()*img.size(0)
        if i==0 and epoch==0:
            print(f"[Batch-0] shape {img.shape} dtype {img.dtype} "
                  f"loss {loss.item():.4f}")
    return loss_sum/tot

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=total=0
    for img,lab in loader:
        img=img.to(device, memory_format=torch.channels_last if device.type=="cuda"
                   else torch.contiguous_format)
        lab=lab.to(device)
        pred=model(img).argmax(1)
        correct+=(pred==lab).sum().item(); total+=lab.size(0)
    return correct/total

# ──────────────────────── QAT helpers ────────────────────────────
def prepare_qat_fx(model):
    from torch.ao.quantization.quantize_fx import prepare_qat_fx
    from torch.ao.quantization import get_default_qat_qconfig
    from torch.ao.quantization.observer import PerChannelMinMaxObserver
    qc = get_default_qat_qconfig("fbgemm")._replace(
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                  qscheme=torch.per_channel_symmetric))
    example=torch.randn(1,3,IMG_SIZE,IMG_SIZE)
    return prepare_qat_fx(model.cpu(), {"":qc}, (example,))

# ───────────────────────────── main ──────────────────────────────
def main(argv:List[str]|None=None):
    ap=argparse.ArgumentParser("MobileNetV3-Small QAT trainer (CPU/CUDA/MPS)")
    ap.add_argument("--data_dir", default="filtered_imagenet2_native")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--qat_epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--width_mult", type=float, default=0.75)
    ap.add_argument("--device", default=None,
                    help="'cuda', 'mps', or 'cpu' (auto if omitted)")
    ap.add_argument("--compile", default=False, action="store_true", help="use torch.compile")
    cfg=ap.parse_args(argv)
    print(cfg)

    # auto device selection --------------------------------------------------
    if cfg.device is None:
        if torch.cuda.is_available():
            cfg.device="cuda"
        elif torch.backends.mps.is_available():
            cfg.device="mps"
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            cfg.device="cpu"
    dev=torch.device(cfg.device)
    print(f"[INFO] using device: {dev}")

    # data -------------------------------------------------------------------
    cmap_file=Path(cfg.data_dir)/"class_mapping.json"
    with open(cmap_file) as f: class_map=json.load(f)
    ncls=len(class_map); print(f"[INFO] #classes = {ncls}")

    tr_loader=get_loader(cfg.data_dir, cfg.batch, True, dev)
    vl_loader=get_loader(cfg.data_dir, cfg.batch, False, dev)

    # model + optimiser ------------------------------------------------------
    model=build_model(ncls, cfg.width_mult, dev)

    if cfg.compile:
        try:
            model=torch.compile(model, mode="reduce-overhead")
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] compile() failed ({e}); continuing uncompiled")

    crit=nn.CrossEntropyLoss(label_smoothing=0.1)
    opt =optim.SGD(model.parameters(), lr=cfg.lr,
                   momentum=0.9, weight_decay=1e-4)
    sched=CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr*0.01)
    scaler=torch.cuda.amp.GradScaler(enabled=dev.type=="cuda")

    # training loop ----------------------------------------------------------
    for ep in range(cfg.epochs):
        l=train_epoch(model,tr_loader,crit,opt,scaler,dev,ep)
        acc=evaluate(model,vl_loader,dev)
        sched.step()
        print(f"Epoch {ep+1}/{cfg.epochs}  loss {l:.4f}  "
              f"val@1 {acc*100:.2f}%  lr {opt.param_groups[0]['lr']:.5f}")

    # QAT --------------------------------------------------------------------
    print("[QAT] fusing & preparing …")
    # _fuse_mobilenetv3(model)  # manual
    qat_dev = torch.device("cpu") if dev.type == "mps" else dev
    qat=prepare_qat_fx(model).to(qat_dev)

    opt_q=optim.SGD(qat.parameters(), lr=cfg.lr*0.05,
                    momentum=0.9, weight_decay=0)

    for qep in range(cfg.qat_epochs):
        qat.train(); tot=loss_sum=0
        for img,lab in tr_loader:
            img=img.to(qat_dev, non_blocking=True,
                       memory_format=torch.channels_last if qat_dev.type=="cuda"
                       else torch.contiguous_format)
            lab=lab.to(qat_dev, non_blocking=True)
            opt_q.zero_grad(set_to_none=True)
            out=qat(img); loss=crit(out,lab)
            loss.backward(); opt_q.step()
            loss_sum+=loss.item()*img.size(0); tot+=img.size(0)
        v=evaluate(qat,vl_loader, qat_dev)
        print(f"[QAT] {qep+1}/{cfg.qat_epochs} "
              f"loss {loss_sum/tot:.4f} val@1 {v*100:.2f}%")

    # convert & export -------------------------------------------------------
    from torch.ao.quantization.quantize_fx import convert_fx
    qat.eval().cpu(); int8_model=convert_fx(qat)

    pt_path=f"mobilenetv3_w{cfg.width_mult}_int8.pt"
    torch.save(int8_model.state_dict(),pt_path)
    print(f"[SAVE] INT8 state_dict ➜ {pt_path}")

    onnx_path=pt_path.replace(".pt",".onnx")
    dummy=torch.randint(0,256,(1,3,IMG_SIZE,IMG_SIZE),dtype=torch.uint8)
    torch.onnx.export(int8_model,dummy,onnx_path,
                      input_names=["input_u8"],output_names=["logits"],
                      opset_version=18,do_constant_folding=True)
    print(f"[SAVE] ONNX (int8) ➜ {onnx_path}")

    # sanity check -----------------------------------------------------------
    with torch.no_grad():
        print("Sanity logits:", int8_model(dummy)[0,:8])

if __name__=="__main__":
    main()
