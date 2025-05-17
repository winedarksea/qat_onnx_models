# train_picodet_qat.py – minimal pipeline: COCO ➜ FP32 ➜ QAT ➜ INT8 ➜ ONNX (with NMS)
from __future__ import annotations
import argparse, math, os, random, time, warnings
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T_v2
import torchvision.datasets as tvsets
import torchvision.ops as tvops
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from picodet_lib import (
    PicoDet, get_backbone, VarifocalLoss, dfl_loss,
    build_dfl_targets
)

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224

# ───────────────────── data & transforms ───────────────────────
import torchvision.transforms.functional as F_tv

def build_transforms(train: bool):
    tfs: List[torch.nn.Module] = []
    if train:
        tfs += [
            T_v2.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), antialias=True),
            T_v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T_v2.RandomHorizontalFlip()
        ]
    else:
        tfs.append(T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
    return T_v2.Compose(tfs)


def collate(batch):
    imgs, tgts = zip(*batch)
    img_ts = []
    for im in imgs:
        t = F_tv.pil_to_tensor(im).contiguous()  # uint8
        img_ts.append(t)
    stacked = torch.stack(img_ts)
    # normalise boxes to absolute pixels (they already are). nothing else.
    return stacked, list(tgts)


def get_loader(root: str, split: str, bsz: int, workers: int = 0):
    ds = tvsets.CocoDetection(
        img_folder := f"{root}/{split}2017",
        ann_file := f"{root}/annotations/instances_{split}2017.json",
        transform=build_transforms(split == 'train')
    )
    return DataLoader(ds, batch_size=bsz, shuffle=split == 'train',
                      collate_fn=collate, num_workers=workers, pin_memory=True,
                      persistent_workers=bool(workers))

# ───────────────────── assigner (SimOTA, cached) ────────────────
class SimOTACache:
    def __init__(self, nc: int, ctr: float = 2.5, topk: int = 10):
        self.nc, self.r, self.k = nc, ctr, topk
        self.cache = {}

    @torch.no_grad()
    def __call__(self, f_shapes: Tuple[Tuple[int, int, int], ...], device: torch.device,
                 tgt: dict):
        # f_shapes = tuple of (H,W,stride)
        centres, strides = [], []
        for (H, W, s) in f_shapes:
            key = (H, W, s, device)
            if key not in self.cache:
                yv, xv = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device), indexing='ij')
                c = torch.stack((xv, yv), 2).reshape(-1, 2) * s + s * 0.5
                self.cache[key] = c
            centres.append(self.cache[key])
            strides.append(torch.full((H * W,), s, device=device))
        centres = torch.cat(centres)
        strides = torch.cat(strides)

        A = centres.size(0)
        M = tgt['boxes'].size(0)
        if M == 0:
            return torch.zeros(A, dtype=torch.bool, device=device), \
                   torch.zeros((A, self.nc), device=device), \
                   torch.zeros((A, 4), device=device)
        cxcy = (tgt['boxes'][:, :2] + tgt['boxes'][:, 2:]) / 2
        dist = (centres[:, None, :] - cxcy[None, :, :]).abs().max(-1).values
        centre_mask = dist < self.r * strides[:, None]
        print(f"Shape of centres: {centres.shape}")
        print(f"Shape of strides: {strides.shape}")
        
        iou = tvops.box_iou(tgt['boxes'], torch.cat([centres - strides[:, None] / 2,
                                                    centres + strides[:, None] / 2], -1))
        print(f"Shape of tgt['boxes']: {tgt['boxes'].shape}")
        print(f"Shape of anchor_candidate_boxes: {torch.cat([centres - strides[:, None] / 2, centres + strides[:, None] / 2], -1).shape}")
        print(f"Shape of iou: {iou.shape}")
        cls_cost = -torch.eye(self.nc, device=device)[tgt['labels']][None, :, :].max(-1).values.T
        print(f"Shape of cls_cost: {cls_cost.shape}")
        print(f"Shape of centre_mask: {centre_mask.shape}")
        print(f"Shape of iou: {iou.shape}")
        cost = 1 - iou + cls_cost + (~centre_mask.T) * 1e5
        matched_gt = cost.topk(self.k, largest=False).indices[:, 0]
        fg = centre_mask.any(1)
        cls_t = torch.zeros((A, self.nc), device=device)
        cls_t[fg, tgt['labels'][matched_gt[fg]]] = 1.
        return fg, cls_t, tgt['boxes'][matched_gt]

# ───────────────────── train / val loops ────────────────────────
def train_epoch(model: PicoDet, loader, opt, scaler, assigner: SimOTACache,
                device: torch.device, epoch: int):
    model.train(); t0, tot = time.time(), 0.
    for i, (imgs, tgts) in enumerate(loader):
        imgs = imgs.to(device)
        cls_ls, obj_ls, reg_ls = model(imgs)  # training path returns lists
        bs = imgs.size(0)
        loss = 0.
        # prepare feature map shapes once per batch
        fmap_shapes = []
        for lv in range(len(cls_ls)):
            H, W = cls_ls[lv].shape[2:]
            s = model.head.strides[lv]
            fmap_shapes.append((H, W, s))

        for b in range(bs):
            cls_p = torch.cat([c[b].permute(1, 2, 0).reshape(-1, model.head.nc) for c in cls_ls])
            obj_p = torch.cat([o[b].permute(1, 2, 0).reshape(-1) for o in obj_ls])
            reg_p = torch.cat([r[b].permute(1, 2, 0).reshape(-1, 4 * (model.head.reg_max + 1)) for r in reg_ls])

            # assign targets
            print(tgts[b])
            annots = tgts[b]
            if len(annots) == 0:
                continue
            
            boxes = torch.tensor(
                [[x, y, x + w, y + h] for a in annots for (x, y, w, h) in [a['bbox']]],
                dtype=torch.float32, device=device
            )
            labels = torch.tensor([a['category_id'] for a in annots], dtype=torch.int64, device=device)
            
            fg, cls_t, box_t = assigner(fmap_shapes, device, {
                'boxes': boxes,
                'labels': labels
            })

            if fg.sum() == 0:
                continue

            joint_logits = cls_p + obj_p.unsqueeze(-1)
            cls_loss = VarifocalLoss()(joint_logits, cls_t)

            # DFL targets & loss
            strides_fg = torch.tensor([s for (_, _, s) in fmap_shapes for _ in range((IMG_SIZE // s) ** 2)],
                                      device=device)[fg]
            offsets_t = (box_t[fg] / strides_fg.unsqueeze(-1)).clamp(0, model.head.reg_max)
            dfl_t = build_dfl_targets(offsets_t, model.head.reg_max)  # (N,4,M+1)
            reg_logits_fg = reg_p[fg]
            reg_loss = dfl_loss(reg_logits_fg, dfl_t)

            # IoU loss
            # decode predicted boxes: expectation method
            pred_off = model.head._dfl_to_ltrb(reg_logits_fg.view(-1, 4 * (model.head.reg_max + 1)).unsqueeze(0))[0]
            centres_fg = torch.cat([assigner.cache[(H, W, s, device)] for (H, W, s) in fmap_shapes])[fg]
            boxes_pred = torch.stack((centres_fg[:, 0] - pred_off[:, 0], centres_fg[:, 1] - pred_off[:, 1],
                                       centres_fg[:, 0] + pred_off[:, 2], centres_fg[:, 1] + pred_off[:, 3]), 1)
            iou_loss = tvops.complete_box_iou_loss(boxes_pred, box_t[fg])

            loss += cls_loss + reg_loss + iou_loss
        if loss == 0:
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item()
        if i % 50 == 0:
            print(f"E{epoch} {i:04d}/{len(loader)} loss {tot / (i + 1):.3f} {(time.time() - t0) / (i + 1):.2f}s")
    return tot / len(loader)

@torch.no_grad()
def quick_val_iou(model: PicoDet, loader, device):
    model.eval(); s = n = 0.
    for imgs, tgt in loader:
        bx, sc, lb = model(imgs.to(device))
        for i in range(imgs.size(0)):
            if tgt[i]['boxes'].numel() == 0:
                continue
            iou = tvops.box_iou(bx[i], tgt[i]['boxes'].to(device)).max(1)[0]
            s += iou.mean().item(); n += 1
    return s / n if n else 0.

# ───────────────────── QAT helpers ───────────────────────────────
from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfig
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

def qat_prepare(model: nn.Module, example: torch.Tensor):
    qmap = get_default_qat_qconfig_mapping('x86')
    wobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    aobs = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_channel_affine)
    qmap = qmap.set_global(QConfig(aobs, wobs))
    # skip preprocess from quant
    qmap = qmap.set_module_name('pre', None)
    return prepare_qat_fx(model.cpu(), qmap, (example,))

# ───────────────────── main ─────────────────────────────────────

def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', choices=['mnv3', 'mnv4s', 'mnv4m'], default='mnv3')
    pa.add_argument('--epochs', type=int, default=50)
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8.onnx')
    cfg = pa.parse_args(argv)

    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(cfg.device)
    print(f'[INFO] device = {dev}')

    # backbone & detector
    ckpt = None
    backbone, feat_chs = get_backbone(cfg.arch, ckpt=ckpt)
    model = PicoDet(backbone, feat_chs).to(dev)

    # data
    tr = get_loader(cfg.coco_root, 'train', cfg.batch, cfg.workers)
    vl = get_loader(cfg.coco_root, 'val', cfg.batch, cfg.workers)

    opt = SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-5)
    sch = CosineAnnealingLR(opt, cfg.epochs, 5e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    assigner = SimOTACache(model.head.nc)

    # FP‑32 training (brief)
    for ep in range(cfg.epochs):
        l = train_epoch(model, tr, opt, scaler, assigner, dev, ep)
        m = quick_val_iou(model, vl, dev)
        sch.step()
        print(f'Epoch {ep + 1}/{cfg.epochs}  loss {l:.3f}  IoU {m:.3f}')

    # -------- QAT fine‑tune (5 epochs) -------------------------
    example = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    qat = qat_prepare(model, example).to(dev).train()
    for p in qat.backbone.parameters():
        p.requires_grad = True  # unfreeze for fake‑quant stats
    opt_q = SGD(qat.parameters(), lr=0.002, momentum=0.9)
    scaler_q = torch.amp.GradScaler(enabled=False)
    for qep in range(5):
        lq = train_epoch(qat, tr, opt_q, scaler_q, assigner, dev, qep)
        print(f'[QAT] {qep + 1}/5  loss {lq:.3f}')

    qat.cpu().eval()
    int8 = convert_fx(qat)

    # ---------------- ONNX export ------------------------------
    int8.eval()
    torch.onnx.export(int8, example, cfg.out,
                      input_names=['images'],
                      output_names=['boxes', 'scores', 'labels'],
                      dynamic_axes={'images': {0: 'B', 2: 'H', 3: 'W'},
                                    'boxes': {0: 'B'},
                                    'scores': {0: 'B'},
                                    'labels': {0: 'B'}},
                      opset_version=18, do_constant_folding=True)
    print(f'[SAVE] ONNX → {cfg.out}')

if __name__ == '__main__':
    main()
