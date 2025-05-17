#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_picodet_qat.py  –  MobileNet-V3 / V4 backbone → PicoDet (Ghost-CSP-PAN)
FX-QAT → INT8 → ONNX (resize + norm + batched NMS inside).

PyTorch 2.7 • CUDA 12.8 • opset-17
"""

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

# ───────────────────── reproducibility ────────────────────────
SEED = 42
IMG_SIZE = 224
random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────── helpers ──────────────────────────────
def act_relu6(): return nn.ReLU6(inplace=True)

# ---------- 1.  GhostConv (fixed) -------------------------------------------------
class GhostConv(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, dw_size=3, ratio=2):
        super().__init__()
        self.c_out = c_out
        init_ch = min(c_out, math.ceil(c_out / ratio))
        cheap_ch = c_out - init_ch                       # may be zero
        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, k//2, bias=False),
            nn.BatchNorm2d(init_ch), act_relu6())
        self.cheap = nn.Sequential() if cheap_ch == 0 else nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_size, 1, dw_size//2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch), act_relu6())

    def forward(self, x):
        px = self.primary(x)
        if self.cheap:
            cx = self.cheap(px)
            y = torch.cat([px, cx], 1)
        else:
            y = px
        return y                                   # guaranteed c_out channels

# ---------- 2.  Depth-wise 5×5 conv ------------------------------------------------
class DWConv5x5(nn.Module):
    def __init__(self, c): super().__init__(); self.dw=nn.Conv2d(c,c,5,1,2,groups=c,bias=False); self.bn=nn.BatchNorm2d(c); self.act=act_relu6()
    def forward(self,x): return self.act(self.bn(self.dw(x)))

# ---------- 3.  CSP block -----------------------------------------------------------
class CSPBlock(nn.Module):
    def __init__(self, c, n=1):
        super().__init__()
        self.cv1, self.cv2 = GhostConv(c, c//2, 1), GhostConv(c, c//2, 1)
        self.m = nn.Sequential(*[GhostConv(c//2, c//2) for _ in range(n)])
        self.cv3 = GhostConv(c, c, 1)
    def forward(self,x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ---------- 4.  CSP-PAN (Ghost + DW 5×5) -------------------------------------------
class CSPPAN(nn.Module):
    def __init__(self, in_chs=(40,112,160), out_ch=96):
        super().__init__()
        self.reduce = nn.ModuleList([GhostConv(c, out_ch, 1) for c in in_chs])
        self.lat    = nn.ModuleList([DWConv5x5(out_ch) for _ in in_chs[:-1]])
        self.out    = nn.ModuleList([CSPBlock(out_ch) for _ in in_chs])

    def forward(self, c3, c4, c5):
        p5 = self.reduce[2](c5)
        p4 = self.reduce[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.reduce[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p4, p3 = self.lat[1](p4), self.lat[0](p3)
        p4 = p4 + F.max_pool2d(p5,2);  p3 = p3 + F.max_pool2d(p4,2)
        return tuple(self.out[i](p) for i,p in enumerate((p3,p4,p5)))

# ---------- 5.  Detection head ------------------------------------------------------
class PicoDetHead(nn.Module):
    def __init__(self, num_classes=80, reg_max=7,
                 num_feats=96, num_levels=3, max_det=100,
                 score_thresh=.3, nms_iou=.5):
        super().__init__()
        self.nc, self.reg_max, self.nl = num_classes, reg_max, num_levels
        self.max_det, self.score_th, self.iou_th = max_det, score_thresh, nms_iou
        self.strides = (8,16,32)

        self.cls_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])
        self.reg_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])

        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(num_feats, 1,     1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(num_feats, 4*(reg_max+1), 1) for _ in range(self.nl)])

        prior = -math.log((1-0.01)/0.01)
        for cp,op in zip(self.cls_pred,self.obj_pred):
            nn.init.constant_(cp.bias, prior); nn.init.constant_(op.bias, prior)

    # —— helper for DFL → l,t,r,b offsets ————————————————————
    def _dfl_to_ltrb(self, x):
        B,_,H,W = x.shape; M = self.reg_max
        x = x.view(B,4,M+1,H,W).softmax(2)
        proj = torch.arange(M+1, device=x.device, dtype=x.dtype)
        return (x*proj).sum(2)                                    # (B,4,H,W)

    # —— inference path ————————————————————————————————
    def _inference(self, cls_logits,obj_logits,reg_logits):
        boxes, scores, _ = [],[],[]
        for lv,(cl,ob,rg) in enumerate(zip(cls_logits,obj_logits,reg_logits)):
            s = self.strides[lv]
            B,C,H,W = cl.shape
            yv,xv = torch.meshgrid(torch.arange(H,device=cl.device),
                                   torch.arange(W,device=cl.device), indexing='ij')
            grid = torch.stack((xv,yv),2).view(1,1,H,W,2)*s + s*0.5        # centre coords
            ltrb = self._dfl_to_ltrb(rg)*s                                 # (B,4,H,W)
            xyxy = torch.cat(((grid[...,0]-ltrb[:,0]).unsqueeze(-1),
                              (grid[...,1]-ltrb[:,1]).unsqueeze(-1),
                              (grid[...,0]+ltrb[:,2]).unsqueeze(-1),
                              (grid[...,1]+ltrb[:,3]).unsqueeze(-1)), -1)   # (B,H,W,4)
            boxes.append(xyxy.view(B,-1,4))
            scores.append(cl.sigmoid().mul_(ob.sigmoid())                  # (B,C,H,W)
                          .permute(0,2,3,1).reshape(B,-1,self.nc))         # (B,N,C)
        boxes, scores = torch.cat(boxes,1), torch.cat(scores,1)

        out_b, out_s, out_l = [],[],[]
        for b in range(boxes.shape[0]):
            bx, sc = boxes[b], scores[b]
            mask = sc.max(1).values > self.score_th
            bx,sc = bx[mask], sc[mask]
            if bx.numel()==0:
                out_b.append(torch.zeros((self.max_det,4),device=bx.device))
                out_s.append(torch.zeros((self.max_det,),device=bx.device))
                out_l.append(torch.full((self.max_det,),-1,
                                        dtype=torch.long,device=bx.device))
                continue
            cls = sc.argmax(1)
            conf= sc.max(1).values
            keep = tvops.batched_nms(bx, conf, cls, self.iou_th)
            keep = keep[:self.max_det]
            sel_b, sel_s, sel_l = bx[keep], conf[keep], cls[keep]
            pad = self.max_det - sel_b.shape[0]
            out_b.append(F.pad(sel_b,(0,0,0,pad)))
            out_s.append(F.pad(sel_s,(0,pad)))
            out_l.append(F.pad(sel_l,(0,pad),value=-1))
        return torch.stack(out_b), torch.stack(out_s), torch.stack(out_l)

    # —— forward ————————————————————————————————————————
    def forward(self, feats:Tuple[torch.Tensor,...]):
        cls_ls, obj_ls, reg_ls = [],[],[]
        for i,f in enumerate(feats):
            c = self.cls_conv(f)
            r = self.reg_conv(f)
            cls_ls.append(self.cls_pred[i](c))
            obj_ls.append(self.obj_pred[i](c))
            reg_ls.append(self.reg_pred[i](r))

        if self.training:
            return cls_ls, obj_ls, reg_ls
        return self._inference(cls_ls,obj_ls,reg_ls)

# ---------- 6.  Backbone loader -----------------------------------------------------
def get_backbone(arch:str, ckpt:str):
    pretrained = ckpt is None
    if arch=="mnv3":
        from torchvision import models as tvm
        net = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            width_mult=1.0, dropout=0.0
        )
        feat_chs = (16, 40, 576)
    else:                       # mobilenet-v4 conv-small / medium
        import timm
        name = {"mnv4s":"mobilenetv4_conv_small",
                "mnv4m":"mobilenetv4_conv_medium"}[arch]
        net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                drop_rate=0.0, drop_path_rate=0.0)
        feat_chs = (40,112,160)        # same C3,C4,C5 dims
    print(f"Using feature channels for {arch}: {feat_chs}")
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    return net, feat_chs

# ---------- 7.  Full detector -------------------------------------------------------
class ResizeNorm(nn.Module):
    """Put resize + mean/std norm inside the exported graph."""
    def __init__(self, size=(IMG_SIZE, IMG_SIZE)):
        super().__init__()
        self.size = size
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        self.register_buffer("m",mean); self.register_buffer("s",std)

    def forward(self,x):
        if x.shape[-2:] != self.size:
            x = F.interpolate(x, self.size, mode='bilinear', align_corners=False)
        return (x - self.m)/self.s

class PicoDet(nn.Module):
    def __init__(self, backbone:nn.Module, feat_chs:Tuple[int,int,int], ncls=80):
        super().__init__()
        self.pre = ResizeNorm()
        self.backbone = backbone
        self.neck = CSPPAN(feat_chs)
        self.head = PicoDetHead(ncls)

    def old_forward(self,x):
        x = self.pre(x)
        feats, h = [], x
        for i,(name,l) in enumerate(self.backbone.features._modules.items()):
            h = l(h)
            if name in {"3","6","12"}: feats.append(h)      # C3,C4,C5
        p3,p4,p5 = self.neck(*feats)
        return self.head((p3,p4,p5))

    def forward(self,x):
        x = self.pre(x)
        feats, h = [], x
        feature_layer_names = {"3","6","14"} # Changed from {"3","6","12"}
        for i,(name,l) in enumerate(self.backbone.features._modules.items()):
            h = l(h)
            if name in feature_layer_names:
                feats.append(h)

        p3,p4,p5 = self.neck(*feats)
        return self.head((p3,p4,p5))

# ---------- 8.  Dataset / Dataloader ------------------------------------------------
import torchvision.transforms.functional as F_tv # Import functional transforms

def build_transforms(train):
    tf = []
    if train:
        # These V2 transforms operate on PIL Images
        tf.append(T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        tf.append(T_v2.RandomHorizontalFlip())
        tf.append(T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
    if not tf:
        # Add a dummy transform if no training-specific transforms were added
        tf.append(T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
        tf.append(T_v2.Lambda(lambda x: x))
    # Do NOT add Tensor conversion V2 transforms here.
    # The output of this Compose will be a Pillow Image if transforms were applied,
    # or the original Pillow Image if tf is empty.
    return T_v2.Compose(tf)

def collate(b):
    # b is a list of tuples: [(img1, target1), (img2, target2), ...]
    # img is a PIL Image because build_transforms returns a Compose that
    # operates on/passes through PIL Images.
    imgs, tg = zip(*b)

    # Manually convert each PIL Image to a float32 Tensor scaled [0, 1]
    img_tensors = []
    for img in imgs:
        # Convert PIL to uint8 Tensor (C, H, W), range [0, 255]
        img_tensor_uint8 = F_tv.pil_to_tensor(img)
        # Convert uint8 Tensor to float32 Tensor, range [0, 1]
        img_tensor_float32 = F_tv.convert_image_dtype(img_tensor_uint8, dtype=torch.float32)
        img_tensors.append(img_tensor_float32)

    # Stack the float Tensors into a batch
    stacked_imgs = torch.stack(img_tensors)

    return stacked_imgs, list(tg)

def get_loader(root, split, bsz, workers):
    ann_file=f"{root}/annotations/instances_{split}2017.json"
    imgs_dir=f"{root}/{split}2017"
    # Pass the transform pipeline to CocoDetection. It will apply these
    # transforms to the loaded PIL Image.
    ds = tvsets.CocoDetection(imgs_dir, ann_file, transform=build_transforms(split=="train"))
    return DataLoader(ds, bsz, split=="train", collate_fn=collate,
                      num_workers=workers, pin_memory=True, persistent_workers=bool(workers))

# ---------- 9.  SimOTA assigner (unchanged) ----------------------------------------
class SimOTA:
    def __init__(self,nc,ctr_rad=2.5,topk=10):
        self.nc,self.r,self.k=nc,ctr_rad,topk
    @torch.no_grad()
    def __call__(self,anchors,strides,tgt):
        # anchors:(N,2) centre xy ; tgt dict with 'boxes','labels'
        A = anchors.size(0); M = tgt["boxes"].size(0)
        if M==0:    # no gt
            return torch.zeros(A,dtype=torch.bool,device=anchors.device), \
                   torch.zeros((A,self.nc),device=anchors.device), \
                   torch.zeros((A,4),device=anchors.device)
        cxcy = (tgt["boxes"][:,:2]+tgt["boxes"][:,2:])/2
        dist  = (anchors[:,None,:]-cxcy[None,:,:]).abs().max(-1).values
        center_mask = dist < self.r*strides[:,None]
        iou = tvops.box_iou(tgt["boxes"], torch.cat([anchors-strides[:,None]/2,
                                                     anchors+strides[:,None]/2],-1))
        cls_cost = -torch.eye(self.nc,device=anchors.device)[tgt["labels"]][None,:,:]  # (1,M,C)
        cost = 1-iou + cls_cost.squeeze(0).max(-1).values.T + (~center_mask)*1e5
        matched_gt = cost.topk(self.k, largest=False).indices[:,0]
        fg = center_mask.any(1)
        cls_t = torch.zeros((A,self.nc),device=anchors.device)
        cls_t[fg, tgt["labels"][matched_gt[fg]]] = 1.
        return fg, cls_t, tgt["boxes"][matched_gt]

# ----------10.  Losses --------------------------------------------------------------
class VarifocalLoss(nn.Module):
    r"""
    Varifocal Loss (VFL) from *Generalized Focal Loss* / *VarifocalNet*.

    Every sample is weighted by its “quality target”
        • pos:  weight = q (IoU-based quality, ∈(0,1])
        • neg:  weight = α · p̂^γ  (online hard negative mining)

    Args
    ----
    alpha   – scaling factor for negatives (default 0.75)
    gamma   – focusing parameter (default 2.0)
    reduction – "none" | "sum" | "mean"
    """
    def __init__(self, alpha: float = .75, gamma: float = 2.,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self,
                logits: torch.Tensor,        # (B, A, C)  raw logits
                targets_q: torch.Tensor      # (B, A, C)  quality scores, 0 for bg
                ) -> torch.Tensor:
        p = logits.sigmoid()
        with torch.no_grad():
            pos_mask = targets_q > 0
            neg_weight = (self.alpha * p.pow(self.gamma)).type_as(targets_q)
            weight = torch.where(pos_mask, targets_q, neg_weight)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets_q, reduction="none") * weight
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss

vfl_loss = VarifocalLoss(alpha=0.75,gamma=2.0)
dfl_reg_max = 7
def dfl_loss(pred,target):
    n = pred.size(0); pred=pred.view(n*4,dfl_reg_max+1); target=target.view(-1)
    return F.cross_entropy(pred, target.long(), reduction='mean')

# ----------11.  Train / Val ---------------------------------------------------------
def train_epoch(model, loader, opt, scaler, assigner, device, epoch):
    model.train(); tot=0; t0=time.time()
    for i,(imgs,tgts) in enumerate(loader):
        imgs = imgs.to(device)
        cls_ls,obj_ls,reg_ls = model(imgs)
        bs = imgs.size(0)
        loss=0
        for b in range(bs):
            cls_p = torch.cat([c[b].permute(1,2,0).reshape(-1,model.head.nc) for c in cls_ls])
            obj_p = torch.cat([o[b].permute(1,2,0).reshape(-1)              for o in obj_ls])
            reg_p = torch.cat([r[b].permute(1,2,0).reshape(-1,4*(dfl_reg_max+1)) for r in reg_ls])
            strides=torch.cat([torch.full((c[b].numel()//model.head.nc,),s,device=device)
                               for s,c in zip(model.head.strides,cls_ls)])
            centres=[]
            for lv,(c_map,s) in enumerate(zip(cls_ls,model.head.strides)):
                H,W = cls_ls[lv].shape[2:]
                yv,xv=torch.meshgrid(torch.arange(H,device=device),
                                     torch.arange(W,device=device),indexing='ij')
                centres.append(torch.stack((xv,yv),2).reshape(-1,2)*s+s*0.5)
            centres=torch.cat(centres)
            fg, cls_t, box_t = assigner(centres,strides,
                                        {"boxes":tgts[b]["boxes"].to(device),
                                         "labels":tgts[b]["labels"].to(device)})
            if fg.sum()==0: continue
            # --- compute losses -----------------------------------------------------
            cls_loss = vfl_loss(cls_p.sigmoid()*obj_p.sigmoid().unsqueeze(-1), cls_t, cls_t)
            reg_dfl_t = ((box_t[fg]/strides[fg][:,None]).clamp(0,dfl_reg_max))
            reg_loss = dfl_loss(reg_p[fg], reg_dfl_t)
            iou_loss = tvops.complete_box_iou_loss(
                centres[fg].repeat(1,2)+torch.tensor([-1,0,1,0],device=device).view(1,4)*0,  # dummy ltrb→xyxy not needed
                box_t[fg])
            loss += cls_loss + reg_loss + iou_loss
        if loss==0: continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item()
        if i%50==0:
            print(f"E{epoch} iter{i}/{len(loader)} loss {tot/(i+1):.3f}  "
                  f"{(time.time()-t0)/(i+1):.2f}s")
    return tot/len(loader)

@torch.no_grad()
def quick_val_iou(model, loader, device):
    model.eval(); s=0;n=0
    for imgs,tgt in loader:
        bxs,scs,lbl = model(imgs.to(device))
        for i in range(imgs.size(0)):
            if tgt[i]["boxes"].numel()==0 or bxs[i].sum()==0: continue
            iou = tvops.box_iou(bxs[i], tgt[i]["boxes"].to(device)).max(1)[0]
            s+=iou.mean().item(); n+=1
    return s/n if n else 0

# ----------12.  Quantization helpers ------------------------------------------------
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping, MovingAveragePerChannelMinMaxObserver, QConfig)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
def qat_prepare(model, example):
    qmap=get_default_qat_qconfig_mapping("fbgemm")
    wobs=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,qscheme=torch.per_channel_symmetric)
    qmap=qmap.set_global(QConfig(qmap.global_qconfig.activation,wobs))
    return prepare_qat_fx(model.cpu(),qmap,(example,))

# ----------13.  Main ----------------------------------------------------------------
def main(argv:List[str]|None=None):
    pa=argparse.ArgumentParser()
    pa.add_argument("--coco_root", default="/Volumes/T7 Shield/img_data/coco")
    pa.add_argument("--backbone_ckpt", default=None)
    pa.add_argument("--arch", choices=["mnv3","mnv4s","mnv4m"], default="mnv3")
    pa.add_argument("--epochs", type=int, default=120)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--workers", type=int, default=0)
    pa.add_argument("--freeze", type=int, default=4, help="freeze first N backbone stages")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--out", default="picodet_int8_nms.onnx")
    cfg=pa.parse_args(argv)
    dev=torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # —— backbone & detector ——————————————————————————————
    backbone,feat_chs = get_backbone(cfg.arch, cfg.backbone_ckpt)
    for n,p in list(backbone.features.named_parameters())[:cfg.freeze]: p.requires_grad=False
    model=PicoDet(backbone,feat_chs).to(dev)

    # —— data / opt ——————————————————————————————————————
    tr=get_loader(cfg.coco_root,"train",cfg.batch,cfg.workers)
    vl=get_loader(cfg.coco_root,"val",cfg.batch,cfg.workers)
    opt=SGD([p for p in model.parameters() if p.requires_grad],
            lr=0.04,momentum=0.9,weight_decay=5e-5)
    sch=CosineAnnealingLR(opt,cfg.epochs,1e-5)
    scaler=torch.amp.GradScaler(enabled=dev.type=="cuda")
    assigner=SimOTA(model.head.nc)

    # —— FP-32 training ————————————————————————————————
    for ep in range(cfg.epochs):
        l=train_epoch(model,tr,opt,scaler,assigner,dev,ep)
        m=quick_val_iou(model,vl,dev)
        sch.step()
        print(f"Epoch {ep+1}/{cfg.epochs}  loss{l:.3f}  IoU{m:.3f}")

    # —— QAT fine-tune (no AMP) ————————————————————————
    example=torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    qat=qat_prepare(model, example).to(dev).train()
    opt_q=SGD(qat.parameters(),lr=0.004,momentum=0.9,weight_decay=1e-5)
    scaler_q=torch.amp.GradScaler(enabled=False)
    for q in range(5):
        lq=train_epoch(qat,tr,opt_q,scaler_q,assigner,dev,f"Q{q}")
        print(f"[QAT] {q+1}/5 loss{lq:.3f}")

    qat.cpu().eval()
    int8=convert_fx(qat)

    # —— ONNX export —————————————————————————————————
    torch.onnx.export(int8, example, cfg.out,
                      input_names=["images"],
                      output_names=["boxes","scores","labels"],
                      dynamic_axes={"images":{0:"B",2:"H",3:"W"},
                                    "boxes":{0:"B"},
                                    "scores":{0:"B"},
                                    "labels":{0:"B"}},
                      opset_version=17, do_constant_folding=True)
    print(f"[SAVE] {cfg.out}")

if __name__=="__main__": main()
