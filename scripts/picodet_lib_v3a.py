# picodet_lib.py Refactored for unified anchors + ATSS
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops


try:
    folder_to_add = r"/home/colin/qat_onnx_models/scripts"
    sys.path.append(folder_to_add)
    from customMobilenetNetv4 import MobileNetV4ConvSmallPico
except ImportError:
    print("Warning: customMobilenetNetv4.py not found. 'mnv4_custom' backbone will not be available.")
    MobileNetV4ConvSmallPico = None

class GhostConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            k: int = 1,
            s: int = 1,
            dw_size: int = 3,
            ratio: float = 2.0,
            inplace_act: bool = False
    ):
        super().__init__()
        init_ch = min(math.ceil(c_out / ratio), c_out)
        cheap_ch = c_out - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU6(inplace=inplace_act)
        )
        self.cheap = None
        if cheap_ch > 0:
            self.cheap = nn.Sequential(
                nn.Conv2d(init_ch, cheap_ch, dw_size, 1, dw_size // 2,
                          groups=init_ch, bias=False),
                nn.BatchNorm2d(cheap_ch),
                nn.ReLU6(inplace=inplace_act)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.primary(x)
        if self.cheap:
            y2 = self.cheap(y1)
            return torch.cat([y1, y2], dim=1)
        return y1

    @property
    def out_channels(self) -> int:
        oc = self.primary[0].out_channels
        if self.cheap:
            oc += self.cheap[0].out_channels
        return oc

class DWConv(nn.Module):
    def __init__(self, c: int, k: int = 5, inplace_act: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=inplace_act)
    def forward(self, x): return self.act(self.bn(self.dw(x)))

class CSPBlock(nn.Module):
    def __init__(self, c: int, n: int = 1, m_k: int = 1, inplace_act: bool = False):
        super().__init__()
        self.cv1 = GhostConv(c, c//2, 1, inplace_act=inplace_act)
        self.cv2 = GhostConv(c, c//2, 1, inplace_act=inplace_act)
        self.m   = nn.Sequential(*[
            GhostConv(c//2, c//2, k=m_k, inplace_act=inplace_act)
            for _ in range(n)
        ])
        self.cv3 = GhostConv(c, c, 1, inplace_act=inplace_act)
        for m in self.cv3.modules():
            if isinstance(m, nn.BatchNorm2d): nn.init.zeros_(m.weight)
    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))

class CSPPAN(nn.Module):
    def __init__(self, in_chs=(40,112,160), out_ch=96, inplace_act: bool = False):
        super().__init__()
        self.reduce = nn.ModuleList([
            GhostConv(c, out_ch, 1, inplace_act=inplace_act) for c in in_chs
        ])
        self.lat = nn.ModuleList([
            DWConv(out_ch, k=5, inplace_act=inplace_act)
            for _ in in_chs[:-1]
        ])
        lst = len(in_chs)-1
        self.out = nn.ModuleList([
            CSPBlock(
                out_ch,
                n=2 if i==lst else 1,
                m_k=3 if i==lst else 1,
                inplace_act=inplace_act
            ) for i in range(len(in_chs))
        ])
    def forward(self, c3, c4, c5):
        p5 = self.reduce[2](c5)
        p4 = self.reduce[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.reduce[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p4, p3 = self.lat[1](p4), self.lat[0](p3)
        p4 = p4 + F.max_pool2d(p3, 2)
        p5 = p5 + F.max_pool2d(p4, 2)
        return self.out[0](p3), self.out[1](p4), self.out[2](p5)

class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2., reduction: str = 'mean'):
        super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = logits.sigmoid()
        with torch.no_grad():
            w = torch.where(targets>0, targets, self.alpha * p.pow(self.gamma))
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=w, reduction='none')
        return loss.sum() if self.reduction=='sum' else loss.mean()

def build_dfl_targets(offsets: torch.Tensor, reg_max: int) -> torch.Tensor:
    x = offsets.clone().clamp(0, reg_max)
    l = x.floor().long()
    r = (l+1).clamp(max=reg_max)
    wr = x - l.float()
    wl = 1-wr
    one_hot_l = F.one_hot(l, reg_max+1).float() * wl.unsqueeze(-1)
    one_hot_r = F.one_hot(r, reg_max+1).float() * wr.unsqueeze(-1)
    return one_hot_l + one_hot_r

def dfl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    N,_,M1 = target.shape
    pred = pred.view(N*4, M1)
    tgt  = target.view(N*4, M1)
    return F.kl_div(F.log_softmax(pred,1), tgt, reduction='batchmean')

def generate_anchors(
    fmap_shapes: List[Tuple[int,int,float]],
    strides_buffer: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    centers, strides = [], []
    for (H,W,s), stride in zip(fmap_shapes, strides_buffer.tolist()):
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yv,xv = torch.meshgrid(y,x, indexing='ij')
        ctr = (torch.stack((xv,yv),2).reshape(-1,2) + 0.5) * stride
        centers.append(ctr)
        strides.append(torch.full((H*W,), stride, device=device))
    return torch.cat(centers,0), torch.cat(strides,0)

class ATSSAssigner:
    def __init__(self, top_k: int = 9):
        self.top_k = top_k
    @torch.no_grad()
    def assign(
        self,
        anchor_centers: torch.Tensor,
        anchor_strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = anchor_centers.device
        A = anchor_centers.size(0)
        M = gt_boxes.size(0)
        if M==0:
            return (
                torch.zeros(A, dtype=torch.bool, device=device),
                torch.full((A,), -1, device=device),
                torch.zeros((A,4), device=device),
                torch.zeros((A,), device=device)
            )
        # IoU
        s2 = anchor_strides.unsqueeze(1)/2
        anchors = torch.cat([anchor_centers-s2, anchor_centers+s2],1)
        ious = tvops.box_iou(gt_boxes, anchors)
        # distances
        gt_ctr = (gt_boxes[:,:2]+gt_boxes[:,2:])/2
        dist = torch.cdist(anchor_centers, gt_ctr, p=2).t()
        # top-k by distance
        k = min(self.top_k, A)
        candidate_idxs = torch.stack([
            torch.topk(-dist[m], k).indices for m in range(M)
        ],0)
        candidate_ious = torch.stack([
            ious[m, candidate_idxs[m]] for m in range(M)
        ],0)
        thr = candidate_ious.mean(1) + candidate_ious.std(1)
        pos_mask = ious >= thr.unsqueeze(1)
        # ensure each GT has at least one
        for m in range(M):
            if pos_mask[m].sum()==0:
                pos_mask[m, ious[m].argmax()]=True
        iou_max, argmax_m = ious.max(0)
        fg = pos_mask.t().any(1)
        gt_inds = argmax_m[fg]
        gt_lbl_out = torch.full((A,), -1, dtype=torch.long, device=device)
        gt_box_out = torch.zeros((A,4), device=device)
        iou_out    = torch.zeros((A,), device=device)
        gt_lbl_out[fg] = gt_labels[gt_inds]
        gt_box_out[fg] = gt_boxes[gt_inds]
        iou_out[fg]    = iou_max[fg]
        return fg, gt_lbl_out, gt_box_out, iou_out

class PicoDetHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 8,
        num_feats: int = 96,
        num_levels: int = 3,
        img_size: int = 224,
        score_thresh: float = 0.05,
        nms_iou: float = 0.6,
    ):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.nl = num_levels
        self.score_th = score_thresh
        self.iou_th = nms_iou
        strides = torch.tensor([8,16,32][:num_levels], dtype=torch.float32)
        self.register_buffer('strides_buffer', strides, persistent=False)
        proj = torch.arange(reg_max+1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', proj, persistent=False)
        self.cls_conv = nn.Sequential(
            GhostConv(num_feats, num_feats, k=3),
            *[GhostConv(num_feats, num_feats) for _ in range(2)]
        )
        self.reg_conv = nn.Sequential(*[
            GhostConv(num_feats, num_feats) for _ in range(2)
        ])
        self.cls_pred = nn.ModuleList([
            nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)
        ])
        self.obj_pred = nn.ModuleList([
            nn.Conv2d(num_feats, 1, 1) for _ in range(self.nl)
        ])
        self.reg_pred = nn.ModuleList([
            nn.Conv2d(num_feats, 4*(reg_max+1), 1) for _ in range(self.nl)
        ])
        self._initialize_biases()
    def _initialize_biases(self):
        prior = 0.01
        bias = -math.log((1-prior)/prior)
        for m in self.cls_pred: nn.init.constant_(m.bias, bias)
        for m in self.obj_pred: nn.init.constant_(m.bias, 0.0)
    def _dfl_to_ltrb_inference(self, x):
        b,n,_ = x.shape
        y = x.view(b,n,4,self.reg_max+1).softmax(3)
        proj = self.dfl_project_buffer.view(1,1,1,-1)
        return (y*proj).sum(3)
    @staticmethod
    def dfl_decode_for_training(
        x_reg_logits: torch.Tensor,
        dfl_project_buffer: torch.Tensor,
        reg_max_val: int
    ) -> torch.Tensor:
        if x_reg_logits.ndim != 2:
            raise ValueError(f"Expected 2D logits, got {x_reg_logits.ndim}D")
        n = x_reg_logits.shape[0]
        x = x_reg_logits.view(n, 4, reg_max_val+1)
        x_soft = x.softmax(2)
        proj = dfl_project_buffer.view(1,1,-1)
        return (x_soft * proj).sum(2)
    def _decode_predictions_for_level(self, cls_l, obj_l, reg_l, i):
        B,_,H,W = cls_l.shape
        stride = self.strides_buffer[i]
        cls_p = cls_l.permute(0,2,3,1).reshape(B,-1,self.nc)
        obj_p = obj_l.permute(0,2,3,1).reshape(B,-1,1)
        reg_p = reg_l.permute(0,2,3,1).reshape(B,-1,4*(self.reg_max+1))
        ltrb = self._dfl_to_ltrb_inference(reg_p) * stride
        yv,xv = torch.meshgrid(
            torch.arange(H, device=cls_l.device),
            torch.arange(W, device=cls_l.device), indexing='ij'
        )
        ctr = (torch.stack((xv,yv),2).view(-1,2)+0.5)*stride
        x1 = ctr[:,0].unsqueeze(0)-ltrb[:,:,0]
        y1 = ctr[:,1].unsqueeze(0)-ltrb[:,:,1]
        x2 = ctr[:,0].unsqueeze(0)+ltrb[:,:,2]
        y2 = ctr[:,1].unsqueeze(0)+ltrb[:,:,3]
        boxes = torch.stack((x1,y1,x2,y2),-1)
        scores = (cls_p+obj_p).sigmoid()
        return boxes, scores
    def forward(self, feats):
        cls_raw,obj_raw,reg_raw = [],[],[]
        for i,f in enumerate(feats):
            c = self.cls_conv(f)
            r = self.reg_conv(f)
            cls_raw.append(self.cls_pred[i](c))
            obj_raw.append(self.obj_pred[i](c))
            reg_raw.append(self.reg_pred[i](r))
        if self.training:
            return tuple(cls_raw), tuple(obj_raw), tuple(reg_raw), tuple(self.strides_buffer)
        else:
            boxes, scores = [], []
            for i in range(self.nl):
                b,s = self._decode_predictions_for_level(
                    cls_raw[i], obj_raw[i], reg_raw[i], i
                )
                boxes.append(b); scores.append(s)
            return torch.concat(boxes,1), torch.concat(scores,1)

class PicoDet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_chs: Tuple[int,int,int],
        num_classes: int = 80,
        neck_out_ch: int = 96,
        img_size: int = 224,
        head_reg_max: int = 8
    ):
        super().__init__()
        self.pre = ResizeNorm((img_size,img_size))
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch)
        self.head = PicoDetHead(
            num_classes=num_classes,
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=len(feat_chs),
            img_size=img_size
        )
    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        c3,c4,c5 = self.backbone(x)
        p3,p4,p5 = self.neck(c3,c4,c5)
        return self.head((p3,p4,p5))

class ResizeNorm(nn.Module):
    def __init__(self, size: Tuple[int,int], mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        super().__init__()
        self.size = size
        self.register_buffer('m', torch.tensor(mean).view(1,3,1,1))
        self.register_buffer('s', torch.tensor(std).view(1,3,1,1))
    def forward(self, x: torch.Tensor):
        x = x.float()/255.0
        x = F.interpolate(x, self.size, mode='bilinear', align_corners=False)
        return (x - self.m)/self.s

@torch.no_grad()
def _get_dynamic_feat_chs(model: nn.Module, img_size: int, device: torch.device) -> Tuple[int,int,int]:
    model.eval()
    inp = torch.randn(1,3,img_size,img_size,device=device)
    orig_dev = next(model.parameters()).device if list(model.parameters()) else 'cpu'
    model.to(device)
    feats = model(inp)
    model.to(orig_dev)
    if not (isinstance(feats,(list,tuple)) and len(feats)==3):
        raise ValueError(f"Expected 3 feats, got {len(feats)}")
    return tuple(f.shape[1] for f in feats)

def get_backbone(arch: str, ckpt: str|None, img_size: int=224):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == "mnv4c":
        if MobileNetV4ConvSmallPico is None: raise ImportError("Cannot create 'mnv4_custom' backbone.")
        print("[INFO] Creating custom MobileNetV4-Small backbone for PicoDet.")
        feature_indices = (MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p3_s8'], MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p4_s16'], MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p5_s32'])
        net = MobileNetV4ConvSmallPico(width_multiplier=1.2, features_only=True, out_features_indices=feature_indices)
        ckpt_path = "mobilenet_w1_2_mnv4c_pretrained_drp0_2_fp32_backbone.pt"
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading pre-trained backbone weights from: {ckpt_path}")
            backbone_sd = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = net.load_state_dict(backbone_sd, strict=False)
            if missing or unexpected: warnings.warn(f"Mismatch loading backbone weights. Missing: {missing}, Unexpected: {unexpected}")
        else:
            print("[INFO] Initializing backbone with random weights.")
        feat_chs = _get_dynamic_feat_chs(net, img_size, dev)
        print(f"[INFO] Custom MNv4 feature channels: {feat_chs}")
    else:
        raise ValueError(f"Unknown arch {arch}")
    net.train()
    return net, feat_chs

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
__all__ = [
    'GhostConv','DWConv','CSPBlock','CSPPAN',
    'VarifocalLoss','build_dfl_targets','dfl_loss',
    'generate_anchors','ATSSAssigner',
    'PicoDetHead','PicoDet',
    'ResizeNorm','TVExtractorWrapper','pick_nodes_by_stride','_get_dynamic_feat_chs','get_backbone'
]
