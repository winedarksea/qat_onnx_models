# PyTorch 2.7 / opset 18
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

# --- Custom Backbone Import ---
try:
    # User should ensure customMobilenetNetv4.py is accessible.
    # The original script had a specific path; adjust if necessary or ensure it's in PYTHONPATH.
    folder_to_add = r"/Users/colincatlin/Documents-NoCloud/qat_onnx_models/scripts" # User-specific path
    # Check if the path and file exist before trying to add to sys.path and import
    custom_module_path = os.path.join(folder_to_add, "customMobilenetNetv4.py")
    if os.path.exists(custom_module_path):
        if folder_to_add not in sys.path:
             sys.path.insert(0, folder_to_add) # Insert at 0 to prioritize
        from customMobilenetNetv4 import MobileNetV4ConvSmallPico
        print(f"[INFO] Successfully imported customMobilenetNetv4 from {folder_to_add}")
    else:
        print(f"Warning: Did not find customMobilenetNetv4.py at {custom_module_path}. 'mnv4c' backbone will not be available.")
        MobileNetV4ConvSmallPico = None
except ImportError as e:
    print(f"Warning: Failed to import customMobilenetNetv4.py ({e}). 'mnv4c' backbone will not be available.")
    MobileNetV4ConvSmallPico = None
except Exception as e: # Catch other potential errors like incorrect path structure
    print(f"An error occurred during custom backbone import setup: {e}")
    MobileNetV4ConvSmallPico = None

# ───────────────────────────── layers ──────────────────────────────
class GhostConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            k: int = 1,
            s: int = 1,
            dw_size: int = 3,
            ratio: int = 2,
            inplace_act: bool = False
         ):
        super().__init__()
        self.c_out = c_out
        init_ch = math.ceil(c_out / ratio)
        init_ch = min(init_ch, c_out)
        self.cheap_ch = c_out - init_ch

        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU6(inplace=inplace_act)  # nn.Hardswish(inplace=inplace_act)
        )
        if self.cheap_ch > 0:
            self.cheap = nn.Sequential(
                nn.Conv2d(init_ch, self.cheap_ch, dw_size, 1, dw_size // 2,
                          groups=init_ch, bias=False),
                nn.BatchNorm2d(self.cheap_ch), nn.ReLU6(inplace=inplace_act)
            )
        else:
            self.cheap = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_primary = self.primary(x)
        if self.cheap:
            y_cheap = self.cheap(y_primary)
            return torch.cat([y_primary, y_cheap], 1)
        return y_primary

class DWConv(nn.Module):
    def __init__(
            self, c: int, k: int = 5,
            inplace_act: bool = False
    ):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=inplace_act)

    def forward(self, x): return self.act(self.bn(self.dw(x)))

class CSPBlock(nn.Module):
    def __init__(self, c: int, n: int = 1, m_k: int = 1, inplace_act: bool = False):
        super().__init__()
        self.cv1 = GhostConv(c, c // 2, 1, inplace_act=inplace_act)
        self.cv2 = GhostConv(c, c // 2, 1, inplace_act=inplace_act)
        self.m = nn.Sequential(*[GhostConv(c // 2, c // 2, k=m_k, inplace_act=inplace_act) for _ in range(n)])
        self.cv3 = GhostConv(c, c, 1, inplace_act=inplace_act)
        # Initialize last BN in residual branch to zero, helps stability
        for m in self.cv3.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ───────────────────────────── neck ────────────────────────────────
class CSPPAN(nn.Module):
    def __init__(self, in_chs=(40, 112, 160), out_ch=96, inplace_act: bool = False):
        super().__init__()
        self.reduce = nn.ModuleList([GhostConv(c, out_ch, 1, inplace_act=inplace_act) for c in in_chs])
        self.lat    = nn.ModuleList([DWConv(out_ch, k=5, inplace_act=inplace_act) for _ in in_chs[:-1]])
        lst_ly = len(in_chs) - 1
        self.out = nn.ModuleList([
            CSPBlock(out_ch, n=2 if i == lst_ly else 1, m_k=3 if i == lst_ly else 1, inplace_act=inplace_act) for i in range(len(in_chs))
        ])

    def forward(self, c3, c4, c5):
        p5 = self.reduce[2](c5)
        p4 = self.reduce[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.reduce[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p4, p3 = self.lat[1](p4), self.lat[0](p3)
        p4 = p4 + F.max_pool2d(p3, 2)
        p5 = p5 + F.max_pool2d(p4, 2)
        return (self.out[0](p3), self.out[1](p4), self.out[2](p5))

# ─────────────────────── losses ──────────────────
class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2., reduction: str = 'mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets_q: torch.Tensor):
        p = logits.sigmoid()
        with torch.no_grad():
            weight = torch.where(targets_q > 0, targets_q, self.alpha * p.pow(self.gamma))
        loss = F.binary_cross_entropy_with_logits(logits, targets_q, weight, reduction='none')
        if self.reduction == 'sum': return loss.sum()
        if self.reduction == 'mean': return loss.mean()
        return loss

def build_dfl_targets(offsets: torch.Tensor, reg_max: int) -> torch.Tensor:
    # Ensure offsets are within [0, reg_max - eps] to avoid issues with floor/ceil at reg_max
    x = offsets.clamp_(0, reg_max - 1e-6) 
    l = x.floor().long()
    r = l + 1
    # Clamp r to reg_max to prevent out-of-bounds in one_hot for r if x is very close to reg_max
    r.clamp_(max=reg_max)
    
    w_r = x - l.float()
    w_l = 1. - w_r
    
    # Target shape: (..., reg_max + 1)
    one_hot_l = F.one_hot(l, num_classes=reg_max + 1).float() * w_l.unsqueeze(-1)
    one_hot_r = F.one_hot(r, num_classes=reg_max + 1).float() * w_r.unsqueeze(-1)
    return one_hot_l + one_hot_r

def dfl_loss(
    pred: torch.Tensor,                # (B, 4*(R+1))  logits
    target: torch.Tensor,              # (B, 4, R+1)   soft / one-hot
    reduction: str = 'mean',           # 'none' | 'sum' | 'mean'
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Robust KL-divergence implementation used by YOLOv8/PP-YOLOE style DFL.

    • Any row in `target` whose total mass is 0 is *ignored* – it contributes
      neither loss nor gradient.  This removes the common NaN source when
      anchors have no positive assignment.
    • The function never returns NaN; when no row is valid the return value is
      `0.` (attached to the computation graph).

    Args
    ----
    pred : raw logits shaped (N, 4*(R+1)) **or** already flattened
    target : soft labels shaped (N, 4, R+1)
    reduction : 'none' | 'sum' | 'mean'
    eps : small constant for numerical stability
    """
    # ---- reshape to (N*4, bins) --------------------------------------
    bins = target.size(-1)
    q = pred.view(-1, bins)            # logits
    p = target.view(-1, bins)          # probs / un-normalised mass

    # ---- make p a proper probability distribution where valid --------
    p = torch.clamp(p, min=0)
    row_sum = p.sum(dim=1, keepdim=True)          # (N*,1)
    valid   = row_sum.squeeze(1) > eps            # Bool mask

    # Early exit (keeps graph):
    if not valid.any():
        return (q.sum() * 0) if reduction != 'none' else q.new_zeros(q.size(0))

    p_norm = torch.where(valid.unsqueeze(1), p / row_sum.clamp(min=eps), p)

    # ---- KL(P‖Q)  =  Σ_i p_i (log p_i − log q_i) ---------------------
    log_q = F.log_softmax(q + eps, dim=1)         # numeric safety

    kl = F.kl_div(log_q[valid], p_norm[valid], reduction='none').sum(dim=1)

    # ---- reduction ---------------------------------------------------
    if reduction == 'none':
        # return per-row loss with zeros for invalid rows
        out = q.new_zeros(q.size(0))
        out[valid] = kl
        return out

    if reduction == 'sum':
        return kl.sum()

    # default: mean over *valid* rows
    return kl.mean()

# ───────────────────── New Centralized Anchor Manager ────────────────
class AnchorManager(nn.Module):
    """Generates and provides anchor points and strides for all FPN levels."""
    def __init__(self, img_size: int, num_levels: int, strides: Tuple[int, ...]):
        super().__init__()
        if len(strides) != num_levels:
            raise ValueError("Length of strides must match num_levels")

        anchor_points_list, strides_list = [], []
        for s in strides:
            h, w = math.ceil(img_size / s), math.ceil(img_size / s)
            yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            # Shape (H*W, 2)
            grid = torch.stack((xv, yv), dim=2).reshape(-1, 2)
            anchor_points = (grid.float() + 0.5) * s
            level_strides = torch.full((len(anchor_points), 1), float(s))

            anchor_points_list.append(anchor_points)
            strides_list.append(level_strides)
        
        # Non-persistent buffers are not part of the model's state_dict
        self.register_buffer('anchor_points_cat', torch.cat(anchor_points_list, dim=0), persistent=False)
        self.register_buffer('strides_cat', torch.cat(strides_list, dim=0), persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.anchor_points_cat, self.strides_cat

# ───────────────────── Refactored Head ────────────────
class PicoDetHead(nn.Module):
    def __init__(self, num_classes: int = 80,
                 reg_max: int = 8,
                 num_feats: int = 96,
                 num_levels: int = 3,
                 inplace_act: bool = False):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.nl = num_levels
        
        dfl_proj = torch.arange(self.reg_max + 1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', dfl_proj, persistent=False)

        # A single common path for features can be more efficient
        self.cls_conv = nn.Sequential(
            GhostConv(num_feats, num_feats, k=3, inplace_act=inplace_act),
            GhostConv(num_feats, num_feats, inplace_act=inplace_act),
            GhostConv(num_feats, num_feats, inplace_act=inplace_act)
        )
        self.reg_conv = nn.Sequential(
            GhostConv(num_feats, num_feats, inplace_act=inplace_act),
            GhostConv(num_feats, num_feats, inplace_act=inplace_act)
        )
        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(num_feats, 1, 1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(num_feats, 4 * (self.reg_max + 1), 1) for _ in range(self.nl)])
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        cls_prior, obj_prior = 0.01, 0.1
        cls_bias = -math.log((1. - cls_prior) / cls_prior)
        obj_bias = -math.log((1 - obj_prior) / obj_prior)
        for conv in self.cls_pred: nn.init.constant_(conv.bias, cls_bias)
        for conv in self.obj_pred: nn.init.constant_(conv.bias, obj_bias)

    @staticmethod
    def _dfl_to_ltrb(reg_logits: torch.Tensor, dfl_project: torch.Tensor, max_val_for_offset: float = 1000.0) -> torch.Tensor:
        """Decodes DFL logits to LTRB offsets."""
        reg_max = dfl_project.numel() - 1 # This is R (e.g., 8)
        # reg_logits shape: (B, A, 4 * (R+1)) or (N_pos, 4 * (R+1))
        # dfl_project shape: (R+1)
        
        input_shape = reg_logits.shape
        # Reshape to (..., 4, R+1)
        reg_logits_reshaped = reg_logits.reshape(*input_shape[:-1], 4, reg_max + 1)
        
        # Softmax over the R+1 dimension
        reg_dist = reg_logits_reshaped.softmax(dim=-1)
        
        # Project: (..., 4, R+1) * (R+1) -> sum over R+1 -> (..., 4)
        # Ensure dfl_project is broadcastable: (1, ..., 1, R+1)
        proj_reshaped = dfl_project.reshape((1,) * (reg_dist.dim() - 1) + (-1,))
        ltrb_offsets = (reg_dist * proj_reshaped).sum(dim=-1)
        
        # Clamp offsets to prevent extremely large box coordinates, which can cause NaNs in IoU
        return torch.clamp(ltrb_offsets, min=0, max=max_val_for_offset)

    def decode_predictions(self, reg_logits_flat, anchor_points, strides):
        """Decodes flat regression logits into bounding boxes."""
        ltrb_offsets = self._dfl_to_ltrb(reg_logits_flat, self.dfl_project_buffer)
        ltrb_pixels = ltrb_offsets * strides
        ap_expanded = anchor_points.unsqueeze(0)
        x1y1 = ap_expanded - ltrb_pixels[..., :2]
        x2y2 = ap_expanded + ltrb_pixels[..., 2:]
        return torch.cat([x1y1, x2y2], dim=-1)

    def forward(self, neck_feature_maps, anchor_points=None, strides=None):
        cls_logits, obj_logits, reg_logits = [], [], []
        for i, f_map in enumerate(neck_feature_maps):
            cls_feat = self.cls_conv(f_map)
            reg_feat = self.reg_conv(f_map)
            cls_logits.append(self.cls_pred[i](cls_feat))
            obj_logits.append(self.obj_pred[i](cls_feat))
            reg_logits.append(self.reg_pred[i](reg_feat))

        if self.training:
            return tuple(cls_logits), tuple(obj_logits), tuple(reg_logits)
        else:
            if anchor_points is None or strides is None:
                raise ValueError("anchor_points and strides must be provided for inference.")
            
            cls_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, self.nc) for lvl in cls_logits], 1)
            obj_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 1) for lvl in obj_logits], 1)
            reg_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 4 * (self.reg_max + 1)) for lvl in reg_logits], 1)

            decoded_boxes = self.decode_predictions(reg_flat, anchor_points, strides)
            decoded_scores = (cls_flat + obj_flat).sigmoid()

            return decoded_boxes, decoded_scores

# ───────────────────── Main Model ────────────────
class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80,
                 neck_out_ch: int = 96,
                 img_size: int = 224,
                 head_reg_max: int = 8,
                 head_score_thresh: float = 0.04,
                 head_nms_iou: float = 0.6,
                 head_max_det: int = 100,
                 inplace_act: bool = False):
        super().__init__()
        self.pre = ResizeNorm(size=(img_size, img_size))
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch, inplace_act=inplace_act)
        
        self.strides = (8, 16, 32)
        self.score_th = head_score_thresh
        self.iou_th = head_nms_iou
        self.max_det = head_max_det
        
        self.anchor_manager = AnchorManager(img_size, len(self.strides), self.strides)
        
        self.head = PicoDetHead(
            num_classes=num_classes, 
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=len(self.strides),
            inplace_act=inplace_act
        )

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)

        if self.training:
            cls_logits, obj_logits, reg_logits = self.head((p3, p4, p5))
            return cls_logits, obj_logits, reg_logits
        else:
            # For inference, get anchors and pass them to the head for decoding
            anchor_points, strides_flat = self.anchor_manager()
            return self.head((p3, p4, p5), anchor_points=anchor_points, strides=strides_flat)

# ───────────────────── Utilities ────────────────
class ResizeNorm(nn.Module):
    def __init__(self, size: Tuple[int, int], mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        super().__init__()
        self.size = tuple(size)
        self.register_buffer('m', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x_float_scaled = x.float() / 255.0
        x_resized = F.interpolate(x_float_scaled, self.size, mode='bilinear', align_corners=False, antialias=False)
        return (x_resized - self.m) / self.s

@torch.no_grad()
def _get_dynamic_feat_chs(model: nn.Module, img_size: int, device: torch.device) -> Tuple[int, ...]:
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    model.to(device)
    features = model(dummy_input)
    model.to(original_device)
    if not isinstance(features, (list, tuple)) or len(features) < 3:
        raise ValueError(f"Backbone expected to return at least 3 feature maps, got {len(features) if isinstance(features, (list, tuple)) else 'N/A'}.")
    return tuple(f.shape[1] for f in features)

def get_backbone(arch: str, img_size: int = 224):
    temp_device_for_init = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == "mnv4c":
        print("[INFO] Creating custom MobileNetV4-Small backbone for PicoDet.")
        # These indices are reliable as they come from the custom class definition
        feature_indices = (
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p3_s8'],
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p4_s16'],
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p5_s32']
        )
        net = MobileNetV4ConvSmallPico(
            width_multiplier=1.2, features_only=True, out_features_indices=feature_indices
        )
        # Use your pre-trained classifier backbone weights
        ckpt_path = "mobilenet_w1_2_mnv4c_pretrained_drp0_2_fp32_backbone.pt"
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading pre-trained backbone weights from: {ckpt_path}")
            backbone_sd = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = net.load_state_dict(backbone_sd, strict=False)
            if missing or unexpected: 
                warnings.warn(f"Mismatch loading backbone weights. Missing: {missing}, Unexpected: {unexpected}")
        else:
            warnings.warn(f"Backbone checkpoint not found at {ckpt_path}. Using random weights.")
        
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        print(f"[INFO] Custom MNv4 feature channels: {feat_chs}")
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Only 'mnv4c' is configured in this refactor.")
    net.train()
    return net, feat_chs

from __future__ import annotations
import argparse, random, time, warnings, copy, math
from typing import List, Tuple

import torch, torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.datasets import CocoDetection
from torchvision.tv_tensors import BoundingBoxes
import torchvision.ops as tvops
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F

from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfig, MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

import onnx
from onnx import TensorProto as TP, helper as oh

warnings.filterwarnings('ignore', category=UserWarning)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 256

# --- Data & Transforms ---
# ... (Keep the CocoDetectionV2, build_transforms, collate_v2 functions) ...
CANONICAL_COCO80_IDS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
CANONICAL_COCO80_MAP: dict[int, int] = {coco_id: i for i, coco_id in enumerate(CANONICAL_COCO80_IDS)}

def _coco_to_tvt(annots, lb_map, canvas):
    boxes, labels = [], []
    W, H = canvas
    for a in annots:
        if a.get("iscrowd", 0): continue
        cid = a["category_id"]
        if cid not in lb_map: continue
        x, y, w, h = a["bbox"]
        # A guard to skip annotations with no area.
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(lb_map[cid])
    if not boxes:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    bbx = BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(H, W))
    return {"boxes": bbx, "labels": torch.as_tensor(labels, dtype=torch.int64)}

class CocoDetectionV2(CocoDetection):
    def __init__(self, img_dir, ann_file, lb_map, transforms=None):
        super().__init__(img_dir, ann_file)
        self.lb_map = lb_map
        self._tf = transforms
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        tgt = _coco_to_tvt(anns, self.lb_map, img.size)
        if self._tf is not None: img, tgt = self._tf(img, tgt)
        return img, tgt

def build_transforms(size, train):
    aug = [
        T.ToImage(),
        T.RandomHorizontalFlip(0.5) if train else T.Identity(),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True) if train else T.Resize((size, size), antialias=False),
        # T.RandomPhotometricDistort(p=0.5),
        # T.RandomZoomOut(fill={Image: (123, 117, 104), "boxes": "mean"}, side_range=(1.0, 1.5), p=0.3),
        # T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0, p=0.7),  # removed to keep testing simple
        T.ColorJitter(0.2, 0.2, 0.2, 0.1) if train else T.Identity(),
        T.ToDtype(torch.uint8, scale=True),
    ]
    return T.Compose(aug)

def collate_v2(batch):
    return torch.stack(list(zip(*batch))[0], 0), list(zip(*batch))[1]

import functools
def collate_and_filter(batch: list, min_box_size: int = 1):
    """
    A custom collate_fn that filters out small bounding boxes and then
    removes any images from the batch that no longer have annotations.
    Args:
        batch: A list of (image, target) tuples from the dataset.
        min_box_size: The minimum width and height a box must have to be kept.
                      A value of 1 or 0 effectively disables filtering.
    Returns:
        A tuple of (images_tensor, targets_list) or (None, None) if the
        entire batch is filtered out.
    """
    filtered_imgs = []
    filtered_tgts = []
    for img, tgt in batch:
        boxes = tgt.get("boxes")
        if boxes is None or boxes.numel() == 0:
            filtered_imgs.append(img)
            filtered_tgts.append(tgt)
            continue
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep_mask = (widths >= min_box_size) & (heights >= min_box_size)
        tgt["boxes"] = boxes[keep_mask]
        tgt["labels"] = tgt["labels"][keep_mask]
        if tgt["boxes"].numel() > 0:
            filtered_imgs.append(img)
            filtered_tgts.append(tgt)
    if not filtered_imgs:
        return None, None
    return torch.stack(filtered_imgs, 0), filtered_tgts

# ───────────────────── Task-Aligned Assigner ────────────────
class TaskAlignedAssigner:
    def __init__(self, top_k: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores_sigmoid, pred_boxes_decoded, gt_boxes, gt_labels):
        # pred_scores_sigmoid: (Num_Anchors, Num_Classes), sigmoid applied
        # pred_boxes_decoded: (Num_Anchors, 4), XYXY format
        # gt_boxes: (Num_GT, 4), XYXY format
        # gt_labels: (Num_GT,), long tensor of class indices
        
        num_anchors, num_classes = pred_scores_sigmoid.shape
        num_gt = gt_boxes.shape[0]
        device = pred_scores_sigmoid.device

        if num_gt == 0: # No ground truth boxes
            return (torch.zeros(num_anchors, dtype=torch.bool, device=device), # fg_mask
                    torch.full((num_anchors,), num_classes, dtype=torch.long, device=device), # assigned_labels (bg)
                    torch.zeros((num_anchors,), dtype=torch.float32, device=device), # assigned_scores (iou)
                    torch.full((num_anchors,), -1, dtype=torch.long, device=device)) # assigned_gt_inds

        # 1. Calculate IoUs between predicted boxes and GT boxes
        ious = tvops.box_iou(pred_boxes_decoded, gt_boxes) # Shape: (Num_Anchors, Num_GT)
        # Handle NaN IoUs if boxes are invalid (e.g. zero area) despite prior checks
        ious = torch.nan_to_num(ious, nan=0.0)


        # 2. Calculate alignment metric
        # Gather scores for GT classes: pred_scores_sigmoid[:, gt_labels]
        # gt_labels might have shape (1, Num_GT) repeated to (Num_Anchors, Num_GT)
        # Or use advanced indexing: pred_scores_sigmoid[:, gt_labels] if gt_labels is 1D tensor of indices
        
        # pred_cls_probs_for_gt_classes shape (Num_Anchors, Num_GT)
        # Each column j corresponds to gt_boxes[j] and gt_labels[j]
        # Element (i, j) is the predicted score of anchor i for the class of gt j
        pred_cls_probs_for_gt_classes = pred_scores_sigmoid[:, gt_labels]

        # Alignment metric = (predicted_score_for_gt_class ^ alpha) * (iou_with_gt ^ beta)
        alignment_metric = (pred_cls_probs_for_gt_classes.pow(self.alpha)) * (ious.pow(self.beta))
        
        # Normalize alignment_metric per GT (commented out in some impls, TOOD paper mentions it for candidate selection)
        # max_per_gt = alignment_metric.max(dim=0, keepdim=True).values.clamp_(min=self.eps)
        # alignment_metric = alignment_metric / max_per_gt


        # 3. Select top-k candidates for each GT based on alignment metric
        # topk along anchors (dim=0) for each GT
        # Remove the clamp that was suppressing valid alignment scores
        dynamic_top_k = min(self.top_k, num_anchors)
        _, top_k_indices_per_gt = torch.topk(alignment_metric, dynamic_top_k, dim=0)


        # 4. Assign GTs to anchors
        # fg_mask: boolean mask for anchors considered foreground (assigned to a GT)
        # assigned_gt_inds: index of the GT assigned to each anchor, or -1 if background
        
        # Initialize: all anchors are background
        assigned_gt_inds = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        
        # Create a mask for candidate anchors (those in top-k for any GT)
        candidate_mask = torch.zeros_like(alignment_metric, dtype=torch.bool) # (Num_Anchors, Num_GT)
        candidate_mask.scatter_(0, top_k_indices_per_gt, True) # Mark top-k anchors for each GT

        # Resolve conflicts: if an anchor is top-k for multiple GTs, assign it to the GT
        # with which it has the highest IoU (or alignment_metric).
        # alignment_metric_masked: only consider candidates, set others to -1
        alignment_metric_masked = torch.where(candidate_mask, alignment_metric, torch.tensor(-1., device=device))
        
        # anchor_max_alignment_score shape (Num_Anchors,), anchor_best_gt_idx shape (Num_Anchors,)
        anchor_max_alignment_score, anchor_best_gt_idx = alignment_metric_masked.max(dim=1)

        # An anchor is foreground if its max_alignment_score (with its best GT) is >= 0
        # (meaning it was a candidate for at least one GT)
        fg_mask = anchor_max_alignment_score >= 0
        
        # For an anchor to be positive, it must also be a candidate for the GT it was assigned to.
        # This prevents a case where an anchor's highest alignment score is for a GT
        # for which it was not in the top-k.
        if fg_mask.any():
            # Get the GT indices for the foreground anchors
            best_gt_indices_for_fg = anchor_best_gt_idx[fg_mask]
            
            # Get the indices of the foreground anchors themselves
            fg_indices = torch.where(fg_mask)[0]
            
            # Check if the foreground anchors were actual candidates for their assigned GT
            is_candidate = candidate_mask[fg_indices, best_gt_indices_for_fg]
            
            # Update fg_mask: only keep anchors that were valid candidates
            # Create a new tensor for the updated mask to avoid in-place modification issues
            updated_fg_mask = fg_mask.clone()
            updated_fg_mask[fg_indices] = is_candidate
            fg_mask = updated_fg_mask

        # Assign the best GT index to foreground anchors
        assigned_gt_inds[fg_mask] = anchor_best_gt_idx[fg_mask]
        
        # 5. Prepare outputs for loss calculation
        # assigned_labels: class label for each anchor (background_class_idx if background)
        # assigned_scores: IoU score for VFL for each positive anchor
        assigned_labels = torch.full((num_anchors,), num_classes, dtype=torch.long, device=device) # Init with background
        assigned_scores = torch.zeros((num_anchors,), dtype=torch.float32, device=device) # Init with 0 score

        if fg_mask.any():
            pos_anchor_indices = torch.where(fg_mask)[0]
            pos_assigned_gt_indices = assigned_gt_inds[pos_anchor_indices]
            
            assigned_labels[pos_anchor_indices] = gt_labels[pos_assigned_gt_indices]
            # VFL uses IoU as the target score for positive samples
            assigned_scores[pos_anchor_indices] = ious[pos_anchor_indices, pos_assigned_gt_indices]

        return fg_mask, assigned_labels, assigned_scores, assigned_gt_inds

# ─── utils/ema.py ──────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device=None):
        self.ema = copy.deepcopy(model).eval() # Creates a new model instance for EMA
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device # Store device for potential future use if needed
        if device is not None:
            self.ema.to(device, non_blocking=True)
        
        # For QAT models, buffers like observer min/max might be initialized empty.
        # It's good to run a forward pass on `model` if it's a QAT model
        # to ensure observers are initialized before the first EMA update.
        # However, this is usually handled by the training loop itself.

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if self.device is None and next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device # Infer device if not set
        
        # Ensure EMA model is on the same device as the live model
        if self.device is not None:  # and self.ema.device != self.device:
            self.ema.to(self.device)

        # Update Parameters
        for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
            if model_param.device != ema_param.device: # Ensure params are on same device
                 model_param_detached = model_param.detach().to(ema_param.device)
            else:
                 model_param_detached = model_param.detach()
            
            if ema_param.shape != model_param_detached.shape:
                warnings.warn(f"EMA: Parameter shape mismatch. EMA: {ema_param.shape}, Model: {model_param_detached.shape}. Skipping EMA for this param.")
                continue
            ema_param.mul_(self.decay).add_(model_param_detached, alpha=1.0 - self.decay)

        # Update Buffers (handles observer states in QAT)
        for (ema_name, ema_buffer), (model_name, model_buffer) in zip(self.ema.named_buffers(), model.named_buffers()):
            if ema_name != model_name: # Should not happen if deepcopy worked
                warnings.warn(f"EMA: Buffer name mismatch. EMA: {ema_name}, Model: {model_name}. Potential error in EMA state.")
                continue

            if model_buffer.device != ema_buffer.device:
                model_buffer_detached = model_buffer.detach().to(ema_buffer.device)
            else:
                model_buffer_detached = model_buffer.detach()

            if ema_buffer.shape != model_buffer_detached.shape:
                # This handles cases like observer min/max starting empty and becoming populated.
                # We need to replace the buffer in self.ema, not just copy into a detached tensor.
                # Find the module path and attribute name for the buffer in self.ema
                module_path_parts = ema_name.split('.')
                attr_name = module_path_parts.pop()
                current_module = self.ema
                try:
                    for part in module_path_parts:
                        current_module = getattr(current_module, part)
                    
                    # Delete old buffer and register the new one (cloned from model_buffer)
                    delattr(current_module, attr_name) 
                    current_module.register_buffer(attr_name, model_buffer_detached.clone())
                    # print(f"DEBUG EMA: Replaced buffer {ema_name} due to shape change. New shape: {model_buffer_detached.shape}")
                except AttributeError:
                    warnings.warn(f"EMA: Could not replace buffer {ema_name} in EMA model structure despite shape mismatch.")
                continue # Skip decay logic for this buffer; it's now synced or warning issued.

            # If shapes match, apply update (copy for non-float, EMA for float)
            if not ema_buffer.dtype.is_floating_point:
                ema_buffer.copy_(model_buffer_detached)
            else: # If it's a floating point buffer that needs EMA (rare, but possible)
                ema_buffer.mul_(self.decay).add_(model_buffer_detached, alpha=1.0 - self.decay)
                
    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.ema.state_dict(), strict=True) # Use strict=True for safety


# ───────────────────── Updated Training Loop ────────────────────────
def train_epoch(
        model, loader, opt, scaler, assigner: TaskAlignedAssigner, device: torch.device,
        epoch: int, max_epochs: int, debug_prints: bool = True, anchor_manager=None,
        is_qat: bool = False
):
    model.train()
    total_loss, total_fg_count = 0.0, 0
    # Use the existing VarifocalLoss from your library
    vfl_loss_fn = VarifocalLoss(reduction='sum')
    if hasattr(model, 'anchor_manager'):
        anchor_points, anchor_strides_flat = model.anchor_manager()
    elif anchor_manager is not None:
        anchor_points, anchor_strides_flat = anchor_manager()
    else:
        raise ValueError("Model doesn't have anchor_manager and none provided")
    for i, (imgs, tgts_batch) in enumerate(loader):
        # Handle the case where the entire batch was filtered out
        if imgs is None or not hasattr(imgs, 'numel') or not imgs.numel():
            if debug_prints:
                print(f"Skipping batch {i} because it became empty after filtering.")
            continue
        imgs = imgs.to(device)
        
        cls_preds_levels, obj_preds_levels, reg_preds_levels = model(imgs)
        
        bs, nc = imgs.size(0), model.head.nc
        reg_max = model.head.reg_max
        cls_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, nc) for lvl in cls_preds_levels], 1)
        obj_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 1) for lvl in obj_preds_levels], 1)
        reg_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 4 * (reg_max + 1)) for lvl in reg_preds_levels], 1)

        # Decode boxes (detached) to use in the assigner
        # During QAT, the model might already output sigmoid-ed scores from a quantized op.
        # However, the assigner requires fresh sigmoid on the logits sum.
        # Let's ensure we use the logits for consistency.
        pred_scores_for_assigner = (cls_p_flat.detach() + obj_p_flat.detach()).sigmoid()
        decoded_boxes = model.head.decode_predictions(reg_p_flat.detach(), anchor_points, anchor_strides_flat)
        
        # We need the raw logits for the loss function, but the assigner works best with probabilities
        pred_scores_for_assigner = (cls_p_flat.detach() + obj_p_flat.detach()).sigmoid()
        
        # Debug: Check if scores are reasonable during QAT
        if debug_prints and i == 0 and epoch == 0:
            print(f"[DEBUG] Raw cls logits stats: min={cls_p_flat.min().item():.6f}, "
                  f"max={cls_p_flat.max().item():.6f}, "
                  f"mean={cls_p_flat.mean().item():.6f}")
            print(f"[DEBUG] Raw obj logits stats: min={obj_p_flat.min().item():.6f}, "
                  f"max={obj_p_flat.max().item():.6f}, "
                  f"mean={obj_p_flat.mean().item():.6f}")
            print(f"[DEBUG] Combined sigmoid scores stats: min={pred_scores_for_assigner.min().item():.6f}, "
                  f"max={pred_scores_for_assigner.max().item():.6f}, "
                  f"mean={pred_scores_for_assigner.mean().item():.6f}")

        all_fg_masks, all_assigned_labels, all_assigned_scores, all_assigned_boxes = [], [], [], []
        
        for b_idx in range(bs):
            gt_boxes = tgts_batch[b_idx]["boxes"].to(device)
            gt_labels = tgts_batch[b_idx]["labels"].to(device)
            
            fg_mask, assigned_labels, assigned_scores, assigned_gt_inds = assigner(
                pred_scores_for_assigner[b_idx], decoded_boxes[b_idx], gt_boxes, gt_labels
            )
            
            all_fg_masks.append(fg_mask)
            all_assigned_labels.append(assigned_labels)
            all_assigned_scores.append(assigned_scores)
            
            # Retrieve the GT boxes for the positive anchors
            assigned_gt_boxes = torch.zeros_like(decoded_boxes[b_idx])
            if fg_mask.any():
                pos_indices = torch.where(fg_mask)[0]
                pos_gt_indices = assigned_gt_inds[pos_indices]
                assigned_gt_boxes[pos_indices] = gt_boxes[pos_gt_indices]
            all_assigned_boxes.append(assigned_gt_boxes)

        fg_mask_batch = torch.stack(all_fg_masks)
        num_fg = fg_mask_batch.sum()
        
        if num_fg == 0:
            if debug_prints and i % 50 == 0: print(f"Epoch {epoch} B {i}: No foreground targets found.")
            continue
            
        total_fg_count += num_fg.item()

        # --- Varifocal Loss (Classification) ---
        assigned_labels_batch = torch.stack(all_assigned_labels)          # [B, A]
        assigned_scores_batch = torch.stack(all_assigned_scores)          # [B, A]
        assigned_scores_batch = torch.nan_to_num(assigned_scores_batch, nan=0.0)
        
        joint_logits = cls_p_flat + obj_p_flat
        vfl_targets  = torch.zeros_like(joint_logits)
        
        # Only foreground anchors contribute to the VFL target
        pos_mask = fg_mask_batch
        
        if pos_mask.any():
            pos_labels  = assigned_labels_batch[pos_mask]                 # 1-D
            pos_scores  = assigned_scores_batch[pos_mask]                 # 1-D
        
            # one-hot for positives only (safe – all < nc)
            pos_onehot = F.one_hot(pos_labels, num_classes=nc).float()    # [N_pos, nc]
            vfl_targets[pos_mask] = pos_onehot * pos_scores.unsqueeze(-1)
        
        loss_vfl = vfl_loss_fn(joint_logits, vfl_targets) / num_fg

        # --- DFL + CIoU (Regression) ---
        reg_preds_fg = reg_p_flat[fg_mask_batch]
        assigned_boxes_fg = torch.stack(all_assigned_boxes)[fg_mask_batch]
        anchor_points_fg = anchor_points.unsqueeze(0).repeat(bs, 1, 1)[fg_mask_batch]
        anchor_strides_fg = anchor_strides_flat.unsqueeze(0).repeat(bs, 1, 1)[fg_mask_batch]
        
        ltrb_targets = torch.cat([
            anchor_points_fg - assigned_boxes_fg[:, :2],
            assigned_boxes_fg[:, 2:] - anchor_points_fg
        ], 1) / anchor_strides_fg
        # Ensure ltrb_targets are non-negative for build_dfl_targets
        ltrb_targets.clamp_(min=0)
        
        dfl_target_dist = build_dfl_targets(ltrb_targets, reg_max)
        loss_dfl = dfl_loss(reg_preds_fg, dfl_target_dist)

        pred_ltrb_offsets = model.head._dfl_to_ltrb(reg_preds_fg, model.head.dfl_project_buffer)
        pred_boxes_fg = torch.cat([
            anchor_points_fg - pred_ltrb_offsets[:,:2] * anchor_strides_fg,
            anchor_points_fg + pred_ltrb_offsets[:,2:] * anchor_strides_fg
        ], 1)
        eps = 1e-9
        keep = (((pred_boxes_fg[:, 2:] - pred_boxes_fg[:, :2]).prod(1) > 0) &
                ((assigned_boxes_fg[:, 2:] - assigned_boxes_fg[:, :2]).prod(1) > 0))
        loss_iou = 0.0
        if keep.any():
            loss_iou = (tvops.complete_box_iou_loss(
                pred_boxes_fg[keep],
                assigned_boxes_fg[keep],
                reduction='sum') / (keep.sum() + eps))

        if is_qat:
            # Use more conservative weights for QAT to aid stability
            w_iou = 1.0
            w_vfl = 0.5
            w_dfl = 0.1
        else:
            w_iou = 2.0
            w_vfl = 1.0 
            w_dfl = 0.25

        loss = w_vfl * loss_vfl + w_dfl * loss_dfl + w_iou * loss_iou
        if not torch.isfinite(loss):
            print(f"[WARN] non-finite loss ({loss.item()}) – using fallback")
            loss = loss.new_full((), 6.0, requires_grad=True)  # constant scalar
        
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()

        if debug_prints and i > 0 and i % 500 == 0:
            liou = loss_iou if isinstance(loss_iou, float) else loss_iou.item()
            print(
                f"E{epoch} B{i}/{len(loader)} | Loss: {loss.item():.4f} "
                f"(VFL:{loss_vfl.item():.4f}, DFL:{loss_dfl.item():.4f}, "
                f"CIoU:{liou:.4f}) | Num_FG: {num_fg.item()}"
            )

    return total_loss / (i + 1) if i > 0 else total_loss

# ───────────────────── QAT Wrapper ────────────────
class QATModelWrapper(nn.Module):
    """Wrapper to maintain model interface after QAT preparation"""
    def __init__(self, original_model: PicoDet, qat_prepared_model: torch.fx.GraphModule):
        super().__init__()
        self.qat_model = qat_prepared_model
        
        # Copy essential attributes from original model
        self.anchor_manager = original_model.anchor_manager
        self.score_th = original_model.score_th
        self.iou_th = original_model.iou_th
        self.max_det = original_model.max_det
        self.head = original_model.head
        self.strides = original_model.strides
    
    def forward(self, x):
        # During training, the QAT model should behave like the original model
        # The QAT prepared model should output the same format as the original
        if self.training:
            # In training mode, just pass through the QAT model
            return self.qat_model(x)
        else:
            # In eval mode, we need to handle the inference format
            with torch.no_grad():
                anchor_points, strides_flat = self.anchor_manager()
                # The qat_model should output cls_logits, obj_logits, reg_logits
                cls_logits, obj_logits, reg_logits = self.qat_model(x)
                cls_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, self.head.nc) for lvl in cls_logits], 1)
                obj_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 1) for lvl in obj_logits], 1)
                reg_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 4 * (self.head.reg_max + 1)) for lvl in reg_logits], 1)
                decoded_boxes = self.head.decode_predictions(reg_flat, anchor_points, strides_flat)
                decoded_scores = (cls_flat + obj_flat).sigmoid()
                return decoded_boxes, decoded_scores
    
    def train(self, mode: bool = True):
        self.training = mode
        self.qat_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.training = False
        self.qat_model.eval()
        return super().eval()


# --- Validation, QAT, and ONNX Export ---
@torch.no_grad()
def quick_val_iou(model, loader, device, epoch_num: int = -1, debug_prints: bool = True):
    model.eval()
    total_iou_sum, num_gt_total = 0.0, 0
    total_preds_after_nms = 0
    num_images_processed = 0
    
    # Correctly determine number of images in the validation set
    try:
        num_total_images = len(loader.dataset)
    except TypeError:
        num_total_images = "N/A"

    # For QAT models, we need to get attributes from the original model
    score_th = model.score_th
    iou_th = model.iou_th
    max_det = model.max_det

    for imgs_batch, tgts_batch in loader:
        imgs_batch = imgs_batch.to(device)
        num_images_processed += imgs_batch.size(0)
        
        # This now works for both regular and QAT-wrapped models
        # because model.eval() sets the QAT wrapper to inference mode.
        raw_pred_boxes, raw_pred_scores = model(imgs_batch)

        for b_idx in range(imgs_batch.size(0)):
            gt_boxes = tgts_batch[b_idx]["boxes"].to(device)
            if gt_boxes.numel() == 0:
                continue
            
            num_gt_total += gt_boxes.size(0)
            
            pred_boxes = raw_pred_boxes[b_idx]
            pred_scores = raw_pred_scores[b_idx]

            conf_per_anchor, labels_per_anchor = torch.max(pred_scores, dim=1)
            keep_by_score = conf_per_anchor >= score_th
            
            boxes_filt = pred_boxes[keep_by_score]
            scores_filt = conf_per_anchor[keep_by_score]
            
            if boxes_filt.numel() == 0:
                continue

            keep_by_nms = tvops.nms(boxes_filt, scores_filt, iou_threshold=iou_th)
            
            if max_det > 0 and len(keep_by_nms) > max_det:
                keep_by_nms = keep_by_nms[:max_det]
            
            final_boxes = boxes_filt[keep_by_nms]
            total_preds_after_nms += final_boxes.size(0)

            if final_boxes.numel() > 0:
                ious = tvops.box_iou(final_boxes, gt_boxes)
                max_iou_per_gt, _ = torch.max(ious, dim=0)
                total_iou_sum += max_iou_per_gt.sum().item()

    avg_preds_after = total_preds_after_nms / num_images_processed if num_images_processed > 0 else 0
    final_mean_iou = total_iou_sum / num_gt_total if num_gt_total > 0 else 0
    
    if debug_prints:
        print(f"--- Val E{epoch_num} ---")
        print(f"Images: {num_images_processed}/{num_total_images}, Total GTs: {num_gt_total}")
        print(f"Avg preds/img (after NMS): {avg_preds_after:.2f}")
        print(f"Validation Mean IoU: {final_mean_iou:.4f}")
    
    return final_mean_iou

pack = "qnnpack"

def qat_prepare_old(model: nn.Module, example_input: torch.Tensor) -> torch.fx.GraphModule:
    # For QAT, inplace activations can sometimes be problematic.
    if True:
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
                if hasattr(module, 'inplace') and module.inplace:
                    module.inplace = False

    qconfig = QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                      weight=MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    qmap = get_default_qat_qconfig_mapping(pack).set_global(qconfig).set_module_name('pre', None)
    return prepare_qat_fx(model.cpu().train(), qmap, example_input.cpu())


def qat_prepare(model: nn.Module, example_input: torch.Tensor) -> torch.fx.GraphModule:
    """
    Prepares the model for Quantization-Aware Training (QAT) using FX graph mode.

    Args:
        model: The PyTorch nn.Module to prepare for QAT.
        example_input: A representative example input tensor for tracing the model.

    Returns:
        A torch.fx.GraphModule prepared for QAT.
    """
    # 1. Define the desired QConfig for most layers
    #    Activations: per-tensor, affine, quint8
    #    Weights: per-channel, symmetric, qint8 (common for conv/linear layers)
    global_qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
    )

    # 2. Get the default QAT qconfig mapping for the chosen backend
    #    'x86' or 'qnnpack' (for ARM) are common backends.
    backend_config = pack
    qconfig_mapping = get_default_qat_qconfig_mapping(backend_config)

    # 3. Apply the global QConfig to the mapping
    #    This will apply `global_qconfig` to all quantizable module types
    #    unless overridden by more specific settings (e.g., per module type/instance).
    qconfig_mapping = qconfig_mapping.set_global(global_qconfig)

    # 4. Specify modules to skip quantization
    #    The 'pre' module (ResizeNorm) handles uint8 input and normalization;
    qconfig_mapping = qconfig_mapping.set_module_name('pre', None)

    # 5. Prepare the model for QAT using FX graph mode
    model.cpu().train() # Ensure model is on CPU and in train mode
    
    prepared_model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_input # Must also be on CPU
    )
    
    return prepared_model

class ONNXExportableModel(nn.Module):
    def __init__(self, qat_model: PicoDet):
        super().__init__()
        self.quantized_model = convert_fx(copy.deepcopy(qat_model).cpu().eval())
    def forward(self, x: torch.Tensor):
        return self.quantized_model(x)

def append_nms_to_onnx(
        in_path: str,
        out_path: str,
        score_thresh: float,
        iou_thresh: float,
        max_det: int,
        *,
        raw_boxes: str = "raw_boxes",     # [B , A , 4]
        raw_scores: str = "raw_scores",   # [B , A , C]
        top_k_before_nms: bool = False,
        k_value: int = 800,
):
    m = onnx.load(in_path)
    g = m.graph

    # ───────── constants ─────────
    g.initializer.extend([
        oh.make_tensor("nms_iou_th",   TP.FLOAT, [], [iou_thresh]),
        oh.make_tensor("nms_score_th", TP.FLOAT, [], [score_thresh]),
        oh.make_tensor("nms_max_det",  TP.INT64, [], [max_det]),
        oh.make_tensor("nms_axis0", TP.INT64, [1], [0]),
        oh.make_tensor("nms_axis1", TP.INT64, [1], [1]),
        oh.make_tensor("nms_axis2", TP.INT64, [1], [2]),
        oh.make_tensor("nms_shape_boxes3d",  TP.INT64, [3], [0, -1, 4]),
        oh.make_tensor("nms_shape_scores3d", TP.INT64, [3], [0, 0, -1]),
    ])
    if top_k_before_nms:
        g.initializer.extend([
            oh.make_tensor("nms_k_topk", TP.INT64, [1], [k_value]),
        ])

    # ───────── reshape / transpose raw outputs ─────────
    boxes3d = "nms_boxes3d"
    g.node.append(oh.make_node(
        "Reshape", [raw_boxes, "nms_shape_boxes3d"], [boxes3d],
        name="nms_Reshape_Boxes3D"))

    scores3d = "nms_scores3d"
    g.node.append(oh.make_node(
        "Reshape", [raw_scores, "nms_shape_scores3d"], [scores3d],
        name="nms_Reshape_Scores3D"))

    scores_bca = "nms_scores_bca"
    g.node.append(oh.make_node(
        "Transpose", [scores3d], [scores_bca],
        perm=[0, 2, 1], name="nms_Transpose_BCA"))  # [B , C , A]

    # ───────── optional Top-K filter ─────────
    if top_k_before_nms:
        max_conf = "nms_max_conf"
        g.node.append(oh.make_node(
            "ReduceMax", [scores_bca, "nms_axis1"], [max_conf],
            keepdims=0, name="nms_ReduceMax"))

        topk_vals = "nms_topk_vals"
        topk_idx  = "nms_topk_idx"                 # [B , K]
        g.node.append(oh.make_node(
            "TopK",
            [max_conf, "nms_k_topk"],
            [topk_vals, topk_idx],
            axis=1, largest=1, sorted=0,
            name="nms_TopK"))

        topk_idx_unsq = "nms_topk_idx_unsq"        # [B , K , 1]
        g.node.append(oh.make_node(
            "Unsqueeze", [topk_idx, "nms_axis2"], [topk_idx_unsq],
            name="nms_UnsqTopKIdx"))

        boxes_topk = "nms_boxes_topk"              # [B , K , 4]
        g.node.append(oh.make_node(
            "GatherND", [boxes3d, topk_idx_unsq], [boxes_topk],
            batch_dims=1, name="nms_GatherBoxesTopK"))

        scores_bac = "nms_scores_bac"
        g.node.append(oh.make_node(
            "Transpose", [scores_bca], [scores_bac],
            perm=[0, 2, 1], name="nms_Transpose_BAC"))

        scores_bkc = "nms_scores_bkc"              # [B , K , C]
        g.node.append(oh.make_node(
            "GatherND", [scores_bac, topk_idx_unsq], [scores_bkc],
            batch_dims=1, name="nms_GatherScoresTopK"))

        scores_bck = "nms_scores_bck"              # [B , C , K]
        g.node.append(oh.make_node(
            "Transpose", [scores_bkc], [scores_bck],
            perm=[0, 2, 1], name="nms_Transpose_BCK"))

        nms_boxes  = boxes_topk
        nms_scores = scores_bck
    else:
        nms_boxes  = boxes3d
        nms_scores = scores_bca

    # ───────── Non-Max Suppression ─────────
    sel = "nms_selected"                          # [N , 3]
    g.node.append(oh.make_node(
        "NonMaxSuppression",
        [nms_boxes, nms_scores,
         "nms_max_det", "nms_iou_th", "nms_score_th"],
        [sel], name="nms_NMS"))

    # split indices (batch , class , anchor)
    g.initializer.extend([
        oh.make_tensor("nms_split111", TP.INT64, [3], [1, 1, 1]),
    ])
    b_col, c_col, a_col = "nms_b", "nms_c", "nms_a"
    g.node.append(oh.make_node(
        "Split", [sel, "nms_split111"], [b_col, c_col, a_col],
        axis=1, name="nms_SplitSel"))

    # squeeze to 1-D
    b_idx, cls_idx, anc_idx = "batch_idx", "class_idx", "anchor_idx"
    for src, dst in [(b_col, b_idx), (c_col, cls_idx), (a_col, anc_idx)]:
        g.node.append(oh.make_node(
            "Squeeze", [src, "nms_axis1"], [dst],
            name=f"nms_Squeeze_{dst}"))

    # ───── gather det_boxes  (batch_dims=1, indices=[anchor]) ─────
    a_unsq = "nms_a_unsq"
    g.node.append(oh.make_node(
        "Unsqueeze", [anc_idx, "nms_axis1"], [a_unsq],
        name="nms_UnsqAnchor"))

    det_boxes = "det_boxes"
    g.node.append(oh.make_node(
        "GatherND", [nms_boxes, a_unsq], [det_boxes],
        batch_dims=1, name="nms_GatherDetBoxes"))

    # ───── gather det_scores  (batch_dims=1, indices=[class,anchor]) ─────
    cls_unsq = "nms_cls_unsq"
    g.node.append(oh.make_node(
        "Unsqueeze", [cls_idx, "nms_axis1"], [cls_unsq],
        name="nms_UnsqClass"))

    idx_scores = "nms_idx_scores"                 # [N , 2]
    g.node.append(oh.make_node(
        "Concat", [cls_unsq, a_unsq], [idx_scores],
        axis=1, name="nms_CatClassAnchor"))

    det_scores = "det_scores"
    g.node.append(oh.make_node(
        "GatherND", [nms_scores, idx_scores], [det_scores],
        batch_dims=1, name="nms_GatherDetScores"))

    # ───────── declare final outputs ─────────
    del g.output[:]   # remove existing outputs
    g.output.extend([
        oh.make_tensor_value_info(det_boxes,  TP.FLOAT, ['N', 4]),
        oh.make_tensor_value_info(det_scores, TP.FLOAT, ['N']),
        oh.make_tensor_value_info(cls_idx,    TP.INT64, ['N']),
        oh.make_tensor_value_info(b_idx,      TP.INT64, ['N']),
    ])

    onnx.checker.check_model(m)
    # onnx.utils.polish_model()
    onnx.save(m, out_path)
    print(f"[SAVE] Final ONNX with NMS → {out_path}")


# ───────────────────────── Main Execution ────────────────────────────────
def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', default='mnv4c', choices=['mnv4c'])
    pa.add_argument('--epochs', type=int, default=2) 
    pa.add_argument('--qat_epochs', type=int, default=1) 
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8_refactored.onnx')
    pa.add_argument('--min_box_size', type=int, default=8, help="Minimum pixel width/height for a GT box to be used in training.")
    cfg = pa.parse_args(argv)
    BACKBONE_FREEZE_EPOCHS = 1

    dev = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[INFO] Device: {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, img_size=IMG_SIZE)
    model = PicoDet(backbone, feat_chs, num_classes=80, neck_out_ch=96, img_size=IMG_SIZE).to(dev)
    model.score_th = 0.02  # inital threshold

    if cfg.min_box_size is not None and cfg.min_box_size > 1:
        print(f"[INFO] Filtering enabled: Dropping GT boxes smaller than {cfg.min_box_size}x{cfg.min_box_size} pixels.")
        custom_collate_fn = functools.partial(collate_and_filter, min_box_size=cfg.min_box_size)
    else:
        print("[INFO] GT box filtering disabled.")
        custom_collate_fn = collate_v2

    train_ds = CocoDetectionV2(f"{cfg.coco_root}/train2017", f"{cfg.coco_root}/annotations/instances_train2017.json", CANONICAL_COCO80_MAP, build_transforms(IMG_SIZE, True))
    val_ds = CocoDetectionV2(f"{cfg.coco_root}/val2017", f"{cfg.coco_root}/annotations/instances_val2017.json", CANONICAL_COCO80_MAP, build_transforms(IMG_SIZE, False))
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers, collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=bool(cfg.workers))
    vl_loader = DataLoader(val_ds, batch_size=cfg.batch*2, num_workers=cfg.workers, collate_fn=custom_collate_fn, pin_memory=True)

    lr = 0.005
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    warmup_iters = len(tr_loader) * 2
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - 1, eta_min=lr * 0.1) # Anneal over remaining epochs
    sch = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_iters])
    scaler = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))
    
    assigner = TaskAlignedAssigner(top_k=13, alpha=1.0, beta=6.0, eps=1e-9)
    ema = ModelEMA(model, decay=0.999, device=dev)
    
    print(f"[INFO] Starting FP32 training for {cfg.epochs} epochs...")
    debug_prints = True
    for ep in range(cfg.epochs):
        print(f"[INFO] Backbone {'frozen' if ep < BACKBONE_FREEZE_EPOCHS else 'unfrozen'} for epoch {ep+1}/{cfg.epochs}...")
        for name, param in model.backbone.named_parameters():
            param.requires_grad_(ep >= BACKBONE_FREEZE_EPOCHS)

        train_loss = train_epoch(model, tr_loader, opt, scaler, assigner, dev, ep, cfg.epochs, debug_prints=debug_prints, anchor_manager=model.anchor_manager)
        ema.update(model)
        
        # It's better to validate with the EMA model
        ema_model_for_val = copy.deepcopy(model)
        ema.copy_to(ema_model_for_val)
        val_iou = quick_val_iou(ema_model_for_val, vl_loader, dev, epoch_num=ep)
        
        print(f"Epoch {ep+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {opt.param_groups[0]['lr']:.6f}\n")
        sch.step()

    print("\n[INFO] Preparing model for QAT...")
    
    # First, copy the EMA weights to the main model for QAT
    ema_model_for_qat = copy.deepcopy(model)
    ema.copy_to(ema_model_for_qat)
    ema_model_for_qat.train()

    # Disable inplace activations before QAT preparation
    for module in ema_model_for_qat.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    # Create QAT model with proper example input
    example = torch.randint(
        low=0, high=256,
        size=(1, 3, IMG_SIZE, IMG_SIZE),
        dtype=torch.uint8
    )
    qat_prepared_graph = qat_prepare(ema_model_for_qat, example)
    
    # Wrap the prepared graph to handle training/eval mode switching
    qat_model = QATModelWrapper(ema_model_for_qat, qat_prepared_graph)
    qat_model = qat_model.to(dev)
    
    # Prime the QAT observers by running several forward passes in training mode
    print("[INFO] Priming QAT observers...")
    qat_model.train()
    prime_count = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tr_loader):
            if i >= 100: break  # Increased from 50 to 100 for better calibration
            try:
                qat_model(imgs.to(dev))
                if (i + 1) % 10 == 0: 
                    print(f"[INFO] Primed {i+1} batches...")
            except Exception as e:
                print(f"[WARNING] Error during priming batch {i}: {e}")
                continue
    print(f"[INFO] Primed QAT observers with {i+1} batches")

    # Use AdamW with a moderate LR for QAT finetuning
    # QAT typically needs higher LR than 1e-5 to recover from quantization noise
    opt_q = torch.optim.AdamW(qat_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(opt_q, T_max=cfg.qat_epochs, eta_min=1e-6)
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))

    print(f"[INFO] Starting QAT finetuning for {cfg.qat_epochs} epochs...")
    for qep in range(cfg.qat_epochs):
        qat_model.train()
        train_loss_q = train_epoch(qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, cfg.qat_epochs, debug_prints=debug_prints, is_qat=True)
        qat_model.eval()
        val_iou_q = quick_val_iou(qat_model, vl_loader, dev, epoch_num=qep, debug_prints=debug_prints)
        print(f"QAT Epoch {qep+1}/{cfg.qat_epochs} | Train Loss: {train_loss_q:.4f} | Val IoU: {val_iou_q:.4f}\n")
        scheduler_q.step()

    model.score_th = 0.05  # export threshold

    print("\n[INFO] Exporting to ONNX...")
    temp_onnx_path = cfg.out.replace(".onnx", "_temp.onnx")

    print("[INFO] Finishing QAT – converting to int8")
    # We need to convert the inner qat_model (the fx.GraphModule)
    qat_model.cpu().eval()
    # Use the EMA model for conversion for better accuracy
    # ema.copy_to(qat_model) # EMA is tricky with QAT models, skip for now
    
    # The model to be converted is the inner fx.GraphModule
    int8_model_graph = convert_fx(copy.deepcopy(qat_model.qat_model))



    # For export, we need a simple wrapper that calls the converted graph
    # and then applies the post-processing (decoding) logic.
    # The QATModelWrapper is not suitable as its forward has Python-control flow.
    class ONNXExportWrapper(nn.Module):
        def __init__(self, int8_core_model, head_ref, anchor_manager_ref):
            super().__init__()
            self.core_model = int8_core_model
            self.head = head_ref
            self.anchor_manager = anchor_manager_ref
        
        def forward(self, x):
            cls_logits, obj_logits, reg_logits = self.core_model(x)
            
            anchor_points, strides_flat = self.anchor_manager()
            
            cls_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, self.head.nc) for lvl in cls_logits], 1)
            obj_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 1) for lvl in obj_logits], 1)
            reg_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 4 * (self.head.reg_max + 1)) for lvl in reg_logits], 1)

            decoded_boxes = self.head.decode_predictions(reg_flat, anchor_points, strides_flat)
            decoded_scores = (cls_flat + obj_flat).sigmoid()
            return decoded_boxes, decoded_scores

    # The head from the original model is needed for decoding parameters
    export_model = ONNXExportWrapper(int8_model_graph, model.head, model.anchor_manager)
    export_model.eval()

    print("[INFO] ONNX export")
    # The model now expects uint8 input because the 'pre' layer is inside
    dummy = torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    torch.onnx.export(
            export_model,
            dummy,
            temp_onnx_path,
            input_names=['images'],
            output_names=['raw_boxes', 'raw_scores'],
            dynamic_axes={'images': {0: 'batch'},
                          'raw_boxes': {0: 'batch'},
                          'raw_scores': {0: 'batch'}},
            opset_version=18)
    append_nms_to_onnx(temp_onnx_path, cfg.out,
                       score_thresh=model.score_th,
                       iou_thresh=model.iou_th,
                       max_det=model.max_det)
    print(f"[SAVE] Intermediate ONNX (INT8 core, no NMS) -> {temp_onnx_path}")
    print(f"[SAVE] Final ONNX with NMS -> {cfg.out}")


if __name__ == '__main__':
    main()