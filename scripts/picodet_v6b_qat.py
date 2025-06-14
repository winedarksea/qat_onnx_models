# PyTorch 2.7 / opset 18
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

if False:
    try:
        # Ensure this path is correct for your setup
        folder_to_add = r"/Users/colincatlin/Documents-NoCloud/qat_onnx_models/scripts"
        sys.path.append(folder_to_add)
        from customMobilenetNetv4 import MobileNetV4ConvSmallPico
    except ImportError:
        print("Warning: customMobilenetNetv4.py not found. 'mnv4_custom' backbone will not be available.")
        MobileNetV4ConvSmallPico = None

# ───────────────────────────── layers (Unchanged) ──────────────────────────────
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

# ───────────────────────────── neck (Unchanged) ────────────────────────────────
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

# ─────────────────────── losses (Unchanged) ──────────────────
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
    x = offsets.clamp_(0, reg_max - 1e-6) # Clamp to avoid index out of bounds
    l = x.floor().long()
    r = l + 1
    w_r = x - l.float()
    w_l = 1. - w_r
    one_hot_l = F.one_hot(l, reg_max + 1).float() * w_l.unsqueeze(-1)
    one_hot_r = F.one_hot(r, reg_max + 1).float() * w_r.unsqueeze(-1)
    return one_hot_l + one_hot_r

def dfl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n, _, m1 = target.shape
    pred = pred.view(n * 4, m1)
    target = target.view(n * 4, m1)
    return F.kl_div(F.log_softmax(pred, dim=1), target, reduction='batchmean')

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
    def _dfl_to_ltrb(reg_logits: torch.Tensor, dfl_project: torch.Tensor) -> torch.Tensor:
        """Decodes DFL logits to LTRB offsets."""
        reg_max = dfl_project.numel() - 1
        input_shape = reg_logits.shape
        reg_logits = reg_logits.view(*input_shape[:-1], 4, reg_max + 1)
        reg_dist = reg_logits.softmax(dim=-1)
        proj = dfl_project.view((1,) * (reg_dist.dim() - 1) + (-1,))
        return (reg_dist * proj).sum(dim=-1)

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

# ───────────────────── Refactored Main Model ────────────────
class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80,
                 neck_out_ch: int = 96,
                 img_size: int = 224,
                 head_reg_max: int = 8,
                 head_score_thresh: float = 0.05,
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
            # Head returns raw logits per level for loss calculation
            cls_logits, obj_logits, reg_logits = self.head((p3, p4, p5))
            return cls_logits, obj_logits, reg_logits, self.strides
        else:
            # For inference, get anchors and pass them to the head for decoding
            anchor_points, strides_flat = self.anchor_manager()
            return self.head((p3, p4, p5), anchor_points=anchor_points, strides=strides_flat)

# ───────────────────── Utilities (Mostly Unchanged) ────────────────
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

# --- Data & Transforms (Unchanged) ---
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
        T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True) if train else T.Resize((size, size), antialias=True),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1) if train else T.Identity(),
        T.ToDtype(torch.uint8, scale=True),
    ]
    return T.Compose(aug)

def collate_v2(batch):
    return torch.stack(list(zip(*batch))[0], 0), list(zip(*batch))[1]

# ───────────────────── New Assigner: Task-Aligned Assigner ────────────────
class TaskAlignedAssigner:
    """Task-Aligned Assigner from the TOOD paper."""
    def __init__(self, top_k: int = 13, alpha: float = 1.0, beta: float = 6.0):
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(self, pred_scores, pred_boxes, gt_boxes, gt_labels):
        A, C = pred_scores.shape
        M = gt_boxes.shape[0]
        device = pred_scores.device

        if M == 0:
            return (torch.zeros(A, dtype=torch.bool, device=device),
                    torch.full((A,), C, dtype=torch.long, device=device), # Use C as background class index
                    torch.zeros((A,), dtype=torch.float32, device=device))

        # 1. Calculate alignment metric for all (anchor, gt) pairs
        ious = tvops.box_iou(pred_boxes, gt_boxes) # (A, M)
        
        gt_class_indices = gt_labels.reshape(1, M).repeat(A, 1)
        pred_cls_probs = pred_scores.sigmoid().gather(1, gt_class_indices) # (A, M)
        
        alignment_metric = (pred_cls_probs.pow(self.alpha)) * (ious.pow(self.beta))

        # 2. Select top-k candidates for each GT based on the alignment metric
        _, top_k_indices = torch.topk(alignment_metric, self.top_k, dim=0) # (M, top_k) -> Transposed op

        # 3. Assign GTs to anchors
        assigned_gt_inds = torch.full((A,), -1, dtype=torch.long, device=device)
        
        # Mark all top-k candidates as potential positives
        candidate_mask = torch.zeros((A, M), dtype=torch.bool, device=device)
        candidate_mask.scatter_(0, top_k_indices, True)

        # Handle cases where one anchor is assigned to multiple GTs
        # The anchor is assigned to the GT with which it has the highest alignment_metric
        anchor_to_gt_max_metric, anchor_to_gt_argmax = alignment_metric.max(dim=1)
        
        # An anchor is a foreground candidate if it's a top-k candidate for *any* GT
        fg_mask = candidate_mask.any(dim=1)
        # Final assignment: anchor is assigned to the GT that gave it the highest alignment score
        assigned_gt_inds[fg_mask] = anchor_to_gt_argmax[fg_mask]
        
        # 4. Prepare outputs for loss calculation
        assigned_labels = torch.full((A,), C, dtype=torch.long, device=device) # Init with background class
        assigned_scores = torch.zeros((A,), dtype=torch.float32, device=device)
        
        if fg_mask.any():
            pos_gt_inds = assigned_gt_inds[fg_mask]
            assigned_labels[fg_mask] = gt_labels[pos_gt_inds]
            assigned_scores[fg_mask] = ious[fg_mask, pos_gt_inds]

        return fg_mask, assigned_labels, assigned_scores

# ─── utils/ema.py ──────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Keeps a moving average of model parameters.
    Call `update(model)` once after every optimiser step.
    Use `ema_model = ema.ema` for evaluation / export.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device=None):
        # Make a deep copy of the model for accumulating averages
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
        if device is not None:
            self.ema.to(device, non_blocking=True)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:   # keep buffers (e.g. running_mean) as-is
                v.copy_(msd[k])
            else:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def copy_to(self, model: torch.nn.Module):
        """Load EMA weights into a model instance (for export or final eval)."""
        model.load_state_dict(self.ema.state_dict(), strict=False)


# ───────────────────── Updated Training Loop ────────────────────────
def train_epoch(
        model: PicoDet, loader, opt, scaler, assigner: TaskAlignedAssigner, device: torch.device,
        epoch: int, max_epochs: int, debug_prints: bool = True
):
    model.train()
    total_loss, total_fg_count = 0.0, 0
    # Use the existing VarifocalLoss from your library
    vfl_loss_fn = VarifocalLoss(reduction='sum')
    
    anchor_points, anchor_strides_flat = model.anchor_manager()
    
    for i, (imgs, tgts_batch) in enumerate(loader):
        imgs = imgs.to(device)
        
        cls_preds_levels, obj_preds_levels, reg_preds_levels, _ = model(imgs)
        
        bs, nc = imgs.size(0), model.head.nc
        reg_max = model.head.reg_max
        cls_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, nc) for lvl in cls_preds_levels], 1)
        obj_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 1) for lvl in obj_preds_levels], 1)
        reg_p_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(bs, -1, 4 * (reg_max + 1)) for lvl in reg_preds_levels], 1)

        # Decode boxes (detached) to use in the assigner
        decoded_boxes = model.head.decode_predictions(reg_p_flat.detach(), anchor_points, anchor_strides_flat)
        
        # We need the raw logits for the loss function, but the assigner works best with probabilities
        pred_scores_for_assigner = (cls_p_flat.detach() + obj_p_flat.detach()).sigmoid()

        all_fg_masks, all_assigned_labels, all_assigned_scores, all_assigned_boxes = [], [], [], []
        
        for b_idx in range(bs):
            gt_boxes = tgts_batch[b_idx]["boxes"].to(device)
            gt_labels = tgts_batch[b_idx]["labels"].to(device)
            
            fg_mask, assigned_labels, assigned_scores = assigner(pred_scores_for_assigner[b_idx], decoded_boxes[b_idx], gt_boxes, gt_labels)
            
            all_fg_masks.append(fg_mask)
            all_assigned_labels.append(assigned_labels)
            all_assigned_scores.append(assigned_scores)
            
            # Retrieve the GT boxes for the positive anchors
            assigned_gt_boxes = torch.zeros_like(decoded_boxes[b_idx])
            if fg_mask.any():
                pos_indices = torch.where(fg_mask)[0]
                ious = tvops.box_iou(decoded_boxes[b_idx][pos_indices], gt_boxes)
                if ious.numel() > 0:
                    _, gt_matches = ious.max(dim=1)
                    assigned_gt_boxes[pos_indices] = gt_boxes[gt_matches]
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
        
        dfl_target_dist = build_dfl_targets(ltrb_targets, reg_max)
        loss_dfl = dfl_loss(reg_preds_fg, dfl_target_dist)

        pred_ltrb_offsets = model.head._dfl_to_ltrb(reg_preds_fg, model.head.dfl_project_buffer)
        pred_boxes_fg = torch.cat([
            anchor_points_fg - pred_ltrb_offsets[:,:2] * anchor_strides_fg,
            anchor_points_fg + pred_ltrb_offsets[:,2:] * anchor_strides_fg
        ], 1)
        eps = 1e-9
        loss_iou = tvops.complete_box_iou_loss(pred_boxes_fg, assigned_boxes_fg, reduction='sum') / (num_fg + eps)

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
            print(f"E{epoch} B{i}/{len(loader)} | Loss: {loss.item():.4f} (VFL:{loss_vfl.item():.4f}, DFL:{loss_dfl.item():.4f}, CIoU:{loss_iou.item():.4f}) | Num_FG: {num_fg.item()}")

    return total_loss / (i + 1) if i > 0 else total_loss

# --- Validation, QAT, and ONNX Export (Unchanged) ---
# ... (Keep quick_val_iou, qat_prepare, ONNXExportableModel, append_nms_to_onnx) ...
@torch.no_grad()
def quick_val_iou(model: PicoDet, loader, device, epoch_num: int = -1, debug_prints: bool = True):
    model.eval()
    total_iou_sum, num_gt_total = 0., 0
    total_preds_after_nms = 0
    num_images_processed = 0
    
    # Correctly determine number of images in the validation set
    if hasattr(loader.sampler, 'indices'):
        num_images_total = len(loader.sampler.indices)
    else:
        num_images_total = len(loader.dataset)

    for imgs_batch, tgts_batch in loader:
        num_images_processed += imgs_batch.size(0)
        raw_pred_boxes, raw_pred_scores = model(imgs_batch.to(device))
        
        for b_idx in range(imgs_batch.size(0)):
            gt_boxes = tgts_batch[b_idx]["boxes"].to(device)
            if gt_boxes.numel() == 0: continue
            num_gt_total += gt_boxes.shape[0]

            scores_per_anchor, labels_per_anchor = torch.max(raw_pred_scores[b_idx], dim=1)
            keep_mask = scores_per_anchor >= model.score_th
            
            if not keep_mask.any(): continue

            boxes_pre_nms = raw_pred_boxes[b_idx][keep_mask]
            scores_pre_nms = scores_per_anchor[keep_mask]
            labels_pre_nms = labels_per_anchor[keep_mask]

            nms_indices = tvops.batched_nms(boxes_pre_nms, scores_pre_nms, labels_pre_nms, model.iou_th)
            
            if len(nms_indices) == 0: continue
            
            final_boxes = boxes_pre_nms[nms_indices[:model.max_det]]
            total_preds_after_nms += final_boxes.shape[0]
            
            iou_matrix = tvops.box_iou(final_boxes, gt_boxes)
            if iou_matrix.numel() > 0:
                max_iou_per_gt, _ = iou_matrix.max(dim=0)
                total_iou_sum += max_iou_per_gt.sum().item()

    avg_preds_after = total_preds_after_nms / num_images_processed if num_images_processed > 0 else 0
    final_mean_iou = total_iou_sum / num_gt_total if num_gt_total > 0 else 0
    
    if debug_prints:
        print(f"--- Val E{epoch_num} ---")
        print(f"Images: {num_images_processed}/{num_images_total}, Total GTs: {num_gt_total}")
        print(f"Avg preds/img (after NMS): {avg_preds_after:.2f}")
        print(f"Validation Mean IoU: {final_mean_iou:.4f}")
    
    return final_mean_iou

def qat_prepare(model: nn.Module, example_input: torch.Tensor) -> torch.fx.GraphModule:
    qconfig = QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                      weight=MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    qmap = get_default_qat_qconfig_mapping("x86").set_global(qconfig).set_module_name('pre', None)
    return prepare_qat_fx(model.cpu().train(), qmap, example_input.cpu())

class ONNXExportableModel(nn.Module):
    def __init__(self, qat_model: PicoDet):
        super().__init__()
        self.quantized_model = convert_fx(copy.deepcopy(qat_model).cpu().eval())
    def forward(self, x: torch.Tensor):
        return self.quantized_model(x)

def append_nms_to_onnx(in_path: str, out_path: str, score_thresh: float, iou_thresh: float, max_det: int):
    model = onnx.load(in_path)
    graph = model.graph
    
    graph.initializer.extend([
        oh.make_tensor("iou_threshold", TP.FLOAT, [], [iou_thresh]),
        oh.make_tensor("score_threshold", TP.FLOAT, [], [score_thresh]),
        oh.make_tensor("max_output_boxes_per_class", TP.INT64, [], [max_det]),
    ])
    
    scores_transposed = "scores_transposed"
    graph.node.append(oh.make_node("Transpose", ["raw_scores"], [scores_transposed], perm=[0, 2, 1]))
    
    selected_indices = "selected_indices"
    graph.node.append(oh.make_node(
        "NonMaxSuppression",
        inputs=["raw_boxes", scores_transposed, "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        outputs=[selected_indices]
    ))
    
    del graph.output[:]
    graph.output.extend([
        oh.make_tensor_value_info(selected_indices, TP.INT64, ['num_detections', 3]),
    ])

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"[SAVE] Final ONNX with NMS indices output -> {out_path}")

# ───────────────────────── Main Execution ────────────────────────────────
def main(argv: List[str] | None = None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--coco_root', default='coco')
    pa.add_argument('--arch', default='mnv4c', choices=['mnv4c'])
    pa.add_argument('--epochs', type=int, default=10) 
    pa.add_argument('--qat_epochs', type=int, default=3) 
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--workers', type=int, default=0)
    pa.add_argument('--device', default=None)
    pa.add_argument('--out', default='picodet_int8_refactored.onnx')
    cfg = pa.parse_args(argv)
    BACKBONE_FREEZE_EPOCHS = 2

    dev = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[INFO] Device: {dev}')

    backbone, feat_chs = get_backbone(cfg.arch, img_size=IMG_SIZE)
    model = PicoDet(backbone, feat_chs, num_classes=80, neck_out_ch=96, img_size=IMG_SIZE).to(dev)

    train_ds = CocoDetectionV2(f"{cfg.coco_root}/train2017", f"{cfg.coco_root}/annotations/instances_train2017.json", CANONICAL_COCO80_MAP, build_transforms(IMG_SIZE, True))
    val_ds = CocoDetectionV2(f"{cfg.coco_root}/val2017", f"{cfg.coco_root}/annotations/instances_val2017.json", CANONICAL_COCO80_MAP, build_transforms(IMG_SIZE, False))
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers, collate_fn=collate_v2, pin_memory=True, persistent_workers=bool(cfg.workers))
    vl_loader = DataLoader(val_ds, batch_size=cfg.batch*2, num_workers=cfg.workers, collate_fn=collate_v2, pin_memory=True)

    lr = 0.005
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - 5, eta_min=0)
    sch = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[5])
    scaler = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))
    
    assigner = TaskAlignedAssigner(top_k=13)
    ema = ModelEMA(model, decay=0.999)
    
    print(f"[INFO] Starting FP32 training for {cfg.epochs} epochs...")
    for ep in range(cfg.epochs):
        if ep < BACKBONE_FREEZE_EPOCHS:
            if ep == 0:
                for p in model.backbone.parameters(): p.requires_grad = False
                print(f"[INFO] Backbone frozen for {BACKBONE_FREEZE_EPOCHS} epochs…")
        elif ep == BACKBONE_FREEZE_EPOCHS:
            for p in model.backbone.parameters(): p.requires_grad = True
            print("[INFO] Backbone unfrozen – full network now training")

        train_loss = train_epoch(model, tr_loader, opt, scaler, assigner, dev, ep, cfg.epochs)
        ema.update(model)  
        val_iou = quick_val_iou(ema.ema, vl_loader, dev, epoch_num=ep)
        print(f"Epoch {ep+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {opt.param_groups[0]['lr']:.6f}\n")
        sch.step()

    print("\n[INFO] Preparing model for QAT...")
    qat_model = qat_prepare(model, torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8))
    qat_model.to(dev)

    opt_q = SGD(qat_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=4e-5)
    scaler_q = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))
    
    print(f"[INFO] Starting QAT finetuning for {cfg.qat_epochs} epochs...")
    ema_q = ModelEMA(qat_model, decay=0.99)
    for qep in range(cfg.qat_epochs):
        train_loss_q = train_epoch(qat_model, tr_loader, opt_q, scaler_q, assigner, dev, qep, cfg.qat_epochs)
        val_iou_q = quick_val_iou(qat_model, vl_loader, dev, epoch_num=qep)
        print(f"QAT Epoch {qep+1}/{cfg.qat_epochs} | Train Loss: {train_loss_q:.4f} | Val IoU: {val_iou_q:.4f}\n")
    
    print("\n[INFO] Exporting to ONNX...")
    # final_exportable_model = ONNXExportableModel(qat_model)
    export_model = copy.deepcopy(qat_model).cpu()
    ema_q.copy_to(export_model)          # use the averaged weights
    final_exportable_model = ONNXExportableModel(export_model)
    
    temp_onnx_path = cfg.out.replace(".onnx", "_temp.onnx")
    torch.onnx.export(
        final_exportable_model,
        torch.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),
        temp_onnx_path,
        input_names=['images_uint8'],
        output_names=['raw_boxes', 'raw_scores'],
        dynamic_axes={'images_uint8': {0: 'batch'}, 'raw_boxes': {0: 'batch'}, 'raw_scores': {0: 'batch'}},
        opset_version=18
    )
    print(f"[SAVE] Intermediate ONNX (INT8 core, no NMS) -> {temp_onnx_path}")
    
    append_nms_to_onnx(temp_onnx_path, cfg.out, model.score_th, model.iou_th, model.max_det)


if __name__ == '__main__':
    main()