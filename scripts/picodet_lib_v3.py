# picodet_lib.py
# PyTorch 2.7 / opset 18
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

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
            nn.BatchNorm2d(init_ch), nn.ReLU6(inplace=inplace_act)
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
        for m in self.cv3.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

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

class AnchorManager(nn.Module):
    """Generates and provides anchor points and strides for all FPN levels."""
    def __init__(self, img_size: int, num_levels: int, strides: Tuple[int, ...]):
        super().__init__()
        if len(strides) != num_levels:
            raise ValueError("Length of strides must match num_levels")

        self.num_levels = num_levels
        # These are not parameters, so register as non-persistent buffers
        self.register_buffer('strides_all', torch.tensor(strides, dtype=torch.float32), persistent=False)

        anchor_points_list = []
        strides_list = []
        for i in range(num_levels):
            s = strides[i]
            h_level, w_level = math.ceil(img_size / s), math.ceil(img_size / s)
            yv, xv = torch.meshgrid(
                torch.arange(h_level, dtype=torch.float32),
                torch.arange(w_level, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack((xv, yv), dim=2).reshape(h_level * w_level, 2)
            anchor_points = (grid + 0.5) * s
            level_strides = torch.full((len(anchor_points), 1), s, dtype=torch.float32)

            anchor_points_list.append(anchor_points)
            strides_list.append(level_strides)
        
        self.register_buffer('anchor_points_cat', torch.cat(anchor_points_list, dim=0), persistent=False)
        self.register_buffer('strides_cat', torch.cat(strides_list, dim=0), persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return concatenated anchors and strides for all levels
        return self.anchor_points_cat, self.strides_cat

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
        
        dfl_project_tensor = torch.arange(self.reg_max + 1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', dfl_project_tensor, persistent=False)

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
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(num_feats, 4 * (self.reg_max + 1), 1) for _ in range(self.nl)]
        )
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
        # reg_logits shape: (B, N_anchors, 4 * (reg_max + 1)) OR (N_anchors, 4 * (reg_max + 1))
        # dfl_project shape: (reg_max + 1)
        reg_max = dfl_project.numel() - 1
        input_shape = reg_logits.shape
        reg_logits = reg_logits.view(*input_shape[:-1], 4, reg_max + 1)
        
        # Softmax over the distribution dimension
        reg_dist = reg_logits.softmax(dim=-1)
        
        # Multiply by the projection vector and sum
        # proj shape needs to be broadcastable: (1, 1, ..., 1, reg_max+1)
        proj = dfl_project.view((1,) * (reg_dist.dim() - 1) + (-1,))
        ltrb_offsets = (reg_dist * proj).sum(dim=-1)
        return ltrb_offsets

    def decode_predictions(self, reg_logits_flat, anchor_points, strides):
        """Decodes flat regression logits into bounding boxes."""
        # reg_logits_flat: (B, A, 4*(reg_max+1))
        # anchor_points: (A, 2)
        # strides: (A, 1)
        ltrb_offsets = self._dfl_to_ltrb(reg_logits_flat, self.dfl_project_buffer)
        ltrb_pixels = ltrb_offsets * strides
        
        # anchor_points need to be broadcastable to (B, A, 2)
        ap_expanded = anchor_points.unsqueeze(0)
        
        x1y1 = ap_expanded - ltrb_pixels[..., :2]
        x2y2 = ap_expanded + ltrb_pixels[..., 2:]
        return torch.cat([x1y1, x2y2], dim=-1)

    def forward(self, neck_feature_maps, anchor_points=None, strides=None):
        raw_cls_logits_levels: List[torch.Tensor] = []
        raw_obj_logits_levels: List[torch.Tensor] = []
        raw_reg_logits_levels: List[torch.Tensor] = []

        for i, f_map_level in enumerate(neck_feature_maps):
            common_feat = self.cls_conv(f_map_level) # Use one common path
            reg_feat = self.reg_conv(f_map_level)
            raw_cls_logits_levels.append(self.cls_pred[i](common_feat))
            raw_obj_logits_levels.append(self.obj_pred[i](common_feat))
            raw_reg_logits_levels.append(self.reg_pred[i](reg_feat))

        if self.training:
            # Training path returns raw logits per level for loss calculation
            return tuple(raw_cls_logits_levels), tuple(raw_obj_logits_levels), tuple(raw_reg_logits_levels)
        else:
            # Inference path decodes predictions
            if anchor_points is None or strides is None:
                raise ValueError("anchor_points and strides must be provided for inference.")
            
            # Flatten all predictions across levels
            cls_logits_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, self.nc) for lvl in raw_cls_logits_levels], 1)
            obj_logits_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 1) for lvl in raw_obj_logits_levels], 1)
            reg_logits_flat = torch.cat([lvl.permute(0, 2, 3, 1).reshape(lvl.shape[0], -1, 4 * (self.reg_max + 1)) for lvl in raw_reg_logits_levels], 1)

            # Decode boxes
            decoded_boxes = self.decode_predictions(reg_logits_flat, anchor_points, strides)
            
            # Calculate scores consistently with training loss
            decoded_scores = cls_logits_flat.sigmoid() * obj_logits_flat.sigmoid()

            return decoded_boxes, decoded_scores

class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80,
                 neck_out_ch: int = 96,
                 img_size: int = 224,
                 head_reg_max: int = 8,
                 # NMS parameters for ONNX export and validation config
                 head_max_det: int = 100,
                 head_score_thresh: float = 0.05,
                 head_nms_iou: float = 0.6,
                 inplace_act_for_head_neck: bool = False):
        super().__init__()
        self.pre = ResizeNorm(size=(img_size, img_size))
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch, inplace_act=inplace_act_for_head_neck)
        
        num_fpn_levels = len(feat_chs)
        # These are now model-level attributes
        self.strides = (8, 16, 32)
        self.img_size = img_size
        self.score_th = head_score_thresh
        self.iou_th = head_nms_iou
        self.max_det = head_max_det
        
        self.anchor_manager = AnchorManager(
            img_size=self.img_size,
            num_levels=num_fpn_levels,
            strides=self.strides
        )
        
        self.head = PicoDetHead(
            num_classes=num_classes, 
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=num_fpn_levels,
            inplace_act=inplace_act_for_head_neck
        )

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        
        if self.training:
            # For training, head only needs features. Anchor info is handled in the training script.
            # Head returns tuple of raw logits per level.
            # Pass strides along for the loss function.
            cls_logits, obj_logits, reg_logits = self.head((p3, p4, p5))
            return cls_logits, obj_logits, reg_logits, self.strides
        else:
            # For inference, get anchors and pass them to the head for decoding.
            # Head returns decoded boxes and scores.
            anchor_points, strides_flat = self.anchor_manager()
            return self.head((p3, p4, p5), anchor_points=anchor_points, strides=strides_flat)

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

def get_backbone(arch: str, ckpt: str | None, img_size: int = 224):
    temp_device_for_init = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == "mnv4c":
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
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        print(f"[INFO] Custom MNv4 feature channels: {feat_chs}")
    else:
        raise ValueError(f'Unknown arch {arch}')
    net.train()
    return net, feat_chs

__all__ = [
    'GhostConv', 'DWConv', 'CSPBlock', 'CSPPAN',
    'VarifocalLoss', 'dfl_loss', 'build_dfl_targets',
    'AnchorManager', 'PicoDetHead', 'ResizeNorm', 'get_backbone', 'PicoDet'
]