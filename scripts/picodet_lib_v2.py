# picodet_lib.py
# PyTorch 2.7 / opset‑18
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from torchvision.models.feature_extraction import create_feature_extractor

try:
    folder_to_add_pos = [
        r"/Users/colincatlin/Documents-NoCloud/qat_onnx_models/scripts",  # mac
        r"C:\Users\Colin\qat_onnx_models\scripts",  # Windows
        r"/home/colin/qat_onnx_models/scripts",  # Linux 1
        r"/home/colin/img_data",
    ]
    for folder_to_add in folder_to_add_pos:
        custom_module_path = os.path.join(folder_to_add, "customMobilenetNetv4.py")
        if os.path.exists(custom_module_path):
            if folder_to_add not in sys.path:
                 sys.path.insert(0, folder_to_add)
            break
    from customMobilenetNetv4 import MobileNetV4ConvSmallPico, MobileNetV4
    print(f"[INFO] Successfully imported customMobilenetNetv4 from {folder_to_add}")
except ImportError as e:
    print(f"Warning: Failed to import customMobilenetNetv4.py ({e}). 'mnv4c' backbone will not be available.")
    MobileNetV4ConvSmallPico = None
except Exception as e: # Catch other potential errors like incorrect path structure
    print(f"An error occurred during custom backbone import setup: {e}")
    MobileNetV4ConvSmallPico = None

# ───────────────────────────── layers ──────────────────────────────
class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, p: int = None, groups: int = 1, bias: bool = False, inplace_act: bool = False, act_layer=nn.ReLU6):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = act_layer(inplace=inplace_act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv1x1(nn.Module):
    """
    MobileOne-style Reparameterized 1x1 Conv.
    
    TRAINING: Multiple parallel 1x1 branches for better gradient flow.
    INFERENCE: Fuses into a single 1x1 Conv (zero latency overhead).
    
    CRITICAL FOR QAT: Call switch_to_deploy() AFTER FP32 training,
    BEFORE qat_prepare(). This ensures QAT trains the fused topology.
    """
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU6, inplace_act=False, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act_layer(inplace=inplace_act)

        if deploy:
            # Inference mode: single 1x1 conv
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        else:
            # Training mode: Multi-branch for over-parameterization
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_scale = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        return self.act(self.rbr_dense(x) + self.rbr_scale(x))

    def switch_to_deploy(self):
        """Fuses multi-branch into single conv. Must be called before QAT!"""
        if self.deploy:
            return
        
        k_dense, b_dense = self._fuse_bn_tensor(self.rbr_dense[0], self.rbr_dense[1])
        k_scale, b_scale = self._fuse_bn_tensor(self.rbr_scale[0], self.rbr_scale[1])
        
        final_weight = k_dense + k_scale
        final_bias = b_dense + b_scale

        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, bias=True)
        self.rbr_reparam.weight.data = final_weight
        self.rbr_reparam.bias.data = final_bias

        del self.rbr_dense
        del self.rbr_scale
        self.deploy = True

    def _fuse_bn_tensor(self, conv, bn):
        """Fuse Conv + BN into single Conv."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class GhostConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            k: int = 1,
            s: int = 1,
            dw_size: int = 3,
            ratio: int = 2,
            inplace_act: bool = False,
            act_layer = nn.ReLU6,
         ):
        super().__init__()
        self.c_out = c_out
        init_ch = min(math.ceil(c_out / ratio), c_out)
        self.cheap_ch = c_out - init_ch

        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            act_layer(inplace=inplace_act),
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
            return torch.cat([y_primary, self.cheap(y_primary)], 1)
        return y_primary

    @property
    def out_channels(self):
        return self.primary[0].out_channels + (self.cheap[0].out_channels if self.cheap else 0)


class DWConv(nn.Module):
    def __init__(self, c: int, k: int = 5, inplace_act: bool = False, act_layer=nn.ReLU6):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = act_layer(inplace=inplace_act)

    def forward(self, x): return self.act(self.bn(self.dw(x)))


class CSPBlock(nn.Module):
    def __init__(self, c: int, n: int = 1, m_k: int = 1, inplace_act: bool = False):
        super().__init__()
        self.cv1 = GhostConv(c, c // 2, 1, inplace_act=inplace_act)
        self.cv2 = GhostConv(c, c // 2, 1, inplace_act=inplace_act)
        self.m = nn.Sequential(*[GhostConv(c // 2, c // 2, k=m_k, inplace_act=inplace_act) for _ in range(n)])
        self.cv3 = GhostConv(c, c, 1, inplace_act=inplace_act)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ───────────────────────────── neck ────────────────────────────────
class CSPPAN(nn.Module):
    """CSP-PAN neck that dynamically handles N input feature levels (minimum 2)."""
    def __init__(
            self,
            in_chs: Tuple[int, ...] = (40, 112, 160),
            out_ch: int = 96,
            lat_k: int = 5,
            inplace_act: bool = False,
            act_layer=nn.ReLU6,
            use_rep_conv: bool = True  # Enable RepConv for better training
    ):
        super().__init__()
        self.in_chs = in_chs
        self.num_levels = len(in_chs)
        assert self.num_levels >= 2, f"CSPPAN requires at least 2 feature levels, got {self.num_levels}"
        
        # Reduce convs: RepConv1x1 for superior training, fuses to standard 1x1 for inference
        ReduceBlock = RepConv1x1 if use_rep_conv else ConvBNAct
        reduce_list = []
        for i, c in enumerate(in_chs):
            if use_rep_conv:
                reduce_list.append(RepConv1x1(c, out_ch, act_layer=act_layer, inplace_act=inplace_act))
            else:
                reduce_list.append(ConvBNAct(c, out_ch, 1, inplace_act=inplace_act))
        self.reduce = nn.ModuleList(reduce_list)
        
        # Lateral convs for all but the deepest level
        self.lat = nn.ModuleList([DWConv(out_ch, k=lat_k, inplace_act=inplace_act) for _ in range(self.num_levels - 1)])
        
        # Output blocks: deeper levels get more complex CSPBlocks
        self.out = nn.ModuleList([
            CSPBlock(out_ch, n=2 if i == self.num_levels - 1 else 1, 
                     m_k=3 if i == self.num_levels - 1 else 1, 
                     inplace_act=inplace_act) 
            for i in range(self.num_levels)
        ])

    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # top-down pathway
        p = [None] * self.num_levels
        p[-1] = self.reduce[-1](features[-1])
        
        for i in range(self.num_levels - 2, -1, -1):
            p[i] = self.reduce[i](features[i]) + F.interpolate(p[i + 1], scale_factor=2, mode='nearest')
        
        for i in range(self.num_levels - 1):
            p[i] = self.lat[i](p[i])
        
        # bottom-up pathway
        for i in range(1, self.num_levels):
            p[i] = p[i] + F.max_pool2d(p[i - 1], 2)
        
        return tuple(self.out[i](p[i]) for i in range(self.num_levels))
    
    def switch_to_deploy(self):
        """Fuse RepConv branches. Call AFTER FP32 training, BEFORE QAT!"""
        for m in self.reduce:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        print("[INFO] CSPPAN neck switched to deploy mode (RepConv branches fused).")


# --- tiny ESE gate ----------------------------------------
class ESE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1, bias=False)
    def forward(self, x):
        w = torch.sigmoid(self.fc(x.mean((2,3), keepdim=True)))
        return x * w

class SpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention Module (CBAM-inspired, QAT-friendly).
    
    Modifications from standard CBAM for object detection:
    - kernel_size=3 (default) works better on small feature maps (8x8 at stride 32)
    - Uses only avg pooling (max pooling doesn't quantize well for QAT)
    - Added BatchNorm for training stability
    - Depthwise-separable pattern for efficiency
    
    Place after neck reduce convs or before detection heads.
    Adds ~0.1ms latency on mobile, can improve localization by 1-2% IoU.
    """
    def __init__(self, kernel_size: int = 3, use_max_pool: bool = False):
        super(SpatialAttention, self).__init__()
        self.use_max_pool = use_max_pool
        padding = (kernel_size - 1) // 2
        in_ch = 2 if use_max_pool else 1
        # Conv + BN for better training; bias=False since BN follows
        self.conv = nn.Conv2d(in_ch, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avg pool across channels (always used, QAT-safe)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        if self.use_max_pool:
            # Optional: max pool (disable for pure QAT compatibility)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            attention_input = torch.cat([avg_out, max_out], dim=1)
        else:
            attention_input = avg_out
        
        attention_map = self.sigmoid(self.bn(self.conv(attention_input)))
        return x * attention_map

# ─────────────────────── losses (VFL · DFL · IoU) ──────────────────
class VarifocalLoss(nn.Module):
    """
    Varifocal Loss with enhanced background penalty.
    
    For detection models with high FP rates (FP/img > 10), the background_weight
    parameter can be increased to penalize false positives more heavily.
    
    Args:
        alpha: Weight for negative (background) samples in focal weighting
        gamma: Focusing parameter (higher = more focus on hard examples)
        background_weight: Additional multiplier for background loss (default 1.0)
                          Set to 1.5-2.0 if model is "trigger happy" with FPs
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2., 
                 background_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.background_weight = background_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets_q: torch.Tensor):
        p = logits.sigmoid()
        with torch.no_grad():
            # Standard VFL weighting
            weight = torch.where(targets_q > 0,
                                  targets_q,
                                  self.alpha * p.pow(self.gamma))
            # Apply additional background penalty if configured
            if self.background_weight != 1.0:
                bg_mask = (targets_q == 0)
                weight = torch.where(bg_mask, weight * self.background_weight, weight)
        
        loss = F.binary_cross_entropy_with_logits(logits, targets_q, weight, reduction='none')
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss

class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) for object detection.

    This loss is designed to address the inconsistency between classification
    scores and localization quality (IoU). It uses the IoU between the predicted
    bounding box and the ground truth box as the target for the classification
    score of the positive class.

    The formula is: QFL(σ) = -|y - σ|^β * ((1 - y)log(1 - σ) + y*log(σ))
    where:
      - σ (sigma) is the predicted score (output of sigmoid).
      - y is the quality target (e.g., IoU score).
      - β (beta) is the focusing parameter (gamma in original Focal Loss).

    Args:
        beta (float): The focusing parameter (also called gamma). Defaults to 2.0.
    """
    def __init__(self, beta: float = 2.0, reduction: str = 'sum'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pred_scores = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        modulating_factor = (targets - pred_scores).abs().pow(self.beta)
        loss = modulating_factor * bce_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            raise ValueError(f"unrecognized reduction {self.reduction}")

def build_dfl_targets(offsets: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Soft distribution targets for DFL."""
    x = offsets.clamp(0, reg_max)
    l = x.floor().long()
    r = (l + 1).clamp(max=reg_max)
    w_r = x - l.float()
    w_l = 1. - w_r
    one_hot_l = F.one_hot(l, reg_max + 1).float() * w_l.unsqueeze(-1)
    one_hot_r = F.one_hot(r, reg_max + 1).float() * w_r.unsqueeze(-1)
    return one_hot_l + one_hot_r  # (N,4,M+1)


def dfl_loss(pred_logits, target_dist, eps: float = 1e-7):
    """DFL loss using KL-divergence."""
    n, _, K = target_dist.shape
    pred = pred_logits.view(n * 4, K)
    tgt = target_dist.view(n * 4, K).clamp_min(eps)
    tgt = tgt / tgt.sum(dim=1, keepdim=True)
    return F.kl_div(F.log_softmax(pred, dim=1), tgt, reduction='batchmean', log_target=False)


class PicoDetHead(nn.Module):
    def __init__(self, num_classes: int = 80,
                 reg_max: int = 8,
                 num_feats: int = 96,
                 num_levels: int = 3,
                 strides: Tuple[int, ...] = None,
                 max_det: int = 100, 
                 score_thresh: float = 0.05, 
                 nms_iou: float = 0.6,
                 img_size: int = 224,
                 cls_conv_depth: int = 3,
                 reg_conv_depth: int = 2,
                 inplace_act: bool = False):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.nl = num_levels
        self.max_det = max_det
        self.score_th = score_thresh
        self.iou_th = nms_iou
        self.reg_conv_depth = reg_conv_depth
        self.cls_conv_depth = cls_conv_depth
        self.img_size = img_size
        first_cls_conv_k = 3

        if strides is None:
            default_strides_map = {
                2: (8, 16),
                3: (8, 16, 32),
                4: (4, 8, 16, 32),
                5: (4, 8, 16, 32, 64),
            }
            strides = default_strides_map.get(num_levels)
            if strides is None:
                raise ValueError(f"No default strides for num_levels={num_levels}. Please provide explicit strides.")
        
        assert len(strides) == num_levels, \
            f"strides length ({len(strides)}) must match num_levels ({num_levels})"
        
        strides_tensor = torch.tensor(strides, dtype=torch.float32)
        self.register_buffer('strides_buffer', strides_tensor, persistent=False)
        
        dfl_project_tensor = torch.arange(self.reg_max + 1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', dfl_project_tensor, persistent=False)

        if self.cls_conv_depth <= 1:
            self.cls_conv = nn.ModuleList([
                nn.Sequential(
                    GhostConv(num_feats, num_feats, k=first_cls_conv_k, inplace_act=inplace_act)
                ) for _ in range(self.nl)
            ])
        else:
            self.cls_conv = nn.ModuleList([
                nn.Sequential(
                    GhostConv(num_feats, num_feats, k=first_cls_conv_k, inplace_act=inplace_act),
                    *[GhostConv(num_feats, num_feats, inplace_act=inplace_act) for _ in range(self.cls_conv_depth - 1)]
                ) for _ in range(self.nl)
            ])
        self.reg_conv = nn.ModuleList([
            nn.Sequential(*[GhostConv(num_feats, num_feats, ratio=2.0, inplace_act=inplace_act) for _ in range(self.reg_conv_depth)])
            for _ in range(self.nl)
        ])
        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(num_feats, 4 * (self.reg_max + 1), 1) for _ in range(self.nl)]
        )
        self.cls_ese = nn.ModuleList([ESE(num_feats) for _ in range(self.nl)])
        self._initialize_biases()

        for i in range(self.nl):
            s = self.strides_buffer[i].item()
            H_level = math.ceil(img_size / s)
            W_level = math.ceil(img_size / s)
            yv, xv = torch.meshgrid(
                torch.arange(H_level, dtype=torch.float32),
                torch.arange(W_level, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack((xv, yv), dim=2).reshape(H_level * W_level, 2)
            anchor_points_center = (grid + 0.5) * s
            self.register_buffer(f'anchor_points_level_{i}', anchor_points_center, persistent=False)

    def _initialize_biases(self):
        # Initialize classification head
        cls_prior = 0.02
        cls_bias = -math.log((1 - cls_prior) / cls_prior)
        for conv in self.cls_pred:
            nn.init.constant_(conv.bias, cls_bias)
            nn.init.normal_(conv.weight, std=0.01)  # Crucial for Focal Loss stability

        # Initialize regression head
        peak_prob = 0.90
        delta = math.log(peak_prob / (1 - peak_prob))
        pattern = torch.zeros(4 * (self.reg_max + 1), device=self.cls_pred[0].weight.device)
        for i in range(4):
            pattern[i * (self.reg_max + 1)] = delta
            
        for conv in self.reg_pred:
            nn.init.normal_(conv.weight, std=0.01)  # Allow gradient flow from start
            conv.bias.data.copy_(pattern)

    def _dfl_to_ltrb_inference(self, x_reg_logits_3d: torch.Tensor) -> torch.Tensor:
        b, n_anchors_img, _ = x_reg_logits_3d.shape
        x_reg_logits_reshaped = x_reg_logits_3d.view(b, n_anchors_img, 4, self.reg_max + 1)
        x_softmax = x_reg_logits_reshaped.softmax(dim=3)
        proj = self.dfl_project_buffer.view(1, 1, 1, -1) 
        ltrb_offsets = (x_softmax * proj).sum(dim=3)
        return ltrb_offsets

    @staticmethod
    def dfl_decode_for_training(
        x_reg_logits: torch.Tensor, 
        dfl_project_buffer: torch.Tensor,
        reg_max_val: int
    ) -> torch.Tensor:
        
        input_shape = x_reg_logits.shape
        if x_reg_logits.ndim == 2: # (N_total_anchors, 4 * (reg_max + 1))
            n_anchors = input_shape[0]
            # Use reg_max_val passed as argument
            x_reg_logits = x_reg_logits.view(n_anchors, 4, reg_max_val + 1) 
            x_softmax = x_reg_logits.softmax(dim=2)
            # Use dfl_project_buffer passed as argument
            proj = dfl_project_buffer.view(1, 1, -1) 
            ltrb_offsets = (x_softmax * proj).sum(dim=2) # (N_total_anchors, 4)
        else:
            raise ValueError(
                f"PicoDetHead.dfl_decode_for_training expects 2D input (N, 4*(M+1)), got {x_reg_logits.ndim}D"
            )
        return ltrb_offsets

    def _decode_predictions_for_level(
        self,
        cls_logit: torch.Tensor, reg_logit: torch.Tensor,
        level_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H_feat, W_feat = cls_logit.shape
        stride = self.strides_buffer[level_idx]

        cls_logit_perm = cls_logit.permute(0,2,3,1).reshape(B, H_feat*W_feat, self.nc)
        scores = cls_logit_perm.sigmoid()
        
        max_scores = scores.max(dim=-1).values
        score_threshold = 0.01
        keep_mask = max_scores > score_threshold
        fallback_max = max_scores.max(dim=1, keepdim=True).values
        keep_mask = keep_mask | (max_scores == fallback_max)

        if not keep_mask.any():
            return (
                torch.zeros(B, H_feat*W_feat, 4, device=cls_logit.device, dtype=cls_logit.dtype),
                scores.new_zeros(B, H_feat*W_feat, self.nc)
            )

        # Build grid on the fly
        yv, xv = torch.meshgrid(
            torch.arange(H_feat, device=cls_logit.device),
            torch.arange(W_feat, device=cls_logit.device),
            indexing='ij'
        )
        anchor_centers = (torch.stack((xv, yv), dim=2).view(-1, 2) + 0.5) * stride

        reg_logit_perm = reg_logit.permute(0,2,3,1).reshape(B, H_feat*W_feat, 4*(self.reg_max+1))
        ltrb = self._dfl_to_ltrb_inference(reg_logit_perm) * stride

        x1 = anchor_centers[:,0].unsqueeze(0) - ltrb[...,0]
        y1 = anchor_centers[:,1].unsqueeze(0) - ltrb[...,1]
        x2 = anchor_centers[:,0].unsqueeze(0) + ltrb[...,2]
        y2 = anchor_centers[:,1].unsqueeze(0) + ltrb[...,3]
        boxes = torch.stack([x1,y1,x2,y2], dim=-1)

        mask_expanded = keep_mask.unsqueeze(-1)
        boxes = boxes.masked_fill(~mask_expanded, 0.0)
        scores = scores.masked_fill(~mask_expanded, 0.0)

        return boxes, scores

    def forward(self, neck_feature_maps: Tuple[torch.Tensor, ...]):
        raw_cls_logits_levels: List[torch.Tensor] = []
        raw_reg_logits_levels: List[torch.Tensor] = []

        for i, f_map_level in enumerate(neck_feature_maps):
            # Unshared head: Use cls_conv[i] and reg_conv[i]
            cls_common_feat = self.cls_ese[i](self.cls_conv[i](f_map_level))
            raw_cls_logits_levels.append(self.cls_pred[i](cls_common_feat))
            raw_reg_logits_levels.append(self.reg_pred[i](self.reg_conv[i](f_map_level)))

        if self.training:
            return (
                tuple(raw_cls_logits_levels),
                tuple(raw_reg_logits_levels),
                tuple(self.strides_buffer[i] for i in range(self.nl))
            )
        else:
            decoded_boxes_all_levels: List[torch.Tensor] = []
            decoded_scores_all_levels: List[torch.Tensor] = []
            for i in range(self.nl):
                boxes_level, scores_level = self._decode_predictions_for_level(
                    raw_cls_logits_levels[i], raw_reg_logits_levels[i], i
                )
                decoded_boxes_all_levels.append(boxes_level)
                decoded_scores_all_levels.append(scores_level)
            return torch.cat(decoded_boxes_all_levels, dim=1), torch.cat(decoded_scores_all_levels, dim=1)


class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, ...],
                 num_classes: int = 80,
                 neck_out_ch: int = 96,
                 img_size: int = 224,
                 head_reg_max: int = 8,
                 head_max_det: int = 100,
                 head_score_thresh: float = 0.08,
                 head_nms_iou: float = 0.55,
                 head_strides: Tuple[int, ...] = None,
                 cls_conv_depth: int = 3,
                 reg_conv_depth: int = 2,
                 lat_k: int = 5,
                 inplace_act_for_head_neck: bool = False):
        super().__init__()
        self.pre = ResizeNorm(size=(img_size, img_size)) 
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch, lat_k=lat_k, inplace_act=inplace_act_for_head_neck)
        num_fpn_levels = len(feat_chs)
        self.num_fpn_levels = num_fpn_levels
        self.img_size = img_size
        
        self.head = PicoDetHead(
            num_classes=num_classes, 
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=num_fpn_levels,
            strides=head_strides,
            max_det=head_max_det,
            score_thresh=head_score_thresh,
            nms_iou=head_nms_iou,
            img_size=img_size,
            reg_conv_depth=reg_conv_depth,
            cls_conv_depth=cls_conv_depth,
            inplace_act=inplace_act_for_head_neck,
        )

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        backbone_features = self.backbone(x)
        neck_outputs = self.neck(backbone_features)
        return self.head(neck_outputs)


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


# ───────────────────────── backbone util ──────────────────────────
class TVExtractorWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, return_nodes_dict: dict):
        super().__init__()
        self.extractor = create_feature_extractor(base_model, return_nodes_dict)
        self.output_keys = list(return_nodes_dict.values())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feature_dict = self.extractor(x)
        return tuple(feature_dict[key] for key in self.output_keys)

def pick_nodes_by_stride(model: nn.Module, img_size: int = 256, desired: Tuple[int, ...] = (8, 16, 32)) -> dict:
    tmp = {}
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    hooks = [m.register_forward_hook(
        lambda mod, inp, out, n=name: tmp.setdefault(n, out.detach().clone().cpu()))
             for name, m in model.named_modules()]
    
    model.eval()
    try:
        model(torch.randn(1, 3, img_size, img_size, device=device))
    finally:
        for h in hooks: h.remove()
    model.train()

    H_in = img_size
    stride_to_name = {}
    sorted_items = sorted(tmp.items(), key=lambda x: len(x[0]))

    for name, feat_tensor in sorted_items:
        if not isinstance(feat_tensor, torch.Tensor) or feat_tensor.ndim < 4:
            continue
        if H_in % feat_tensor.shape[-2] == 0:
            stride = H_in // feat_tensor.shape[-2]
            if stride in desired and stride not in stride_to_name and '.' in name:
                stride_to_name[stride] = name
    
    if len(stride_to_name) != len(desired):
        warnings.warn(
            f"pick_nodes_by_stride: Could not find all desired strides {desired}. "
            f"Found: {stride_to_name}."
        )

    final_return_nodes = {}
    map_idx_to_c_label = {s_val: f"C{i+3}" for i, s_val in enumerate(sorted(list(desired)))}

    for s_val in desired:
        if s_val in stride_to_name:
            final_return_nodes[stride_to_name[s_val]] = map_idx_to_c_label[s_val]
        else:
            warnings.warn(f"Desired stride {s_val} not found by pick_nodes_by_stride.")

    return final_return_nodes


@torch.no_grad()
def _get_dynamic_feat_chs(model: nn.Module, img_size: int, device: torch.device, 
                          min_features: int = 2) -> Tuple[int, ...]:
    """Dynamically determine feature channel dimensions from backbone."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    model.to(device)
    
    features = model(dummy_input)
    
    model.to(original_device)
    
    if not isinstance(features, (list, tuple)):
        raise ValueError(
            f"Backbone expected to return list/tuple of feature maps, "
            f"got {type(features).__name__}"
        )
    
    if len(features) < min_features:
        raise ValueError(
            f"Backbone expected to return at least {min_features} feature maps, "
            f"got {len(features)}. Check backbone configuration."
        )
    
    return tuple(f.shape[1] for f in features)


def get_backbone(arch: str, ckpt: str | None, img_size: int = 224):
    pretrained = ckpt is None
    temp_device_for_init = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch_list = ["ssh_hybrid_s", "ssh_hybrid_s_bl", "ssh_hybrid_m", "ssh_hybrid_l", "conv_s", "conv_m", "conv_l", "conv_xl"]

    if arch == "mnv3":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights, width_mult=1.0)
        base.to(temp_device_for_init)

        desired_strides_tuple = (8, 16, 32)
        return_nodes = pick_nodes_by_stride(base, img_size=img_size, desired=desired_strides_tuple)
        base.cpu()

        if len(return_nodes) != len(desired_strides_tuple):
            warnings.warn(f"pick_nodes_by_stride for '{arch}' failed. Using hardcoded defaults.")
            return_nodes = {'features.3': 'C3', 'features.6': 'C4', 'features.12': 'C5'}
        
        print(f"[INFO] Using return_nodes for {arch}: {return_nodes}")

        net = TVExtractorWrapper(base, return_nodes)
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        
        if ckpt is not None and not pretrained:
            sd = torch.load(ckpt, map_location='cpu')
            missing, unexpected = base.load_state_dict(sd, strict=False)
            if missing: warnings.warn(f'Missing keys in base model ckpt ({arch}): {missing}')
            if unexpected: warnings.warn(f'Unexpected keys in base model ckpt ({arch}): {unexpected}')
            feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)

    elif arch == "mnv4c-s":
        net = MobileNetV4(
            variant='conv_s',
            width_multiplier=1.0,
            features_only=True,
            out_features_names=['p3_s8', 'p4_s16', 'p5_s32'],
        )
        feature_info = net.get_feature_info()
        feat_chs = tuple(info['num_chs'] for info in feature_info)
        print(f"[INFO] Detected feature channels: {feat_chs}")
        ckpt = f"mobilenet_w1_0_conv_s_pretrained_drp0_2_filteredimagenet2native_{img_size}px_fp32_backbone.pt"
    elif arch == "mnv4c-m":
        net = MobileNetV4(
            variant='conv_m',
            width_multiplier=1.0,
            out_features_names=['p2_s4', 'p3_s8', 'p4_s16', 'p5_s32'],
            features_only=True,
        )
        feature_info = net.get_feature_info()
        feat_chs = tuple(info['num_chs'] for info in feature_info)
        print(f"[INFO] Detected feature channels: {feat_chs}")
        ckpt = f"mobilenet_w1_0_conv_m_pretrained_drp0_2_filteredimagenet2native_{img_size}px_fp32_backbone.pt"
    elif arch in arch_list:
        net = MobileNetV4(
            variant=arch,
            width_multiplier=1.0,
            out_features_names=['p3_s8', 'p4_s16', 'p5_s32'],
            features_only=True,
        )
        feature_info = net.get_feature_info()
        feat_chs = tuple(info['num_chs'] for info in feature_info)
        print(f"[INFO] Detected feature channels: {feat_chs} for {arch}")
        ckpt = f"mobilenet_w1_0_{arch}_pretrained_drp0_2_filteredimagenet2native_{img_size}px_fp32_backbone.pt"
    elif arch == "mnv4c":
        if MobileNetV4ConvSmallPico is None:
            raise ImportError("Cannot create 'mnv4_custom' backbone. `customMobilenetNetv4.py` not found.")
        
        print("[INFO] Creating custom MobileNetV4-Small backbone for PicoDet.")
        feature_indices = (
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p3_s8'],
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p4_s16'],
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p5_s32'],
        )

        net = MobileNetV4ConvSmallPico(
            width_multiplier=1.2,
            features_only=True,
            out_features_indices=feature_indices,
        )

        ckpt = "mobilenet_w1_2_mnv4c_pretrained_drp0_2_fp32_backbone.pt"
    else:
        raise ValueError(f'Unknown arch {arch}')
    
    if arch in ["mnv4c", "mnv4c-s", "mnv4c-m"] or arch in arch_list:
        if os.path.exists(ckpt):
            print(f"[INFO] Loading pre-trained backbone weights from: {ckpt}")
            backbone_sd = torch.load(ckpt, map_location='cpu')
            missing_keys, unexpected_keys = net.load_state_dict(backbone_sd, strict=False)

            if missing_keys:
                warnings.warn(f"Warning: Missing keys when loading backbone weights: {missing_keys}")
            if unexpected_keys:
                warnings.warn(f"Warning: Unexpected keys when loading backbone weights: {unexpected_keys}")
            print("[INFO] Successfully loaded backbone weights.")
        else:
            print("[WARNING] Initializing backbone with random weights (no checkpoint provided).")
        
        # Now get the feature channel dimensions dynamically.
        # The custom backbone already returns a list of features, so no wrapper is needed.
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        print(f"[INFO] Custom MNv4 feature channels: {feat_chs}")

    net.train()
    return net, feat_chs

    
# Convenience export
__all__ = [
    'GhostConv', 'DWConv', 'CSPBlock', 'CSPPAN', 'ESE', 'SpatialAttention',
    'VarifocalLoss', 'dfl_loss', 'build_dfl_targets',
    'PicoDetHead', 'ResizeNorm', 'get_backbone', 'PicoDet',
    'QualityFocalLoss',
]
