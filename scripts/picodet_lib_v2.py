# picodet_lib.py
# PyTorch 2.7 / opset‑18
from __future__ import annotations
import math, warnings, sys, os
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from torchvision.models.feature_extraction import create_feature_extractor
import timm

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
class GhostConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            k: int = 1,
            s: int = 1,
            dw_size: int = 3,  # dw_size 5 is maybe a bit slower but more accurate than 3
            ratio: int = 2,  # 1.5 generally slower, maybe more accurate, 3.0 faster, maybe less accurate
            inplace_act: bool = False
         ):
        super().__init__()
        self.c_out = c_out # Store c_out
        init_ch = math.ceil(c_out / ratio)
        # Ensure init_ch is not greater than c_out, especially if c_out is small.
        init_ch = min(init_ch, c_out)
        self.cheap_ch = c_out - init_ch

        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU6(inplace=inplace_act)  # ReLU6, HardSwish
        )
        if self.cheap_ch > 0:
            self.cheap = nn.Sequential(
                nn.Conv2d(init_ch, self.cheap_ch, dw_size, 1, dw_size // 2,
                          groups=init_ch, bias=False),
                nn.BatchNorm2d(self.cheap_ch), nn.ReLU6(inplace=inplace_act)  # ReLU6, HardSwish
            )
        else:
            self.cheap = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_primary = self.primary(x)
        if self.cheap:
            y_cheap = self.cheap(y_primary)
            return torch.cat([y_primary, y_cheap], 1)
        return y_primary

    @property
    def out_channels(self):
        # utility to query channels
        return self.primary[0].out_channels + (self.cheap[0].out_channels if self.cheap else 0)


class DWConv(nn.Module):
    def __init__(
            self, c: int, k: int = 5,  # k=5 for 5x5, 3 for 3x3
            inplace_act: bool = False
    ):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=inplace_act)  # ReLU6, HardSwish

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
    def __init__(self, in_chs=(40, 112, 160), out_ch=96, lat_k=5, inplace_act: bool = False): # out_ch=64 would be faster than 96
        super().__init__()
        self.in_chs = in_chs
        self.reduce = nn.ModuleList([GhostConv(c, out_ch, 1, inplace_act=inplace_act) for c in in_chs])
        self.lat    = nn.ModuleList([DWConv(out_ch, k=lat_k, inplace_act=inplace_act) for _ in in_chs[:-1]])
        lst_ly = len(in_chs) - 1
        self.out = nn.ModuleList([
            CSPBlock(out_ch, n=2 if i == lst_ly else 1, m_k = 3 if i == lst_ly else 1, inplace_act=inplace_act) for i in range(len(in_chs))
        ])

    def forward(self, c3, c4, c5):
        # top-down ------------------------------------------------------------
        p5 = self.reduce[2](c5)
        reduced_c4 = self.reduce[1](c4)
        interpolated_p5 = F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = reduced_c4 + interpolated_p5
        p3 = self.reduce[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        p4, p3 = self.lat[1](p4), self.lat[0](p3)

        # bottom-up  ---------------------------------------------------------
        p4 = p4 + F.max_pool2d(p3, 2)
        p5 = p5 + F.max_pool2d(p4, 2)

        return (self.out[0](p3), self.out[1](p4), self.out[2](p5))

# ─────────────────────── losses (VFL · DFL · IoU) ──────────────────
class VarifocalLoss(nn.Module):
    """Implementation that expects *raw* joint logits (cls+obj)."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2., reduction: str = 'mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets_q: torch.Tensor):
        p = logits.sigmoid()
        with torch.no_grad():
            weight = torch.where(targets_q > 0,
                                  targets_q,
                                  self.alpha * p.pow(self.gamma))
        loss = F.binary_cross_entropy_with_logits(logits, targets_q, weight, reduction='none')
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss

def build_dfl_targets(offsets: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Soft distribution targets for DFL (Distilled Focal Loss)."""
    # offsets: (N,4)
    x = offsets.clamp_(0, reg_max)
    l = x.floor().long()
    r = (l + 1).clamp_(max=reg_max)
    w_r = x - l.float()
    w_l = 1. - w_r
    one_hot_l = F.one_hot(l, reg_max + 1).float() * w_l.unsqueeze(-1)
    one_hot_r = F.one_hot(r, reg_max + 1).float() * w_r.unsqueeze(-1)
    return one_hot_l + one_hot_r  # (N,4,M+1)


def dfl_loss(pred_logits, target_dist, eps: float = 1e-7):
    """
    pred_logits : (N*4, M+1)  raw logits
    target_dist : (N,4,M+1)   soft distribution from build_dfl_targets
    """
    n, _, K = target_dist.shape                          # K = M+1
    pred = pred_logits.view(n * 4, K)
    tgt  = target_dist.view(n * 4, K)
    # 1.  Clamp targets so log(·) is defined and sum ≈ 1
    tgt = tgt.clamp_min(eps)
    tgt = tgt / tgt.sum(dim=1, keepdim=True)
    # 2.  log-probabilities of the prediction
    logp = F.log_softmax(pred, dim=1)
    # 3.  KL-divergence, normalised per *box* (not per batch)
    return F.kl_div(logp, tgt, reduction='batchmean', log_target=False)


class PicoDetHead(nn.Module):
    def __init__(self, num_classes: int = 80,
                 reg_max: int = 8,
                 num_feats: int = 96,
                 num_levels: int = 3, 
                 # NMS parameters are stored for use by external NMS/ONNX appending
                 max_det: int = 100, 
                 score_thresh: float = 0.05, 
                 nms_iou: float = 0.6,
                 img_size: int = 224,
                 cls_conv_depth: int = 3,  # 2
                 inplace_act: bool = False):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.nl = num_levels
        self.max_det = max_det
        self.score_th = score_thresh
        self.iou_th = nms_iou
        self.reg_conv_depth = 2
        self.cls_conv_depth = cls_conv_depth
        self.img_size = img_size
        first_cls_conv_k = 3  # 1

        strides_tensor = torch.tensor([8, 16, 32][:num_levels], dtype=torch.float32)
        self.register_buffer('strides_buffer', strides_tensor, persistent=False)
        
        dfl_project_tensor = torch.arange(self.reg_max + 1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', dfl_project_tensor, persistent=False)

        if self.cls_conv_depth <= 1:
            self.cls_conv = nn.Sequential(
                GhostConv(num_feats, num_feats, k=first_cls_conv_k, inplace_act=inplace_act)
            )
        else:
            self.cls_conv = nn.Sequential(
                GhostConv(num_feats, num_feats, k=first_cls_conv_k, inplace_act=inplace_act),
                *[GhostConv(num_feats, num_feats, inplace_act=inplace_act) for _ in range(self.cls_conv_depth - 1)]
            )
        self.reg_conv = nn.Sequential(*[GhostConv(num_feats, num_feats, ratio=3.0, inplace_act=inplace_act) for _ in range(self.reg_conv_depth)])
        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        # self.obj_pred = nn.ModuleList([nn.Conv2d(num_feats, 1, 1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(num_feats, 4 * (self.reg_max + 1), 1) for _ in range(self.nl)]
        )
        
        # Initialized to 1.0 to match the original additive behavior at the start.
        self.logit_scale = nn.Parameter(torch.ones(1), requires_grad=True)
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
            # _anchor_points_centers_list.append(anchor_points_center) # Not strictly needed if directly registering
            self.register_buffer(f'anchor_points_level_{i}', anchor_points_center, persistent=False)

    def _initialize_biases(self) -> None:
        """
        Initialise conv-biases so that:
          • class-scores start with the common “p = 0.01” prior  (≈ –4.595)
          • objectness scores start at 0.0  (p = 0.5) so that
            the *joint* score  sigmoid(cls + obj)  also starts at 0.01
        This prevents the network from being over-confidently negative and
        makes it possible for early positive boxes to cross a 0.05 threshold.
        """
        # classification branches
        cls_prior = 0.05
        cls_bias  = -math.log((1. - cls_prior) / cls_prior)      # –4.595, use -2.19 for multiplicative sigmoid
        for conv in self.cls_pred:
            if conv.bias is not None:
                nn.init.constant_(conv.bias, cls_bias)

        # objectness branches   (neutral ⇒ bias = 0.0)
        # obj_prior = 0.1
        # obj_bias = -math.log((1 - obj_prior) / obj_prior)
        # for conv in self.obj_pred:
        #     if conv.bias is not None:
        #         nn.init.constant_(conv.bias, obj_bias)

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
        cls_logit: torch.Tensor, reg_logit: torch.Tensor,  # obj_logit: torch.Tensor, 
        level_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H_feat, W_feat = cls_logit.shape
        stride = self.strides_buffer[level_idx]

        # build grid *on the fly* so it always matches H_feat,W_feat
        yv, xv = torch.meshgrid(
            torch.arange(H_feat, device=cls_logit.device),
            torch.arange(W_feat, device=cls_logit.device),
            indexing='ij'
        )
        anchor_centers = (torch.stack((xv, yv), dim=2).view(-1, 2) + 0.5) * stride

        cls_logit_perm = cls_logit.permute(0,2,3,1).reshape(B, H_feat*W_feat, self.nc)
        # obj_logit_perm = obj_logit.permute(0,2,3,1).reshape(B, H_feat*W_feat, 1)
        reg_logit_perm = reg_logit.permute(0,2,3,1).reshape(B, H_feat*W_feat, 4*(self.reg_max+1))

        ltrb = self._dfl_to_ltrb_inference(reg_logit_perm) * stride

        x1 = anchor_centers[:,0].unsqueeze(0) - ltrb[...,0]
        y1 = anchor_centers[:,1].unsqueeze(0) - ltrb[...,1]
        x2 = anchor_centers[:,0].unsqueeze(0) + ltrb[...,2]
        y2 = anchor_centers[:,1].unsqueeze(0) + ltrb[...,3]
        boxes = torch.stack([x1,y1,x2,y2], dim=-1)

        # Use the learned scaler during inference to combine logits.
        # scores = (cls_logit_perm + self.logit_scale * obj_logit_perm).sigmoid()
        scores = cls_logit_perm.sigmoid() * self.logit_scale
        return boxes, scores

    def forward(self, neck_feature_maps: Tuple[torch.Tensor, ...]):
        raw_cls_logits_levels: List[torch.Tensor] = []
        # raw_obj_logits_levels: List[torch.Tensor] = []
        raw_reg_logits_levels: List[torch.Tensor] = []

        for i, f_map_level in enumerate(neck_feature_maps):
            cls_common_feat = self.cls_conv(f_map_level)
            reg_common_feat = self.reg_conv(f_map_level)
            raw_cls_logits_levels.append(self.cls_pred[i](cls_common_feat))
            # raw_obj_logits_levels.append(self.obj_pred[i](cls_common_feat))
            raw_reg_logits_levels.append(self.reg_pred[i](reg_common_feat))


        if self.training:
            strides_outputs_list = [self.strides_buffer[i] for i in range(self.nl)] # List[Tensor]

            return (
                tuple(raw_cls_logits_levels),
                # tuple(raw_obj_logits_levels),
                tuple(raw_reg_logits_levels),
                tuple(strides_outputs_list)
            )
        else: # Inference path
            decoded_boxes_all_levels: List[torch.Tensor] = []
            decoded_scores_all_levels: List[torch.Tensor] = []
            for i in range(self.nl):
                # cls_l, obj_l, reg_l = raw_cls_logits_levels[i], raw_obj_logits_levels[i], raw_reg_logits_levels[i]
                # boxes_level, scores_level = self._decode_predictions_for_level(cls_l, obj_l, reg_l, i)
                cls_l, reg_l = raw_cls_logits_levels[i], raw_reg_logits_levels[i]
                boxes_level, scores_level = self._decode_predictions_for_level(cls_l, reg_l, i)
                decoded_boxes_all_levels.append(boxes_level)
                decoded_scores_all_levels.append(scores_level)
            batched_all_boxes = torch.cat(decoded_boxes_all_levels, dim=1)
            batched_all_scores = torch.cat(decoded_scores_all_levels, dim=1)
            return batched_all_boxes, batched_all_scores


class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80,
                 neck_out_ch: int = 96,  # suggestion: 64 if img≤320; 96 if 320 < img≤512; 128 if larger.
                 img_size: int = 224,
                 head_reg_max: int = 8,  # suggestion: 256 px → 7/8, 320 px → 11, 640 px → 15 or 16
                 head_max_det: int = 100, # Will be used by ONNX NMS logic
                 head_score_thresh: float = 0.05, # Will be used by ONNX NMS logic
                 head_nms_iou: float = 0.6, # Will be used by ONNX NMS logic
                 cls_conv_depth: int = 3,
                 lat_k: int = 5,
                 inplace_act_for_head_neck: bool = False):
        super().__init__()
        self.pre = ResizeNorm(size=(img_size, img_size)) 
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch, lat_k=lat_k, inplace_act=inplace_act_for_head_neck)
        num_fpn_levels = len(feat_chs)
        self.debug_count = 0
        self.img_size = img_size
        
        self.head = PicoDetHead(
            num_classes=num_classes, 
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=num_fpn_levels,
            max_det=head_max_det,
            score_thresh=head_score_thresh,
            nms_iou=head_nms_iou,
            img_size=img_size,
            cls_conv_depth=cls_conv_depth,
            inplace_act=inplace_act_for_head_neck,
        )

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        backbone_features = self.backbone(x)
        c3, c4, c5 = backbone_features[0], backbone_features[1], backbone_features[2]
        p3, p4, p5 = self.neck(c3, c4, c5)
        
        if self.debug_count == 0:
            input_img_h = x.shape[2]
            print(f"[DEBUG P_STRIDES] Input H: {input_img_h}")
            print(f"[DEBUG P_STRIDES] P3 shape: {p3.shape}, Actual Stride P3: {input_img_h / p3.shape[2]}")
            print(f"[DEBUG P_STRIDES] P4 shape: {p4.shape}, Actual Stride P4: {input_img_h / p4.shape[2]}")
            print(f"[DEBUG P_STRIDES] P5 shape: {p5.shape}, Actual Stride P5: {input_img_h / p5.shape[2]}")
            print(f"[DEBUG P_STRIDES] Head expected strides: {self.head.strides_buffer.tolist()}")
            self.debug_count = 1
        return self.head((p3, p4, p5))


class ResizeNorm(nn.Module):
    def __init__(self, size: Tuple[int, int], mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        super().__init__()
        self.size = tuple(size)
        self.register_buffer('m', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):

        x_float_scaled = x.float() / 255.0

        # Interpolation, faster if antialias=False
        x_resized = F.interpolate(x_float_scaled, self.size, mode='bilinear', align_corners=False, antialias=False) # antialias=False for speed/simplicity

        # Normalization (mean/std)
        return (x_resized - self.m) / self.s


# ───────────────────────── backbone util ──────────────────────────
# Helper to wrap torchvision's create_feature_extractor to output a list
class TVExtractorWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, return_nodes_dict: dict):
        super().__init__()
        self.extractor = create_feature_extractor(base_model, return_nodes_dict)
        self.output_keys = list(return_nodes_dict.values())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Changed to Tuple
        feature_dict = self.extractor(x)
        # Assuming 3 features are always returned, otherwise adjust Tuple annotation
        return tuple(feature_dict[key] for key in self.output_keys) # Return as tuple

def pick_nodes_by_stride(model: nn.Module, img_size: int = 256, desired: Tuple[int, ...] = (8, 16, 32)) -> dict:
    tmp = {}
    # Ensure model is on the correct device for the dummy forward pass
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    hooks = [m.register_forward_hook(
        lambda mod, inp, out, n=name: tmp.setdefault(n, out.detach().clone().cpu())) # Detach, clone, move to CPU to save GPU mem
             for name, m in model.named_modules()]
    
    # Perform dummy forward pass
    model.eval()
    try:
        model(torch.randn(1, 3, img_size, img_size, device=device))
    finally:
        for h in hooks: h.remove()
    model.train()

    H_in = img_size
    stride_to_name = {}
    # Sort tmp by length of name to prefer shorter, higher-level module names if multiple have same feature shape
    # Or sort by some other heuristic if needed, e.g. depth in the network. For now, simple iteration.
    sorted_items = sorted(tmp.items(), key=lambda x: len(x[0]))

    for name, feat_tensor in sorted_items:
        if not isinstance(feat_tensor, torch.Tensor) or feat_tensor.ndim < 4: # Basic check for valid feature map
            continue
        
        # Check if feature map height is a divisor of input height
        if H_in % feat_tensor.shape[-2] == 0:
            stride = H_in // feat_tensor.shape[-2] # H and W are assumed equal after square resize
            if stride in desired and stride not in stride_to_name:
                # Basic check to avoid picking trivial layers like initial stem conv if not intended
                # This heuristic might need adjustment depending on the backbone.
                # Here, we assume a feature map is "significant" if its channel count isn't tiny.
                # For MobileNetV3-Small, stride 8 has 24 channels, stride 16 has 40, stride 32 has 576 (before final classifier)
                # The features.12 (last conv before pooling/classifier in mnv3) is what we want for C5.
                # Heuristic: pick module if it's not too shallow (e.g. name contains a certain depth)
                # or if it's a common pattern like 'features.X'
                if '.' in name: # Prefer modules with some depth in their name
                    stride_to_name[stride] = name
    
    if len(stride_to_name) != len(desired):
        warnings.warn(
            f"pick_nodes_by_stride: Could not find all desired strides {desired}. "
            f"Found: {stride_to_name}. Check backbone structure or adjust selection logic."
        )
        # Potentially return a partial map or raise an error, or fallback
        # For now, let's allow partial if user wants to debug

    # Map found strides to C3, C4, C5 based on sorted stride values
    # Example: if desired=(8,16,32), then stride 8 -> C3, 16 -> C4, 32 -> C5
    # sorted_found_strides = sorted(s for s in desired if s in stride_to_name) # Strides that were actually found
    
    # Ensure desired strides are present before trying to map them
    final_return_nodes = {}
    map_idx_to_c_label = {s_val: f"C{i+3}" for i, s_val in enumerate(sorted(list(desired)))}

    for s_val in desired:
        if s_val in stride_to_name:
            final_return_nodes[stride_to_name[s_val]] = map_idx_to_c_label[s_val]
        else:
            warnings.warn(f"Desired stride {s_val} not found by pick_nodes_by_stride.")
            # Optionally, you could add a fallback here to hardcoded values if a stride is missing.
            # e.g., if s_val == 8: final_return_nodes['fallback_name_for_s8'] = 'C3'

    return final_return_nodes


@torch.no_grad()
def _get_dynamic_feat_chs(model: nn.Module, img_size: int, device: torch.device) -> Tuple[int, int, int]:
    # ... (no change to this helper itself, but it relies on 'model' (TVExtractorWrapper or timm model) being correct)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    model.to(device)
    
    features = model(dummy_input)
    
    model.to(original_device)
    
    if not isinstance(features, (list, tuple)) or len(features) != 3:
        # If using pick_nodes_by_stride, the number of features should match len(desired_strides)
        # that were successfully found and mapped.
        num_expected_features = 3 # Assuming C3, C4, C5
        current_len = len(features) if isinstance(features, (list, tuple)) else 0
        raise ValueError(
            f"Backbone expected to return {num_expected_features} feature maps, "
            f"got {current_len} (type: {type(features)}). "
            f"This might be due to pick_nodes_by_stride not finding all required feature layers."
        )
    return tuple(f.shape[1] for f in features)


def get_backbone(arch: str, ckpt: str | None, img_size: int = 224):
    pretrained = ckpt is None
    temp_device_for_init = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch_list = ["ssh_hybrid_s", "ssh_hybrid_s_bl", "ssh_hybrid_m", "ssh_hybrid_l", "conv_s", "conv_m", "conv_l", "conv_xl"]

    if arch == "mnv3":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights, width_mult=1.0)
        base.to(temp_device_for_init) # Move base model to device for pick_nodes_by_stride

        # Use the helper function to get return_nodes
        # It's important that 'base' is the nn.Module whose layer names are being sought.
        desired_strides_tuple = (8, 16, 32)
        return_nodes = pick_nodes_by_stride(base, img_size=img_size, desired=desired_strides_tuple)
        
        base.cpu() # Move back to CPU if it was on temp_device for pick_nodes

        if len(return_nodes) != len(desired_strides_tuple):
            warnings.warn(
                f"pick_nodes_by_stride for '{arch}' did not find all {len(desired_strides_tuple)} desired strides. "
                f"Found nodes for: {list(return_nodes.values())}. Falling back to hardcoded defaults for '{arch}'."
            )
            # Fallback to original hardcoded nodes if auto-detection fails
            # (Ensure these hardcoded names are correct for the version of torchvision being used)
            return_nodes = {
                'features.3': 'C3',  # Stride 8
                'features.6': 'C4',  # Stride 16
                'features.12': 'C5', # Stride 32 (output of the last ConvBN in the 'features' sequence)
            }
        
        print(f"[INFO] Using return_nodes for {arch}: {return_nodes}")

        net = TVExtractorWrapper(base, return_nodes) # Wrap the original base model
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        
        # Checkpoint loading for mnv3 (for 'base' model)
        if ckpt is not None and not pretrained:
            sd = torch.load(ckpt, map_location='cpu')
            # Load into base model (net.extractor.model is 'base')
            missing, unexpected = base.load_state_dict(sd, strict=False)
            if missing: warnings.warn(f'Missing keys in base model ckpt ({arch}): {missing}')
            if unexpected: warnings.warn(f'Unexpected keys in base model ckpt ({arch}): {unexpected}')
            # Re-calculate feat_chs might be needed if ckpt changed arch details, though unlikely for channels
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
        ckpt = "mobilenet_w1_0_mnv4c-s_pretrained_drp0_2_fp32_backbone.pt"
    elif arch == "mnv4c-m":
        net = MobileNetV4(
            variant='conv_m',
            width_multiplier=1.0,
            out_features_names=['p2_s4', 'p3_s8', 'p4_s16', 'p5_s32'],  # might break without CSPPAN change
            features_only=True,
        )
        feature_info = net.get_feature_info()
        feat_chs = tuple(info['num_chs'] for info in feature_info)
        print(f"[INFO] Detected feature channels: {feat_chs}")
        ckpt = "mobilenet_w1_0_mnv4c-m_pretrained_drp0_2_fp32_backbone.pt"
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
        ckpt = f"mobilenet_w1_0_{arch}_pretrained_drp0_2_fp32_backbone.pt"
    elif arch == "mnv4c":  # older but still functional
        if MobileNetV4ConvSmallPico is None:
            raise ImportError("Cannot create 'mnv4_custom' backbone. `customMobilenetNetv4.py` not found or failed to import.")
        
        print("[INFO] Creating custom MobileNetV4-Small backbone for PicoDet.")
        
        # Define the feature levels we want, corresponding to strides 8, 16, 32
        # The custom model's _DEFAULT_FEATURE_INDICES makes this easy and reliable.
        feature_indices = (
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p3_s8'],  # Block index for stride 8
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p4_s16'], # Block index for stride 16
            MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p5_s32'], # Block index for stride 32
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
    'GhostConv', 'DWConv', 'CSPBlock', 'CSPPAN',
    'VarifocalLoss', 'dfl_loss', 'build_dfl_targets',
    'PicoDetHead', 'ResizeNorm', 'get_backbone', 'PicoDet'
]
