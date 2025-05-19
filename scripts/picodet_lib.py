# picodet_lib.py – model components, losses, backbone utils
# PyTorch ≥ 2.7 / opset‑18 ready
from __future__ import annotations
import math, warnings
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.ops as tvops
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ───────────────────────────── layers ──────────────────────────────
class GhostConv(nn.Module):
    """GhostConv from GhostNet with optional cheap branch."""
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1,
                 dw_size: int = 3, ratio: int = 2):
        super().__init__()
        init_ch = min(c_out, math.ceil(c_out / ratio))
        cheap_ch = c_out - init_ch
        pad = k // 2
        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_ch, k, s, pad, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU6(inplace=True)
        )
        self.cheap = nn.Sequential() if cheap_ch == 0 else nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_size, 1, dw_size // 2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch), nn.ReLU6(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        y = self.primary(x)
        if self.cheap:
            y = torch.cat([y, self.cheap(y)], 1)
        return y[:, : self.out_channels]

    @property
    def out_channels(self):
        # utility to query channels
        return self.primary[0].out_channels + (self.cheap[0].out_channels if self.cheap else 0)

class DWConv5x5(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class CSPBlock(nn.Module):
    def __init__(self, c: int, n: int = 1):
        super().__init__()
        self.cv1, self.cv2 = GhostConv(c, c // 2, 1), GhostConv(c, c // 2, 1)
        self.m = nn.Sequential(*[GhostConv(c // 2, c // 2) for _ in range(n)])
        self.cv3 = GhostConv(c, c, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ───────────────────────────── neck ────────────────────────────────
class CSPPAN(nn.Module):
    def __init__(self, in_chs=(40, 112, 160), out_ch=96):
        super().__init__()
        self.reduce = nn.ModuleList([GhostConv(c, out_ch, 1) for c in in_chs])
        self.lat    = nn.ModuleList([DWConv5x5(out_ch) for _ in in_chs[:-1]])
        self.out    = nn.ModuleList([CSPBlock(out_ch)  for _ in in_chs])

    def forward(self, c3, c4, c5):
        # top-down ------------------------------------------------------------
        p5 = self.reduce[2](c5)                               # stride 32
        # print(f"Shape of c5: {c5.shape}")
        # print(f"Shape of p5 (after reduce[2]): {p5.shape}")
        reduced_c4 = self.reduce[1](c4)
        # print(f"Shape of c4: {c4.shape}")
        # print(f"Shape of reduced_c4 (self.reduce[1](c4)): {reduced_c4.shape}")
        interpolated_p5 = F.interpolate(p5, scale_factor=2, mode='nearest')
        # print(f"Shape of interpolated_p5: {interpolated_p5.shape}")
        # p4 = self.reduce[1](c4) + F.interpolate(p5, 2, mode='nearest')  # s16
        p4 = reduced_c4 + interpolated_p5
        # p3 = self.reduce[0](c3) + F.interpolate(p4, 2, mode='nearest')  # s8
        p3 = self.reduce[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        p4, p3 = self.lat[1](p4), self.lat[0](p3)

        # bottom-up  ---------------------------------------------------------
        p4 = p4 + F.max_pool2d(p3, 2)   # 28→14 or 56→28 etc.
        p5 = p5 + F.max_pool2d(p4, 2)   # 14→7  or 28→14 etc.

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

def dfl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred: (N*4, M+1) logits ; target: (N,4,M+1)
    n, _, m1 = target.shape
    pred = pred.view(n * 4, m1)
    target = target.view(n * 4, m1)
    loss = F.kl_div(F.log_softmax(pred, dim=1), target, reduction='batchmean')
    return loss

# ───────────────────────── detection head ──────────────────────────
class PicoDetHead(nn.Module):
    def __init__(self, num_classes: int = 80, reg_max: int = 7, num_feats: int = 96,
                 num_levels: int = 3, max_det: int = 100,
                 score_thresh: float = 0.3, nms_iou: float = 0.5):
        super().__init__()
        self.nc, self.reg_max, self.nl = num_classes, reg_max, num_levels
        self.max_det, self.score_th, self.iou_th = max_det, score_thresh, nms_iou
        self.strides = (8, 16, 32)

        self.cls_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])
        self.reg_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])

        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(num_feats, 1, 1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(num_feats, 4 * (reg_max + 1), 1) for _ in range(self.nl)])

        # bias init (PicoDet / YOLOX style)
        cls_bias, obj_bias = -4.6, -6.0
        for cp in self.cls_pred:
            nn.init.constant_(cp.bias, cls_bias)
        for op in self.obj_pred:
            nn.init.constant_(op.bias, obj_bias)

    # -------------------------- helpers --------------------------
    def _dfl_to_ltrb(self, x: torch.Tensor):
        # x input shape: (B, 4*(self.reg_max+1), H, W)
        # e.g. (251, 32, 1, 1)
        B, C_in, H, W = x.shape 
        M = self.reg_max
        
        # Expected C_in is 4 * (M+1)
        # print(f"[_dfl_to_ltrb DEBUG] Input x shape: {x.shape}, B={B}, C_in={C_in}, H={H}, W={W}, M={M}")

        # Reshape to (B, 4, M+1, H, W)
        x_reshaped = x.view(B, 4, M + 1, H, W)
        # print(f"[_dfl_to_ltrb DEBUG] x_reshaped shape: {x_reshaped.shape}")

        x_softmax = x_reshaped.softmax(dim=2) # Softmax over M+1 dimension
        # print(f"[_dfl_to_ltrb DEBUG] x_softmax shape: {x_softmax.shape}")

        proj = torch.arange(M + 1, device=x.device, dtype=x.dtype) # Shape (M+1,)
        # print(f"[_dfl_to_ltrb DEBUG] proj shape: {proj.shape}, values: {proj}")

        # Explicitly reshape proj for broadcasting
        # Target shape for proj to broadcast with x_softmax: (1, 1, M+1, 1, 1)
        proj_reshaped = proj.view(1, 1, M + 1, 1, 1)
        # print(f"[_dfl_to_ltrb DEBUG] proj_reshaped shape: {proj_reshaped.shape}")
        
        # Multiply and sum
        # (B, 4, M+1, H, W) * (1, 1, M+1, 1, 1) -> (B, 4, M+1, H, W)
        weighted_sum_components = x_softmax * proj_reshaped
        # print(f"[_dfl_to_ltrb DEBUG] weighted_sum_components shape: {weighted_sum_components.shape}")

        # Sum over the M+1 dimension (dim=2)
        # (B, 4, M+1, H, W) -> sum(dim=2) -> (B, 4, H, W)
        out = weighted_sum_components.sum(dim=2)
        # print(f"[_dfl_to_ltrb DEBUG] Output out shape: {out.shape}, numel: {out.numel()}")
        
        return out

    # ------------------------- inference -------------------------
    def _inference_single(self, feats_out: Tuple[torch.Tensor, ...]):
        cls_ls, obj_ls, reg_ls = feats_out
        boxes, scores, labels = [], [], []
        for lv, (cl, ob, rg) in enumerate(zip(cls_ls, obj_ls, reg_ls)):
            s = self.strides[lv]
            B, C, H, W = cl.shape
            yv, xv = torch.meshgrid(torch.arange(H, device=cl.device),
                                    torch.arange(W, device=cl.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, H, W, 2).float() * s + s * 0.5
            ltrb = self._dfl_to_ltrb(rg) * s  # (B,4,H,W)
            # xyxy
            xyxy = torch.stack((grid[..., 0] - ltrb[:, 0], grid[..., 1] - ltrb[:, 1],
                                grid[..., 0] + ltrb[:, 2], grid[..., 1] + ltrb[:, 3]), -1)  # (B,H,W,4)
            boxes.append(xyxy.view(B, -1, 4))
            # joint logits
            joint = cl + ob  # broadcasting (B,C,H,W) + (B,1,H,W)
            scores.append(joint.sigmoid().permute(0, 2, 3, 1).reshape(B, -1, self.nc))
        boxes = torch.cat(boxes, 1)
        scores = torch.cat(scores, 1)

        b_out, s_out, l_out = [], [], []
        for b in range(boxes.size(0)):
            bx, sc = boxes[b], scores[b]
            conf, cls = sc.max(1)
            keep_mask = conf > self.score_th
            if keep_mask.sum() == 0:
                # pad empty
                b_out.append(torch.zeros((self.max_det, 4), device=bx.device))
                s_out.append(torch.zeros((self.max_det,), device=bx.device))
                l_out.append(torch.full((self.max_det,), -1, device=bx.device, dtype=torch.long))
                continue
            bx, conf, cls = bx[keep_mask], conf[keep_mask], cls[keep_mask]
            # NMS – torchvision.ops.nms exports to onnx::NonMaxSuppression.
            keep = tvops.nms(bx, conf, self.iou_th)[: self.max_det]
            sel_b, sel_s, sel_l = bx[keep], conf[keep], cls[keep]
            pad = self.max_det - sel_b.shape[0]
            b_out.append(F.pad(sel_b, (0, 0, 0, pad)))
            s_out.append(F.pad(sel_s, (0, pad)))
            l_out.append(F.pad(sel_l, (0, pad), value=-1))
        return torch.stack(b_out), torch.stack(s_out), torch.stack(l_out)

    # --------------------------- fwd -----------------------------
    def forward(self, feats: Tuple[torch.Tensor, ...]):
        cls_ls, obj_ls, reg_ls = [], [], []
        for i, f in enumerate(feats):
            c = self.cls_conv(f)
            r = self.reg_conv(f)
            cls_ls.append(self.cls_pred[i](c))
            obj_ls.append(self.obj_pred[i](c))
            reg_ls.append(self.reg_pred[i](r))
        if self.training:
            return cls_ls, obj_ls, reg_ls
        return self._inference_single((cls_ls, obj_ls, reg_ls))

# ──────────────────────── preprocess wrapper ──────────────────────
class ResizeNormOLD(nn.Module):
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.size = size
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('m', mean, persistent=False)
        self.register_buffer('s', std, persistent=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-2:] != self.size:
            x = F.interpolate(x, self.size, mode='bilinear', align_corners=False)
        return (x.float() / 255. - self.m) / self.s

class ResizeNorm(nn.Module):
    def __init__(self, size: Tuple[int, int], mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        super().__init__()
        self.size = tuple(size) # e.g., (IMG_SIZE, IMG_SIZE)
        # Register m and s as buffers so they are moved to device with the model
        # and included in state_dict.
        self.register_buffer('m', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        # FX-friendly: always interpolate to the target self.size.
        # If input x is already self.size, F.interpolate is often a no-op or very fast.
        # Using antialias=True is generally recommended for better image quality during resizing.
        # It might require a newer PyTorch version. If it causes issues, you can remove it.
        x = F.interpolate(x, self.size, mode='bilinear', align_corners=False, antialias=True) 
        
        # Normalization
        return (x.float() / 255.0 - self.m) / self.s

# ───────────────────────── backbone util ──────────────────────────
from torchvision.models.feature_extraction import create_feature_extractor
import timm

@torch.no_grad()
def _dummy_out_chs(model: nn.Module, feat_nodes: List[str]):
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    return tuple(out[n].shape[1] for n in feat_nodes)

def get_backbone_old(arch: str = 'mnv3', ckpt: str | None = None, width: float = 1.0, *, pretrained: bool = True):
    """Return backbone "features‑only" module + (C3, C4, C5) channel dims."""
    if arch == 'mnv3':
        weights = mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None, width_mult=width).state_dict()
        base = mobilenet_v3_small(weights=None, width_mult=width)
        if pretrained:
            base.load_state_dict(weights)
        return_nodes = {
            'features.3': 'C3',  # stride 8
            'features.6': 'C4',  # stride 16
            'features.12': 'C5'  # stride 32
        }
        extractor = create_feature_extractor(base, return_nodes)
        chs = _dummy_out_chs(extractor, list(return_nodes.values()))
        backbone = extractor
    elif arch in {'mnv4s', 'mnv4m'}:
        name = {'mnv4s': 'mobilenetv4_conv_small', 'mnv4m': 'mobilenetv4_conv_medium'}[arch]
        backbone = timm.create_model(name, pretrained=pretrained, features_only=True,
                                     out_indices=(2, 4, 6))  # layers whose reductions are 8/16/32
        chs = tuple(backbone.feature_info.channels())
    else:
        raise ValueError(f'unknown arch {arch}')

    if ckpt is not None:
        sd = torch.load(ckpt, map_location='cpu')
        miss, unexp = backbone.load_state_dict(sd, strict=False)
        if miss:
            warnings.warn(f'Missing keys in ckpt: {miss}')
        if unexp:
            warnings.warn(f'Unexpected keys in ckpt: {unexp}')
    return backbone, chs

def get_backbone_older(arch: str, ckpt: str | None):
    pretrained = ckpt is None
    if arch == "mnv3":
        from torchvision import models as tvm
        base = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            width_mult=1.0,
            dropout=0.0,
        )
        class V3Backbone(torch.nn.Module):
            """Return C3, C4, C5 (28×28, 14×14, 7×7) from mobilenet_v3_small."""
            idxs = {3, 6, 12}                     # ← the correct stride 8/16/32 layers
             
            def __init__(self, mdl):
                super().__init__(); self.features = mdl.features
             
            def forward(self, x):
                outs = []
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if i in self.idxs:
                        outs.append(x)
                return outs                      # [c3, c4, c5]
         
        net = V3Backbone(base)
        feat_chs = (24, 40, 576)
    
    else:  # MobileNet-V4 Conv-Small / Medium
        import timm
        name = {
            "mnv4s": "mobilenetv4_conv_small",
            "mnv4m": "mobilenetv4_conv_medium"
        }[arch]
        base = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            features_only=True,
            out_indices=(2, 3, 4),              # stride 8/16/32
        )

        class TimmBackbone(torch.nn.Module):
            def __init__(self, mdl): super().__init__(); self.mdl = mdl
            def forward(self, x): return self.mdl(x)        # already returns list
        net = TimmBackbone(base)
        feat_chs = tuple(m["num_chs"] for m in base.feature_info)
    
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    
    return net, feat_chs


# ───────────────────────── full detector ─────────────────────────
class PicoDetOLD(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80):
        super().__init__()
        self.pre = ResizeNorm()
        self.backbone = backbone
        self.neck = CSPPAN(feat_chs)
        self.head = PicoDetHead(num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        feats_out = self.backbone(x)  # returns dict (mnv3 extractor) or list(tuple) (timm)
        if isinstance(feats_out, dict):
            c3, c4, c5 = feats_out['C3'], feats_out['C4'], feats_out['C5']
        else:
            c3, c4, c5 = feats_out  # list[Tensor]
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))

###########

# picodet_lib.py modifications:


# Helper to wrap torchvision's create_feature_extractor to output a list
class TVExtractorWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, return_nodes_dict: dict):
        super().__init__()
        self.extractor = create_feature_extractor(base_model, return_nodes_dict)
        # Ensure order of features C3, C4, C5
        # Assumes return_nodes_dict is ordered or keys are named to reflect C3, C4, C5
        # For the specific keys used, this will be ['C3', 'C4', 'C5']
        self.output_keys = list(return_nodes_dict.values())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_dict = self.extractor(x)
        return [feature_dict[key] for key in self.output_keys]

@torch.no_grad()
def _get_dynamic_feat_chs(model: nn.Module, img_size: int, device: torch.device) -> Tuple[int, int, int]:
    """Helper to get output channels from a model that returns a list/tuple of 3 feature maps."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    # If model is not on device yet, move it temporarily
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu' # Handle no params case
    model.to(device)
    
    features = model(dummy_input)
    
    model.to(original_device) # Move back
    model.train() # Set back to train mode
    
    if not isinstance(features, (list, tuple)) or len(features) != 3:
        raise ValueError(f"Backbone expected to return 3 feature maps, got {len(features) if isinstance(features, (list,tuple)) else type(features)}")
    return tuple(f.shape[1] for f in features)

def get_backbone(arch: str, ckpt: str | None, img_size: int = 224): # Added img_size for dummy input
    pretrained = ckpt is None
    # Use a temporary device for dummy forward pass if model is on CPU initially
    # to avoid issues if _get_dynamic_feat_chs needs CUDA for some ops within model
    temp_device_for_init = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == "mnv3":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        # Ensure dropout is 0.0 as per your original get_backbone intent
        # MobileNetV3 constructor doesn't take dropout, but we can set it if default is non-zero for eval
        base = mobilenet_v3_small(weights=weights, width_mult=1.0)
        # Default dropout in mobilenet_v3_small is 0.2. Set to 0 for detection.
        # The dropout layer is typically the last layer before classifier in `base.classifier[-1]`
        # or within the feature blocks if `stochastic_depth_prob` is used (not for mnv3 dropout).
        # For MobileNetV3, dropout is usually in the classifier head, which we are bypassing.
        # If 'width_mult' affects internal dropout rates, this is handled by torchvision.
        # The main concern is `base.dropout` if it existed at top level, or in classifier.
        # Since we use `create_feature_extractor` on `base.features`, classifier dropout is not an issue.

        # Node names from your `get_backbone_old`
        return_nodes = {
            'features.3': 'C3',  # Stride 8
            'features.6': 'C4',  # Stride 16
            'features.12': 'C5', # Stride 32 (output of the last ConvBNReLU in the 'features' sequence)
        }
        # Wrap the feature extractor
        net = TVExtractorWrapper(base, return_nodes)
        # Dynamically get feature channels
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
        
        # To load a checkpoint for the base MobileNetV3 model (not the wrapped one):
        if ckpt is not None and pretrained is False: # only load if not using default imagenet pretrained
            sd = torch.load(ckpt, map_location='cpu')
            # The state_dict will be for `base`, so load into `net.extractor.model` (which is `base`)
            missing, unexpected = base.load_state_dict(sd, strict=False)
            if missing: warnings.warn(f'Missing keys in base model ckpt: {missing}')
            if unexpected: warnings.warn(f'Unexpected keys in base model ckpt: {unexpected}')
            # Re-calculate feat_chs if model changed, though typically channels are fixed by arch
            feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)

    elif arch in {'mnv4s', 'mnv4m'}:
        name_map = {
            'mnv4s': 'mobilenetv4_conv_small.pyt_in1k', # Using a specific pretrained tag
            'mnv4m': 'mobilenetv4_conv_medium.pyt_in1k'
        }
        model_name = name_map[arch]
        
        # Using out_indices from your `get_backbone_old` as they might be more tested
        # These indices should correspond to features with strides 8, 16, 32
        # For many timm models, stages producing these strides are 2,3,4 or similar.
        # (2,4,6) suggests MobileNetV4 might have more feature levels reported by feature_info.
        # It's CRITICAL to verify these indices for the specific timm model version.
        # A common pattern is `out_indices = [i for i, r in enumerate(model.feature_info.reduction()) if r in {8,16,32}]`
        # For now, let's use (2,3,4) as it's more standard for FPNs tapping S8,S16,S32 from 5-stage backbones.
        # If errors persist with mnv4, (2,4,6) or dynamic index finding would be the next step.
        
        # Create a temporary model to get correct indices for strides 8, 16, 32
        temp_model = timm.create_model(model_name, pretrained=False, features_only=True) # Get all stages
        desired_strides = {8, 16, 32}
        actual_reductions = temp_model.feature_info.reduction()
        out_indices_mnv4 = [i for i, r in enumerate(actual_reductions) if r in desired_strides]
        
        if len(out_indices_mnv4) != 3:
            warnings.warn(f"Could not find exactly 3 features for strides 8,16,32 in {model_name}. "
                          f"Found indices {out_indices_mnv4} for reductions {actual_reductions}. "
                          f"Defaulting to (2,3,4) or previously used (2,4,6) if this seems off.")
            # Fallback to a common default or your previously used one if dynamic check fails badly
            out_indices_mnv4 = (2,3,4) # Or (2,4,6) if you had better success with it

        del temp_model

        net = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices_mnv4,
            num_classes=0,      # Not used for features_only
            drop_rate=0.0,      # Standardize for detection
            drop_path_rate=0.0  # Standardize for detection
        )
        # feat_chs = tuple(net.feature_info.channels()) # Channels for the selected out_indices
        # More robust: get from a dummy forward pass of the configured `net`
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)


        # Load checkpoint for the timm model directly
        if ckpt is not None and pretrained is False:
            sd = torch.load(ckpt, map_location='cpu')
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing: warnings.warn(f'Missing keys in timm model ckpt: {missing}')
            if unexpected: warnings.warn(f'Unexpected keys in timm model ckpt: {unexpected}')
            feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
    else:
        raise ValueError(f'Unknown arch {arch}')

    # If ckpt is for a fully wrapped model (e.g. saved PicoDet.backbone)
    # and not for the base model (handled for mnv3 above, timm direct)
    # This part of checkpoint loading is tricky: is it for base or wrapped?
    # The training script implies `ckpt` might be for the `get_backbone` output.
    # The mnv3 logic above loads a *base* model checkpoint.
    # If `ckpt` is intended for the `net` object itself (already wrapped/configured):
    if ckpt is not None and arch == "mnv3" and pretrained is True: # If imagenet was loaded, but we have a specific wrapped ckpt
        # This case is if `ckpt` is for a `TVExtractorWrapper` saved state
        sd = torch.load(ckpt, map_location='cpu')
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing: warnings.warn(f'Missing keys in wrapped mnv3 ckpt: {missing}')
        if unexpected: warnings.warn(f'Unexpected keys in wrapped mnv3 ckpt: {unexpected}')
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)
    elif ckpt is not None and arch != "mnv3" and pretrained is True: # If imagenet was loaded for timm, but specific ckpt for net
        sd = torch.load(ckpt, map_location='cpu')
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing: warnings.warn(f'Missing keys in timm net ckpt: {missing}')
        if unexpected: warnings.warn(f'Unexpected keys in timm net ckpt: {unexpected}')
        feat_chs = _get_dynamic_feat_chs(net, img_size, temp_device_for_init)


    net.train() # Ensure model is in train mode by default after potential .eval() in helper
    return net, feat_chs

# In PicoDet class forward method:
class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80, neck_out_ch: int = 96, IMG_SIZE: int = 224): # Added neck_out_ch
        super().__init__()
        self.pre = ResizeNorm((IMG_SIZE, IMG_SIZE)) # Ensure IMG_SIZE is accessible or passed
        self.backbone = backbone
        # Pass the dynamic feat_chs and the desired neck output channels
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch)
        self.head = PicoDetHead(num_classes=num_classes, num_feats=neck_out_ch) # Head takes neck's out_ch

    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        # Backbone is now expected to return a list/tuple [c3, c4, c5]
        # The elements are tensors: (B, CH_c3, H/8, W/8), (B, CH_c4, H/16, W/16), (B, CH_c5, H/32, W/32)
        c3, c4, c5 = self.backbone(x)
        # Neck processes these features
        # Input to neck: c3, c4, c5
        # Output from neck: p3, p4, p5 (these will have `neck_out_ch` channels)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))
    
# Convenience export
__all__ = [
    'GhostConv', 'DWConv5x5', 'CSPBlock', 'CSPPAN',
    'VarifocalLoss', 'dfl_loss', 'build_dfl_targets',
    'PicoDetHead', 'ResizeNorm', 'get_backbone', 'PicoDet'
]
