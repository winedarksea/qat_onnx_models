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
            nn.BatchNorm2d(init_ch), nn.ReLU6(inplace=False)
        )
        self.cheap = nn.Sequential() if cheap_ch == 0 else nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_size, 1, dw_size // 2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch), nn.ReLU6(inplace=False)
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
        self.act = nn.ReLU6(inplace=False)

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



class PicoDetHead(nn.Module):
    def __init__(self, num_classes: int = 80, reg_max: int = 7, num_feats: int = 96,
                 num_levels: int = 3, max_det: int = 100,
                 score_thresh: float = 0.05, nms_iou: float = 0.6,
                 img_size: int = 224): # Add img_size here to compute H, W
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.nl = num_levels
        self.max_det = max_det
        self.score_th = score_thresh
        self.iou_th = nms_iou
        
        strides_tensor = torch.tensor([8, 16, 32][:num_levels], dtype=torch.float32)
        self.register_buffer('strides_buffer', strides_tensor, persistent=False)
        
        dfl_project_tensor = torch.arange(self.reg_max + 1, dtype=torch.float32)
        self.register_buffer('dfl_project_buffer', dfl_project_tensor, persistent=False)

        # Head convolutional layers (same as before)
        self.cls_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])
        self.reg_conv = nn.Sequential(*[GhostConv(num_feats, num_feats) for _ in range(2)])
        self.cls_pred = nn.ModuleList([nn.Conv2d(num_feats, self.nc, 1) for _ in range(self.nl)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(num_feats, 1, 1) for _ in range(self.nl)])
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(num_feats, 4 * (self.reg_max + 1), 1) for _ in range(self.nl)]
        )

        self._initialize_biases()
        
        # --- NEW: Pre-calculate and register anchor points ---
        self.anchor_points_centers_levels: nn.ModuleList[torch.Tensor] = nn.ModuleList() # Use ModuleList for buffers
        # Or just a list of buffers if you manage device placement carefully
        # self.register_buffer cannot be used in a loop directly to create a list of buffers with dynamic names
        # A common pattern is to store them as a list of tensors and then register them manually if needed,
        # or let them be part of nn.ModuleList if they need to be torch.nn.Module (not the case here)
        # For simple tensors, storing them as attributes that are tensors works if they are used.
        
        _anchor_points_centers_list = []
        for i in range(self.nl):
            s = self.strides_buffer[i].item() # Get scalar stride
            # Important: Use integer division for H, W if img_size/s might not be exact integer
            H_level = math.ceil(img_size / s) # Or int(img_size / s) if you are sure it's exact
            W_level = math.ceil(img_size / s) # Or int(img_size / s)
            
            yv, xv = torch.meshgrid(
                torch.arange(H_level, dtype=torch.float32),
                torch.arange(W_level, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack((xv, yv), dim=2).reshape(H_level * W_level, 2)
            anchor_points_center = (grid + 0.5) * s
            _anchor_points_centers_list.append(anchor_points_center)
            # Register each as a buffer
            self.register_buffer(f'anchor_points_level_{i}', anchor_points_center, persistent=False)

    def _initialize_biases(self): # (Same as before)
        cls_bias_init = -math.log((1 - 0.01) / 0.01)
        for layer_list in [self.cls_pred, self.obj_pred]:
            for layer in layer_list:
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, cls_bias_init)

    def _dfl_to_ltrb_inference(self, x_reg_logits_3d: torch.Tensor) -> torch.Tensor:
        """
        Decodes DFL regression logits to LTRB box offsets,
        SPECIALIZED FOR 3D INPUT (Batch, Num_Anchors, 4*(reg_max+1)) for inference tracing.
        """
        # Expects x_reg_logits_3d: (B, N_anchors_per_image, 4*(reg_max+1))
        b, n_anchors_img, _ = x_reg_logits_3d.shape
        
        # This assertion can be helpful during dev, but remove/comment for final tracing if it causes issues
        # assert x_reg_logits_3d.ndim == 3, f"Expected 3D input for _dfl_to_ltrb_inference, got {x_reg_logits_3d.ndim}D"

        x_reg_logits_reshaped = x_reg_logits_3d.view(b, n_anchors_img, 4, self.reg_max + 1)
        x_softmax = x_reg_logits_reshaped.softmax(dim=3) # Softmax over reg_max+1 dimension
        
        # self.dfl_project_buffer shape (M+1), view for broadcast: (1,1,1,M+1)
        proj = self.dfl_project_buffer.view(1, 1, 1, -1) 
        ltrb_offsets = (x_softmax * proj).sum(dim=3) # (B, N_anchors_per_image, 4)
        return ltrb_offsets

    # Keep the original _dfl_to_ltrb for training if needed by your loss calculation
    # Or, if your loss calculation also flattens to a consistent shape (e.g., 2D),
    # you might have another specialized version or adjust your training code.
    def _dfl_to_ltrb_original_for_training_etc(self, x_reg_logits: torch.Tensor) -> torch.Tensor:
        input_shape = x_reg_logits.shape
        if x_reg_logits.ndim == 4: # (B, C, H, W) - Not used in current inference path
            b, _, h, w = input_shape
            x_reg_logits = x_reg_logits.view(b, 4, self.reg_max + 1, h, w)
            x_softmax = x_reg_logits.softmax(dim=2)
            proj = self.dfl_project_buffer.view(1, 1, -1, 1, 1)
            ltrb_offsets = (x_softmax * proj).sum(dim=2)
        elif x_reg_logits.ndim == 2: # (N_total_anchors, 4 * (reg_max + 1)) - Might be used in training
            n_anchors = input_shape[0]
            x_reg_logits = x_reg_logits.view(n_anchors, 4, self.reg_max + 1)
            x_softmax = x_reg_logits.softmax(dim=2)
            proj = self.dfl_project_buffer.view(1, 1, -1)
            ltrb_offsets = (x_softmax * proj).sum(dim=2)
        elif x_reg_logits.ndim == 3: # (B, N_anchors_per_image, 4*(reg_max+1)) - Used in inference
            b, n_anchors_img, _ = input_shape
            x_reg_logits = x_reg_logits.view(b, n_anchors_img, 4, self.reg_max + 1)
            x_softmax = x_reg_logits.softmax(dim=3)
            proj = self.dfl_project_buffer.view(1, 1, 1, -1)
            ltrb_offsets = (x_softmax * proj).sum(dim=3)
        else:
            raise ValueError(f"_dfl_to_ltrb expects 2D, 3D or 4D input, got {x_reg_logits.ndim}D")
        return ltrb_offsets

    def _decode_predictions_for_level(self,
                                      cls_logit: torch.Tensor, obj_logit: torch.Tensor, reg_logit: torch.Tensor,
                                      level_idx: int
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H_feat, W_feat = cls_logit.shape
        stride = self.strides_buffer[level_idx]
        num_anchors_level = H_feat * W_feat

        anchor_points_center = getattr(self, f'anchor_points_level_{level_idx}')
        anchor_points_center = anchor_points_center.to(cls_logit.device)

        cls_logit_perm = cls_logit.permute(0, 2, 3, 1).reshape(B, num_anchors_level, self.nc)
        obj_logit_perm = obj_logit.permute(0, 2, 3, 1).reshape(B, num_anchors_level, 1)
        reg_logit_perm = reg_logit.permute(0, 2, 3, 1).reshape(B, num_anchors_level, 4 * (self.reg_max + 1))

        # --- Use the specialized version for inference tracing ---
        ltrb_offsets = self._dfl_to_ltrb_inference(reg_logit_perm) 
        ltrb_offsets_scaled = ltrb_offsets * stride

        ap_expanded = anchor_points_center.unsqueeze(0)
        x1 = ap_expanded[..., 0] - ltrb_offsets_scaled[..., 0]
        y1 = ap_expanded[..., 1] - ltrb_offsets_scaled[..., 1]
        x2 = ap_expanded[..., 0] + ltrb_offsets_scaled[..., 2]
        y2 = ap_expanded[..., 1] + ltrb_offsets_scaled[..., 3]
        boxes_xyxy_level = torch.stack([x1, y1, x2, y2], dim=-1)

        scores_level = (cls_logit_perm + obj_logit_perm).sigmoid()
        
        return boxes_xyxy_level, scores_level

    def _batch_nms_and_pad(self,
                           batched_boxes: torch.Tensor,    # (B, Total_Anchors_All_Levels, 4)
                           batched_scores: torch.Tensor   # (B, Total_Anchors_All_Levels, NC)
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, N_anchors, NC = batched_scores.shape
        device = batched_boxes.device

        # 1. Reshape inputs for batched_nms
        # boxes: (B * N_anchors, 4)
        # scores_flat: (B * N_anchors, NC)
        boxes_flat = batched_boxes.reshape(-1, 4)
        scores_flat = batched_scores.reshape(-1, NC)

        # 2. Create idxs tensor for batched_nms
        # This tensor indicates which batch item each box belongs to.
        # (B * N_anchors,)
        idxs_batch_membership = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, N_anchors).reshape(-1)
        
        final_batched_boxes = torch.zeros((B, self.max_det, 4), dtype=batched_boxes.dtype, device=device)
        final_batched_scores = torch.zeros((B, self.max_det), dtype=batched_scores.dtype, device=device)
        final_batched_labels = torch.full((B, self.max_det), -1, dtype=torch.long, device=device)

        # Process each class separately for NMS, then combine and pick top overall.
        # Or, more commonly for object detection, get max score per anchor and do class-agnostic NMS per image.
        # Let's assume we want to do NMS per image, considering the top score for each anchor.

        # Get the top score and corresponding class for each anchor in each batch item
        # conf_per_anchor_batched: (B, N_anchors)
        # labels_per_anchor_batched: (B, N_anchors)
        conf_per_anchor_batched, labels_per_anchor_batched = torch.max(batched_scores, dim=2)

        # Flatten these for batched_nms (or loop and do per-image NMS if batched_nms struggles with class handling here)
        # conf_flat: (B * N_anchors,)
        conf_flat = conf_per_anchor_batched.reshape(-1)
        
        # Apply score threshold *before* NMS to reduce candidates
        keep_by_score_mask = conf_flat >= self.score_th
        
        boxes_to_nms = boxes_flat[keep_by_score_mask]
        scores_to_nms = conf_flat[keep_by_score_mask]
        idxs_batch_membership_to_nms = idxs_batch_membership[keep_by_score_mask]
        # We also need to keep track of the original labels for the boxes that pass the score threshold
        original_labels_flat = labels_per_anchor_batched.reshape(-1)
        labels_to_nms = original_labels_flat[keep_by_score_mask]

        if boxes_to_nms.numel() == 0: # No boxes passed score threshold
            return final_batched_boxes, final_batched_scores, final_batched_labels

        # Apply batched NMS
        # `batched_nms` performs NMS independently for items with different `idxs`.
        # It means it will do NMS per batch item effectively.
        keep_indices_after_nms = tvops.batched_nms(
            boxes_to_nms,          # (K, 4)
            scores_to_nms,         # (K,)
            idxs_batch_membership_to_nms, # (K,)
            self.iou_th
        )
        # keep_indices_after_nms are indices into boxes_to_nms, scores_to_nms, etc.

        # Select the detections that were kept by NMS
        boxes_kept = boxes_to_nms[keep_indices_after_nms]
        scores_kept = scores_to_nms[keep_indices_after_nms]
        labels_kept = labels_to_nms[keep_indices_after_nms]
        batch_indices_kept = idxs_batch_membership_to_nms[keep_indices_after_nms]

        # Now, distribute these kept detections back into the per-batch item slots,
        # select top self.max_det for each, and pad.
        # This part is tricky because `batched_nms` output is flat.

        for b_idx in range(B):
            # Find all detections belonging to the current batch item b_idx
            detections_for_this_batch_item_mask = (batch_indices_kept == b_idx)
            
            current_boxes = boxes_kept[detections_for_this_batch_item_mask]
            current_scores = scores_kept[detections_for_this_batch_item_mask]
            current_labels = labels_kept[detections_for_this_batch_item_mask]

            # Sort by score to pick top_k (if NMS doesn't already sort, it usually does)
            # NMS output is often sorted by score, but to be safe:
            # Not strictly necessary if NMS output is sorted, but good for explicit top-k.
            # Or, batched_nms might already respect some max_detections_per_class or similar.
            # torchvision.ops.batched_nms does not have a top-k per class/group.
            # So we take all, then filter to self.max_det for each image.

            num_dets_for_batch_item = current_boxes.shape[0]
            
            if num_dets_for_batch_item > 0:
                # If we need to sort by score (NMS usually does, but let's assume we might need to re-sort or just take top)
                # For simplicity, NMS output is generally good enough to take the first N.
                # If NMS output isn't sorted by score, you'd sort here.
                # For now, assume tvops.batched_nms gives high-scoring ones first within each group.
                
                num_to_fill = min(num_dets_for_batch_item, self.max_det)
                
                final_batched_boxes[b_idx, :num_to_fill] = current_boxes[:num_to_fill]
                final_batched_scores[b_idx, :num_to_fill] = current_scores[:num_to_fill]
                final_batched_labels[b_idx, :num_to_fill] = current_labels[:num_to_fill]
        
        return final_batched_boxes, final_batched_scores, final_batched_labels

    def forward(self, neck_feature_maps: Tuple[torch.Tensor, ...]):
        # (This part is likely fine from previous fixes, assuming _decode_predictions_for_level is okay)
        raw_cls_logits_levels: List[torch.Tensor] = []
        raw_obj_logits_levels: List[torch.Tensor] = []
        raw_reg_logits_levels: List[torch.Tensor] = []

        for i, f_map_level in enumerate(neck_feature_maps):
            cls_common_feat = self.cls_conv(f_map_level)
            reg_common_feat = self.reg_conv(f_map_level)
            raw_cls_logits_levels.append(self.cls_pred[i](cls_common_feat))
            raw_obj_logits_levels.append(self.obj_pred[i](cls_common_feat))
            raw_reg_logits_levels.append(self.reg_pred[i](reg_common_feat))

        if self.training:
            return raw_cls_logits_levels, raw_obj_logits_levels, raw_reg_logits_levels
        else: # Inference path
            decoded_boxes_all_levels: List[torch.Tensor] = []
            decoded_scores_all_levels: List[torch.Tensor] = []
            
            for i in range(self.nl):
                cls_l, obj_l, reg_l = raw_cls_logits_levels[i], raw_obj_logits_levels[i], raw_reg_logits_levels[i]
                boxes_level, scores_level = self._decode_predictions_for_level(
                    cls_l, obj_l, reg_l, i
                )
                decoded_boxes_all_levels.append(boxes_level)
                decoded_scores_all_levels.append(scores_level)

            batched_all_boxes = torch.cat(decoded_boxes_all_levels, dim=1)
            batched_all_scores = torch.cat(decoded_scores_all_levels, dim=1)
            return self._batch_nms_and_pad(batched_all_boxes, batched_all_scores) # Calls the modified version

# In PicoDet class:
class PicoDet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_chs: Tuple[int, int, int],
                 num_classes: int = 80, neck_out_ch: int = 96, 
                 img_size: int = 224, # This img_size IS CRUCIAL
                 head_reg_max: int = 7, 
                 head_max_det: int = 100,
                 head_score_thresh: float = 0.05,
                 head_nms_iou: float = 0.6):
        super().__init__()
        self.pre = ResizeNorm(size=(img_size, img_size)) 
        self.backbone = backbone
        self.neck = CSPPAN(in_chs=feat_chs, out_ch=neck_out_ch)
        num_fpn_levels = len(feat_chs) 
        
        self.head = PicoDetHead(
            num_classes=num_classes, 
            reg_max=head_reg_max,
            num_feats=neck_out_ch,
            num_levels=num_fpn_levels,
            max_det=head_max_det,
            score_thresh=head_score_thresh,
            nms_iou=head_nms_iou,
            img_size=img_size # <<< --- PASS IMG_SIZE TO HEAD
        )

    def forward(self, x: torch.Tensor): # (Same as before)
        x = self.pre(x)
        backbone_features = self.backbone(x)
        c3, c4, c5 = backbone_features[0], backbone_features[1], backbone_features[2]
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))


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
        x = F.interpolate(x, self.size, mode='bilinear', align_corners=False, antialias=True) 
        # x = F.interpolate(x, self.size, mode='nearest', align_corners=False, antialias=False)  # faster

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
    
# Convenience export
__all__ = [
    'GhostConv', 'DWConv5x5', 'CSPBlock', 'CSPPAN',
    'VarifocalLoss', 'dfl_loss', 'build_dfl_targets',
    'PicoDetHead', 'ResizeNorm', 'get_backbone', 'PicoDet'
]
