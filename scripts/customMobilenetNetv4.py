import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type, Union, Dict

try:
    from timm.models.layers import DropPath
except ImportError:
    # Fallback DropPath if timm is not available
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor.div_(keep_prob)
            return x * random_tensor

        def extra_repr(self) -> str:
            return f'drop_prob={round(self.drop_prob,3):0.3f}'


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act_layer: Optional[Type[nn.Module]] = nn.ReLU,
        bn_layer: Type[nn.Module] = nn.BatchNorm2d,
        padding: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias
        )
        self.bn = bn_layer(out_channels)
        self.act = act_layer(inplace=False) if act_layer is nn.ReLU else act_layer() if act_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MNV4_FusedInvertedBottleneck(nn.Module):
    """Fused Inverted Bottleneck block for MobileNetV4."""
    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size, stride, act_layer, bn_layer, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv_fused = ConvBNAct(in_channels, expanded_channels, kernel_size=kernel_size, stride=stride, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_proj = ConvBNAct(expanded_channels, out_channels, kernel_size=1, stride=1, act_layer=None, bn_layer=bn_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_fused(x)
        x = self.conv_proj(x)
        if self.use_residual:
            x = self.drop_path(x) + shortcut
        return x


class MNV4_InvertedBottleneck(nn.Module):
    """Inverted Bottleneck block (IB) for MobileNetV4."""
    def __init__(self, in_channels, out_channels, expanded_channels, dw_kernel_size, stride, act_layer, bn_layer, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv_pw_exp = ConvBNAct(in_channels, expanded_channels, kernel_size=1, stride=1, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_dw = ConvBNAct(expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=stride, groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_pw_proj = ConvBNAct(expanded_channels, out_channels, kernel_size=1, stride=1, act_layer=None, bn_layer=bn_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_pw_exp(x)
        x = self.conv_dw(x)
        x = self.conv_pw_proj(x)
        if self.use_residual:
            x = self.drop_path(x) + shortcut
        return x


class MNV4_ExtraDepthwiseBlock(nn.Module):
    """Extra Depthwise block (ExtraDW) for MobileNetV4."""
    def __init__(self, in_channels, out_channels, expanded_channels, dw_kernel_size1, dw_kernel_size2, stride, act_layer, bn_layer, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv_pw_exp = ConvBNAct(in_channels, expanded_channels, kernel_size=1, stride=1, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_dw1 = ConvBNAct(expanded_channels, expanded_channels, kernel_size=dw_kernel_size1, stride=stride, groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_dw2 = ConvBNAct(expanded_channels, expanded_channels, kernel_size=dw_kernel_size2, stride=1, groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_pw_proj = ConvBNAct(expanded_channels, out_channels, kernel_size=1, stride=1, act_layer=None, bn_layer=bn_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_pw_exp(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_pw_proj(x)
        if self.use_residual:
            x = self.drop_path(x) + shortcut
        return x


class MNV4_ConvNeXtLikeBlock(nn.Module):
    """ConvNeXt-like block for MobileNetV4."""
    def __init__(self, in_channels, out_channels, expanded_channels, dw_kernel_size, stride, act_layer, bn_layer, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv_dw = ConvBNAct(in_channels, in_channels, kernel_size=dw_kernel_size, stride=stride, groups=in_channels, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_pw_exp = ConvBNAct(in_channels, expanded_channels, kernel_size=1, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_pw_proj = ConvBNAct(expanded_channels, out_channels, kernel_size=1, act_layer=None, bn_layer=bn_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        x_mlp = self.conv_pw_exp(x)
        x_mlp = self.conv_pw_proj(x_mlp)
        if self.use_residual:
            return self.drop_path(x_mlp) + shortcut
        return x_mlp


class MNV4_FFNBlock(nn.Module):
    """Feed Forward Network block (1x1 convs) for MobileNetV4."""
    def __init__(self, in_channels, out_channels, expanded_channels, stride, act_layer, bn_layer, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        # This block does not apply stride. Stride > 1 would need a downsampling projection on the shortcut.
        # As per the table, stride is always 1 for FFN blocks.
        assert stride == 1, "FFN block only supports stride 1"
        
        self.conv_pw_exp = ConvBNAct(in_channels, expanded_channels, kernel_size=1, act_layer=act_layer, bn_layer=bn_layer)
        self.conv_pw_proj = ConvBNAct(expanded_channels, out_channels, kernel_size=1, act_layer=None, bn_layer=bn_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_pw_exp(x)
        x = self.conv_pw_proj(x)
        if self.use_residual:
            x = self.drop_path(x) + shortcut
        return x


class SSM2D(nn.Module):
    """
    Hardware-Friendly 2D-Selective-Scan module.

    This module replaces the Py-loop-based scan with a parallel,
    convolutional implementation suitable for NPUs. It operates entirely
    on NCHW tensors.
    """
    def __init__(self, in_channels, d_state=16, d_conv=3, expand=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.d_inner = int(expand * in_channels)
        self.d_state = d_state
        self.d_conv = d_conv

        # 1. Input Projection
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.d_inner, kernel_size=1),
            nn.BatchNorm2d(self.d_inner),
        )

        # 2. 2D Selective Scan (implemented as four parallel 1D causal convolutions)
        # We use separate convolutions for each of the 4 directions
        self.conv_h_fwd = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=(1, d_conv), padding=(0, d_conv - 1),
            groups=self.d_inner, bias=True
        )
        self.conv_h_bwd = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=(1, d_conv), padding=(0, d_conv - 1),
            groups=self.d_inner, bias=True
        )
        self.conv_v_fwd = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=(d_conv, 1), padding=(d_conv - 1, 0),
            groups=self.d_inner, bias=True
        )
        self.conv_v_bwd = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=(d_conv, 1), padding=(d_conv - 1, 0),
            groups=self.d_inner, bias=True
        )

        # 3. Gating and Merging
        self.gate = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1)
        
        # 4. Output Projection
        self.out_proj = nn.Conv2d(self.d_inner, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
    
        x_proj = F.silu(self.in_proj(x))
    
        # horizontal scans
        h_fwd = self.conv_h_fwd(x_proj)[..., :W]                        # (B,C,H,W)
        h_bwd = torch.flip(self.conv_h_bwd(torch.flip(x_proj, [3])),   # flip-→conv-→flip
                           [3])[..., :W]
    
        # vertical scans
        v_fwd = self.conv_v_fwd(x_proj)[..., :H, :]                     # (B,C,H,W)
        v_bwd = torch.flip(self.conv_v_bwd(torch.flip(x_proj, [2])),
                           [2])[..., :H, :]
    
        scan_result  = h_fwd + h_bwd + v_fwd + v_bwd                   # now shapes match
        gated_result = scan_result * F.silu(self.gate(x_proj))
        return self.out_proj(gated_result)


class MNV4_SSHybridBlock(nn.Module):
    """
    Revised State Space Hybrid Block for MobileNetV4.
    Uses NPU-friendly components (Conv2d, BatchNorm2d) and operates on NCHW.
    """
    def __init__(self, in_channels, out_channels, stride, drop_path_rate=0.0, **kwargs):
        super().__init__()
        assert stride == 1, "SSHybridBlock currently only supports stride 1"
        self.use_residual = in_channels == out_channels
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Use BatchNorm2d for normalization, which is NPU-friendly
        # self.norm = nn.BatchNorm2d(in_channels)  # removed as there is an inner BatchNorm2d in SSM2D
        
        # The core State Space Model, now fully convolutional
        self.ssm = SSM2D(in_channels=in_channels, **kwargs)

        # Pointwise convolution to adjust channels if needed (rarely, as per spec)
        self.proj_conv = ConvBNAct(in_channels, out_channels, kernel_size=1, act_layer=None) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        shortcut = x
        
        x = self.norm(x)
        x = self.ssm(x)
        x = self.proj_conv(x) # Project if needed
        
        if self.use_residual:
            x = self.drop_path(x) + shortcut
            
        return x


class MobileNetV4ConvSmallPico(nn.Module):
    """
    Custom MobileNetV4-Conv-Small backbone.
    Designed for QAT, INT8 ONNX export, and PicoDet.
    Allows width multiplier adjustment.
    """
    # block_type, dw_k1, dw_k2, exp_ch_abs, out_ch_abs, stride
    _MNV4_CONV_S_BLOCK_SPECS = [
        ["FusedIB", None, 3, 32, 32, 2],
        ["FusedIB", None, 3, 96, 64, 2],    # Stride 8 out, index 1
        ["ExtraDW", 5, 5, 192, 96, 2],      # Stride 16 out, index 2
        ["IB", None, 3, 192, 96, 1],
        ["IB", None, 3, 192, 96, 1],
        ["ConvNext", 3, None, 384, 96, 1],
        ["ConvNext", 3, None, 192, 96, 1],
        ["ConvNext", 3, None, 192, 96, 1],
        ["ExtraDW", 3, 3, 576, 128, 2],     # Stride 32 out, index 8
        ["ExtraDW", 5, 5, 512, 128, 1],
        ["IB", None, 5, 512, 128, 1],
        ["IB", None, 5, 384, 128, 1],
        ["IB", None, 3, 512, 128, 1],
        ["IB", None, 3, 512, 128, 1],
    ]

    # Indices in _MNV4_CONV_S_BLOCK_SPECS that are typical feature extraction points
    # These are 0-indexed based on the `self.blocks` list.
    _DEFAULT_FEATURE_INDICES: Dict[str, int] = {
        'p3_s8': 1,    # Output of block 1 (stride 8)
        'p4_s16': 2,   # Output of block 2 (stride 16)
        'p5_s32': 8,   # Output of block 8 (stride 32)
    }

    def __init__(
        self,
        width_multiplier: float = 1.0,
        num_classes: int = 1000,
        out_features_names: Optional[List[str]] = None, # e.g. ['p3_s8', 'p4_s16', 'p5_s32']
        out_features_indices: Optional[Tuple[int, ...]] = None, # e.g. (1, 2, 8) for block indices
        features_only: bool = False,
        drop_rate: float = 0.0, # Dropout for classifier head
        drop_path_rate: float = 0.0, # Stochastic depth for blocks
        act_layer: Type[nn.Module] = nn.ReLU,
        bn_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.features_only = features_only if num_classes > 0 else True # Force features_only if no classifier
        self.act_layer = act_layer
        self.bn_layer = bn_layer
        self.width_multiplier = width_multiplier

        # Determine which features to output
        self.return_features_indices: Optional[Tuple[int, ...]] = None
        if out_features_indices:
            self.return_features_indices = tuple(sorted(list(set(out_features_indices))))
        elif out_features_names:
            self.return_features_indices = tuple(sorted(list(set(
                self._DEFAULT_FEATURE_INDICES[name] for name in out_features_names
                if name in self._DEFAULT_FEATURE_INDICES
            ))))
        
        if self.return_features_indices and not self.features_only:
            print("Warning: `out_features_indices` or `out_features_names` is set, but `features_only` is False. "
                  "The model will return classification logits. To get feature maps, set `features_only=True`.")


        # Stem convolution (from Table 11)
        stem_out_channels = _make_divisible(32 * width_multiplier, 8)
        self.stem = ConvBNAct(
            3, stem_out_channels, kernel_size=3, stride=2,
            act_layer=act_layer, bn_layer=bn_layer
        )
        current_channels = stem_out_channels

        # Build blocks
        self.blocks = nn.ModuleList()
        num_blocks_total = len(self._MNV4_CONV_S_BLOCK_SPECS)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks_total)]

        for i, spec in enumerate(self._MNV4_CONV_S_BLOCK_SPECS):
            block_type, dw_k1, dw_k2, exp_ch_abs, out_ch_abs, stride = spec

            exp_channels = _make_divisible(exp_ch_abs * width_multiplier, 8)
            out_channels = _make_divisible(out_ch_abs * width_multiplier, 8)
            block_drop_path_rate = dpr[i]

            if block_type == "FusedIB":
                block = MNV4_FusedInvertedBottleneck(
                    current_channels, out_channels, exp_channels,
                    kernel_size=dw_k2, stride=stride, act_layer=act_layer, bn_layer=bn_layer,
                    drop_path_rate=block_drop_path_rate
                )
            elif block_type == "IB":
                block = MNV4_InvertedBottleneck(
                    current_channels, out_channels, exp_channels,
                    dw_kernel_size=dw_k2, stride=stride, act_layer=act_layer, bn_layer=bn_layer,
                    drop_path_rate=block_drop_path_rate
                )
            elif block_type == "ExtraDW":
                block = MNV4_ExtraDepthwiseBlock(
                    current_channels, out_channels, exp_channels,
                    dw_kernel_size1=dw_k1, dw_kernel_size2=dw_k2, stride=stride,
                    act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=block_drop_path_rate
                )
            elif block_type == "ConvNext":
                block = MNV4_ConvNeXtLikeBlock(
                    current_channels, out_channels, exp_channels,
                    dw_kernel_size=dw_k1, stride=stride, act_layer=act_layer, bn_layer=bn_layer,
                    drop_path_rate=block_drop_path_rate
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            
            self.blocks.append(block)
            current_channels = out_channels
        
        self.final_block_out_channels = current_channels # Channels after last block

        # Classifier Head (if not features_only)
        if not self.features_only:
            # Based on Table 11 and common Timm head structure for MNv4
            # Input to head is `self.final_block_out_channels`
            # Conv2D (1x1) -> 960
            # Conv2D (1x1) -> 1280
            # AvgPool (7x7 global)
            # Conv2D (1x1) -> num_classes (or Linear layer after pool)

            _head_dim1 = _make_divisible(960 * width_multiplier, 8)
            _head_dim2 = _make_divisible(1280 * width_multiplier, 8) # This is input to FC
            self.num_features = _head_dim2 # Timm compatibility

            head_modules = []
            head_modules.append( # Corresponds to 'Conv2D - 960'
                ConvBNAct(self.final_block_out_channels, _head_dim1, kernel_size=1,
                          act_layer=act_layer, bn_layer=bn_layer)
            )
            head_modules.append( # Corresponds to 'Conv2D - 1280'
                ConvBNAct(_head_dim1, _head_dim2, kernel_size=1,
                          act_layer=act_layer, bn_layer=bn_layer)
            )
            head_modules.append(nn.AdaptiveAvgPool2d(1))
            head_modules.append(nn.Flatten(1))
            if drop_rate > 0.0:
                # Ensure inplace=False for QAT compatibility if it's not default
                head_modules.append(nn.Dropout(p=drop_rate, inplace=False))
            head_modules.append(nn.Linear(_head_dim2, num_classes))
            self.head = nn.Sequential(*head_modules)
        else:
            self.head = nn.Identity()
            self.num_features = self.final_block_out_channels


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        
        if self.features_only and self.return_features_indices:
            features_out = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self.return_features_indices:
                    features_out.append(x)
            return features_out
        elif self.features_only: # Return only the final feature map if no specific indices given
            for block in self.blocks:
                x = block(x)
            return [x] # Return as a list for consistency
        else: # Classification mode
            # If specific features are requested for collection (e.g. for distillation)
            # but model is in classification mode.
            # This part is not standard for typical classification script, usually handled by hooks.
            # For simplicity, if not `features_only`, we ignore `return_features_indices` for return value.
            for block in self.blocks:
                x = block(x)
            x = self.head(x)
            return x

    def get_feature_info(self):
        """ Returns information about feature maps for specified default feature indices """
        # This is a helper, actual channel counts depend on width_multiplier
        info = []
        # Simulate channel progression
        current_channels = _make_divisible(32 * self.width_multiplier, 8) # after stem
        current_stride = 2 # after stem

        for i, spec in enumerate(self._MNV4_CONV_S_BLOCK_SPECS):
            block_type, _, _, _, out_ch_abs, stride = spec
            out_channels = _make_divisible(out_ch_abs * self.width_multiplier, 8)
            current_channels = out_channels
            current_stride *= stride
            
            is_feature_level = False
            for name, idx in self._DEFAULT_FEATURE_INDICES.items():
                if i == idx:
                    is_feature_level = True
                    info.append(dict(num_chs=current_channels, reduction=current_stride, module=f'blocks.{i}', name=name))
                    break
        return info


class MobileNetV4(nn.Module):
    """
    MobileNetV4 Universal ConvNet Backbone.
    Supports multiple variants like 'conv_s' and 'conv_m'.
    Designed for classification, object detection (multi-scale features), and VLM integration.
    """
    # [block_type, dw_k1, dw_k2, exp_ch, out_ch, stride]
    _MNV4_CONV_S_SPECS = [
        ["FusedIB", None, 3, 32, 32, 2],
        ["FusedIB", None, 3, 96, 64, 2],
        ["ExtraDW", 5, 5, 192, 96, 2],
        ["IB", None, 3, 192, 96, 1], ["IB", None, 3, 192, 96, 1],
        ["ConvNext", 3, None, 384, 96, 1], ["ConvNext", 3, None, 192, 96, 1], ["ConvNext", 3, None, 192, 96, 1],
        ["ExtraDW", 3, 3, 576, 128, 2],
        ["ExtraDW", 5, 5, 512, 128, 1], ["IB", None, 5, 512, 128, 1], ["IB", None, 5, 384, 128, 1],
        ["IB", None, 3, 512, 128, 1], ["IB", None, 3, 512, 128, 1],
    ]

    _MNV4_CONV_M_SPECS = [
        # Stem is handled separately, spec starts after stem
        ["FusedIB", None, 3, 128, 48, 2],
        ["ExtraDW", 3, 5, 192, 80, 2],
        ["ExtraDW", 3, 3, 160, 80, 1],
        ["ExtraDW", 3, 5, 480, 160, 2],
        ["ExtraDW", 3, 3, 160, 160, 1], ["ExtraDW", 3, 3, 160, 160, 1], ["ExtraDW", 3, 5, 640, 160, 1],
        ["ExtraDW", 3, 3, 640, 160, 1], ["ConvNext", 3, None, 640, 160, 1], ["FFN", None, None, 320, 160, 1],
        ["ConvNext", 3, None, 640, 160, 1],
        ["ExtraDW", 5, 5, 960, 256, 2],
        ["ExtraDW", 5, 5, 1024, 256, 1], ["ExtraDW", 3, 5, 1024, 256, 1], ["ExtraDW", 3, 5, 1024, 256, 1],
        ["FFN", None, None, 512, 256, 1], ["ConvNext", 3, None, 1024, 256, 1], ["ExtraDW", 3, 5, 1024, 256, 1],
        ["ExtraDW", 5, 5, 1024, 256, 1], ["FFN", None, None, 512, 256, 1], ["FFN", None, None, 512, 256, 1],
        ["ConvNext", 5, None, 1024, 256, 1],
    ]
    _MNV4_CONV_L_SPECS: List[List[Union[str, int, None]]] = [
        # block_type , dw_k1 , dw_k2 ,  exp_ch , out_ch , stride
        ["FusedIB", None, 3,   96,    48,   2],      # S4
        ["ExtraDW", 3,    5,  192,    96,   2],      # S8
        ["ExtraDW", 3,    3,  384,    96,   1],
        ["ExtraDW", 3,    5,  384,   192,   2],      # S16
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    5,  768,   192,   1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["ConvNext", 3, None, 768,   192,   1],
        ["ExtraDW", 5,    5,  768,   512,   2],      # S32
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
        ["ExtraDW", 5,    3, 2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
        ["ExtraDW", 5,    3, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
        ["ConvNext", 5, None,2048,   512,   1],
    ]
    _MNV4_CONV_XL_SPECS = [
        # block_type, dw_k1, dw_k2, exp_ch, out_ch, stride
        ["FusedIB", None, 3,   128,    64,    2],  # S4
        ["ExtraDW", 3,    5,   256,   128,    2],  # S8
        ["ExtraDW", 3,    3,   512,   128,    1],
        ["ExtraDW", 3,    5,   512,   256,    2],  # S16
        ["ExtraDW", 3,    3,  1024,   256,    1],
        ["ExtraDW", 3,    3,  1024,   256,    1],
        ["ExtraDW", 5,    3,  1024,   256,    1],
        ["ConvNext", 3, None, 1024,   256,    1],
        ["ConvNext", 3, None, 1024,   256,    1],
        ["ConvNext", 3, None, 1024,   256,    1],
    
        ["ExtraDW", 5,    5,  1024,   640,    2],  # S32
        ["ExtraDW", 5,    5,  2048,   640,    1],
        ["ExtraDW", 5,    5,  2048,   640,    1],
        ["ExtraDW", 5,    5,  2048,   640,    1],
        ["ExtraDW", 5,    5,  2048,   640,    1],
        ["ConvNext", 5, None, 2048,   640,    1],
        ["ConvNext", 5, None, 2048,   640,    1],
        ["ConvNext", 5, None, 2048,   640,    1],
        ["ConvNext", 5, None, 2048,   640,    1],
        ["ConvNext", 5, None, 2048,   640,    1],
        ["ExtraDW", 5,    3,  2048,   640,    1],
        ["ExtraDW", 3,    5,  2048,   640,    1],
    ]
    _MNV4_SSHYBRID_S_SPECS = [
        # Stages S4->S8->S16 (Identical to the original conv_s model)
        ["FusedIB",  None, 3,   32,  32, 2],
        ["FusedIB",  None, 3,   96,  64, 2],
        ["ExtraDW",  5,    5,  192,  96, 2],
        ["IB",       None, 3,  192,  96, 1],
        ["IB",       None, 3,  192,  96, 1],
        ["ConvNext", 3,    None,384,  96, 1],
        ["ConvNext", 3,    None,192,  96, 1],
        ["ConvNext", 3,    None,192,  96, 1],
        # Stage S32 (7x7 features - P5) - Refined with alternating blocks
        ["ExtraDW",  3,    3,  576, 128, 2],
        ["ExtraDW",  5,    5,  512, 128, 1],
        ["IB",       None, 5,  512, 128, 1],
        ["SSHybrid", None, None,None,128, 1],
        ["IB",       None, 3,  512, 128, 1],
        ["SSHybrid", None, None,None,128, 1],
    ]
    _MNV4_SSHYBRID_S_BALANCED = [
        # S4->S8 (P3)
        ["FusedIB",  None, 3,   32,  32, 2],
        ["FusedIB",  None, 3,   96,  64, 2],
        # S16 Stage (14x14 features - P4)
        ["ExtraDW",  5,    5,  192,  96, 2],
        ["IB",       None, 3,  192,  96, 1],
        ["IB",       None, 3,  192,  96, 1],
        ["ConvNext", 3, None, 384, 96, 1],
        ["SSHybrid", None, None,None, 96,  1],
        ["ConvNext", 3, None,  192,  96, 1],
        # S32 Stage (7x7 features - P5)
        ["ExtraDW",  3,    3,  576, 128, 2],
        ["ExtraDW",  5,    5,  512, 128, 1],
        ["IB",       None, 5,  512, 128, 1],
        ["IB",       None, 5,  384, 128, 1],
        ["IB",       None, 3,  512, 128, 1],
        ["SSHybrid", None, None,None,128, 1],
    ]
    _MNV4_SSHYBRID_M_SPECS = [
        # block_type , dw_k1 , dw_k2 , exp_ch , out_ch , stride
        ["FusedIB", None, 3,   128,   48,   2],
        ["ExtraDW", 3,    5,   192,   80,   2],
        ["ExtraDW", 3,    3,   160,   80,   1],
           
        ["ExtraDW", 3,    5,   480,  160,   2],
        ["ExtraDW", 3,    3,   640,  160,   1],
        ["ExtraDW", 3,    3,   640,  160,   1],
        ["ExtraDW", 3,    5,   640,  160,   1],
        # ---------- SS-hybrid “16×16 stage” ----------
        ["SSHybrid", None, None, None, 160, 1],  # replaces first MQA
        ["ExtraDW", 3,    3,   640,  160,   1],
        ["SSHybrid", None, None, None, 160, 1],
        ["ConvNext", 3, None, 640,  160,   1],
        ["SSHybrid", None, None, None, 160, 1],
        ["FFN",     None,None, 640,  160,   1],
        ["SSHybrid", None, None, None, 160, 1],
        ["ConvNext", 3, None, 640,  160,   1],
        # ---------- downsample to 8×8 ----------
        ["ExtraDW", 5,    5,   960,  256,   2],
        ["ExtraDW", 5,    5,  1024,  256,   1],
        ["ExtraDW", 3,    5,  1024,  256,   1],
        ["ExtraDW", 3,    5,  1024,  256,   1],
        ["FFN",     None,None,1024,  256,   1],
        ["ConvNext",3, None, 1024,  256,   1],
        ["ExtraDW", 3,    5,   512,  256,   1],
        ["SSHybrid",None,None,None, 256,   1],
        ["ExtraDW", 5,    5,  1024,  256,   1],
        ["SSHybrid",None,None,None, 256,   1],
        ["FFN",     None,None,1024,  256,   1],
        ["SSHybrid",None,None,None, 256,   1],
        ["FFN",     None,None,1024,  256,   1],
        ["SSHybrid",None,None,None, 256,   1],
        ["ConvNext",5, None,  512,  256,   1],
    ]

    _MNV4_SSHYBRID_L_SPECS = [
        ["FusedIB", None, 3,   96,    48,   2],
        ["ExtraDW", 3,    5,  192,    96,   2],
        ["ExtraDW", 3,    3,  384,    96,   1],
        ["ExtraDW", 3,    5,  384,   192,   2],
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    3,  768,   192,   1],
        ["ExtraDW", 3,    5,  768,   192,   1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        # ---------- SS-hybrid “24×24 stage” ----------
        ["SSHybrid", None,None,None, 192, 1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["SSHybrid", None,None,None, 192, 1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["SSHybrid", None,None,None, 192, 1],
        ["ExtraDW", 5,    3,  768,   192,   1],
        ["SSHybrid", None,None,None, 192, 1],
        ["ConvNext",3, None, 768,   192,   1],
        # ---------- downsample to 12×12 ----------
        ["ExtraDW", 5,    5,  768,   512,   2],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["ConvNext",5, None,2048,   512,   1],
        ["ExtraDW", 5,    3, 2048,   512,   1],
        ["ConvNext",5, None,2048,   512,   1],
        ["ConvNext",5, None,2048,   512,   1],
        ["ExtraDW", 5,    3, 2048,   512,   1],
        ["ExtraDW", 5,    5, 2048,   512,   1],
        ["SSHybrid",None,None,None, 512,   1],
        ["ConvNext",5, None,2048,   512,   1],
        ["SSHybrid",None,None,None, 512,   1],
        ["ConvNext",5, None,2048,   512,   1],
        ["SSHybrid",None,None,None, 512,   1],
        ["ConvNext",5, None,2048,   512,   1],
    ]
        
    _MODEL_SPECS = {
        'conv_s': _MNV4_CONV_S_SPECS,
        'conv_m': _MNV4_CONV_M_SPECS,
        'conv_l': _MNV4_CONV_L_SPECS,
        'conv_xl': _MNV4_CONV_XL_SPECS,
        "ssh_hybrid_s": _MNV4_SSHYBRID_S_SPECS,
        "ssh_hybrid_s_bl": _MNV4_SSHYBRID_S_BALANCED,
        'ssh_hybrid_m': _MNV4_SSHYBRID_M_SPECS,
        'ssh_hybrid_l': _MNV4_SSHYBRID_L_SPECS,
    }
    
    _FEATURE_INDICES = {
        'conv_s': {'p3_s8': 1, 'p4_s16': 2, 'p5_s32': 8}, # Block indices for strides 8, 16, 32
        'conv_m': {'p2_s4': 0, 'p3_s8': 1, 'p4_s16': 3, 'p5_s32': 11}, # Block indices for strides 4, 8, 16, 32
        'conv_l': {'p2_s4': 0, 'p3_s8': 1, 'p4_s16': 3, 'p5_s32': 11},
        'conv_xl': {
            'p2_s4': 0,
            'p3_s8': 1,
            'p4_s16': 3,
            'p5_s32': 10,
        },
        'ssh_hybrid_s': {'p3_s8': 1, 'p4_s16': 2, 'p5_s32': 8},
        'ssh_hybrid_s_bl': {'p3_s8': 1, 'p4_s16': 2, 'p5_s32': 8},
        "ssh_hybrid_m": {
            'p2_s4':  0,
            'p3_s8':  1,
            'p4_s16': 3,
            'p5_s32': 15,
        },
        "ssh_hybrid_l": {
            'p2_s4':  0,
            'p3_s8':  1,
            'p4_s16': 3,
            'p5_s32': 18,
        }
    }

    def __init__(
        self,
        variant: str = 'conv_s',
        width_multiplier: float = 1.0,
        num_classes: int = 1000,
        out_features_names: Optional[List[str]] = None,
        out_features_indices: Optional[Tuple[int, ...]] = None,
        features_only: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Type[nn.Module] = nn.ReLU,
        bn_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.variant = variant
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes
        self.features_only = features_only
        self.act_layer = act_layer
        self.bn_layer = bn_layer

        block_specs = self._MODEL_SPECS[variant]
        default_feature_indices = self._FEATURE_INDICES[variant]

        self.return_features_indices: Optional[Tuple[int, ...]] = None
        if out_features_indices:
            self.return_features_indices = tuple(sorted(list(set(out_features_indices))))
        elif out_features_names:
            self.return_features_indices = tuple(sorted(list(set(
                default_feature_indices[name] for name in out_features_names
            ))))
        
        # Stem convolution
        stem_out_channels = _make_divisible(32 * width_multiplier, 8)
        self.stem = ConvBNAct(3, stem_out_channels, kernel_size=3, stride=2, act_layer=act_layer, bn_layer=bn_layer)
        current_channels = stem_out_channels

        # Build blocks
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(block_specs))]
        
        block_builder_map = {
            "FusedIB": MNV4_FusedInvertedBottleneck, "IB": MNV4_InvertedBottleneck,
            "ExtraDW": MNV4_ExtraDepthwiseBlock, "ConvNext": MNV4_ConvNeXtLikeBlock,
            "FFN": MNV4_FFNBlock,
            "SSHybrid": MNV4_SSHybridBlock,
        }

        for i, spec in enumerate(block_specs):
            block_type, dw_k1, dw_k2, exp_ch_abs, out_ch_abs, stride = spec
            exp_channels = _make_divisible(exp_ch_abs * width_multiplier, 8) if exp_ch_abs else 0
            out_channels = _make_divisible(out_ch_abs * width_multiplier, 8)
            
            builder = block_builder_map[block_type]
            if block_type == "FusedIB":
                block = builder(current_channels, out_channels, exp_channels, kernel_size=dw_k2, stride=stride, act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=dpr[i])
            elif block_type == "IB":
                block = builder(current_channels, out_channels, exp_channels, dw_kernel_size=dw_k2, stride=stride, act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=dpr[i])
            elif block_type == "ExtraDW":
                block = builder(current_channels, out_channels, exp_channels, dw_kernel_size1=dw_k1, dw_kernel_size2=dw_k2, stride=stride, act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=dpr[i])
            elif block_type == "ConvNext":
                block = builder(current_channels, out_channels, exp_channels, dw_kernel_size=dw_k1, stride=stride, act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=dpr[i])
            elif block_type == "FFN":
                block = builder(current_channels, out_channels, exp_channels, stride=stride, act_layer=act_layer, bn_layer=bn_layer, drop_path_rate=dpr[i])
            elif block_type == "SSHybrid":
                block = builder(
                    current_channels, out_channels,
                    stride=stride,                     # must be 1 in your specs
                    drop_path_rate=dpr[i],
                    d_state=16, d_conv=3, expand=2,    # defaults; tweak freely
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")

            self.blocks.append(block)
            current_channels = out_channels
        
        self.final_block_out_channels = current_channels

        # Classifier Head
        if num_classes > 0 and not features_only:
            # Head specs differ slightly per variant but follow a similar pattern
            if variant in ['conv_s', 'conv_m', "ssh_hybrid_s_bl", "ssh_hybrid_s", "ssh_hybrid_m"]:
                head_dim1 = 960 # Fixed for 's' and 'm' from paper
                head_dim2 = 1280 # Fixed for 's' and 'm' from paper
            elif variant in ["conv_l", "ssh_hybrid_l"]: # Fallback for future variants
                head_dim1 = _make_divisible(960 * width_multiplier, 8)
                head_dim2 = _make_divisible(1280 * width_multiplier, 8)
            elif variant in ["conv_xl"]:
                head_dim1 = _make_divisible(1280 * width_multiplier, 8)
                head_dim2 = _make_divisible(2048 * width_multiplier, 8)
            else:
                raise NotImplementedError()
            
            self.num_features = head_dim2
            self.head = nn.Sequential(
                ConvBNAct(self.final_block_out_channels, head_dim1, 1, act_layer=act_layer, bn_layer=bn_layer),
                ConvBNAct(head_dim1, head_dim2, 1, act_layer=act_layer, bn_layer=bn_layer),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=drop_rate, inplace=False) if drop_rate > 0. else nn.Identity(),
                nn.Linear(head_dim2, num_classes)
            )
        else:
            self.num_features = self.final_block_out_channels
            self.head = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run the stem and blocks.
        - If out_features_indices is set, returns a list of specified intermediate feature maps.
        - Otherwise, returns the final feature map tensor.
        This design supports both object detection (multi-scale) and VLM (single-scale) use cases.
        """
        x = self.stem(x)
        if self.return_features_indices:
            features_out = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self.return_features_indices:
                    features_out.append(x)
            return features_out
        else:
            for block in self.blocks:
                x = block(x)
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.features_only:
            return x
        # If returning multiple features, the head operates on the last one
        if isinstance(x, list):
            x = x[-1]
        x = self.head(x)
        return x

    def get_feature_info(self):
        """ Returns information about feature maps for specified default feature indices """
        info = []
        spec_list = self._MODEL_SPECS[self.variant]
        feature_indices_map = self._FEATURE_INDICES[self.variant]

        current_channels = _make_divisible(32 * self.width_multiplier, 8)
        current_stride = 2

        for i, spec in enumerate(spec_list):
            _, _, _, _, out_ch_abs, stride = spec
            out_channels = _make_divisible(out_ch_abs * self.width_multiplier, 8)
            current_channels = out_channels
            current_stride *= stride
            
            for name, idx in feature_indices_map.items():
                if i == idx:
                    info.append(dict(num_chs=current_channels, reduction=current_stride, module=f'blocks.{i}', name=name))
                    break
        return info


if __name__ == '__main__':
    # Example Usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classification mode
    print("Testing Classification Mode:")
    model_cls = MobileNetV4ConvSmallPico(num_classes=100, width_multiplier=0.75).to(device)
    model_cls.eval()
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output_cls = model_cls(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output classification shape: {output_cls.shape}")
    # print(model_cls) # Print model structure

    # Feature extraction mode (for PicoDet-like usage)
    print("\nTesting Feature Extraction Mode (specific indices):")
    # Requesting features after blocks 1, 2, and 8 (0-indexed)
    # These correspond to P3 (stride 8), P4 (stride 16), P5 (stride 32)
    feature_indices_to_extract = (
        MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p3_s8'],
        MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p4_s16'],
        MobileNetV4ConvSmallPico._DEFAULT_FEATURE_INDICES['p5_s32'],
    )
    model_feat = MobileNetV4ConvSmallPico(
        num_classes=0, # or features_only=True
        out_features_indices=feature_indices_to_extract,
        width_multiplier=1.0
    ).to(device)
    model_feat.eval()
    output_feat_list = model_feat(dummy_input)
    print(f"Number of feature maps: {len(output_feat_list)}")
    for i, feat in enumerate(output_feat_list):
        print(f"Feature map {i} shape: {feat.shape}")

    print("\nTesting Feature Extraction Mode (named features):")
    model_feat_named = MobileNetV4ConvSmallPico(
        features_only=True, # Important for feature extraction
        out_features_names=['p3_s8', 'p5_s32'], # Request P3 and P5
        width_multiplier=1.0
    ).to(device)
    model_feat_named.eval()
    output_feat_list_named = model_feat_named(dummy_input)
    print(f"Number of feature maps (named): {len(output_feat_list_named)}")
    for i, feat in enumerate(output_feat_list_named):
        print(f"Feature map {i} (named) shape: {feat.shape}")

    # Check parameter names for head to ensure compatibility with script's state_dict extraction
    if model_cls.num_classes > 0:
        print("\nClassifier head parameter names (first few):")
        for name, _ in model_cls.head.named_parameters():
            if 'weight' in name or 'bias' in name: # Print some actual param names
                 print(name)
            if len(name) > 30: # Stop early
                 break
    
    print("\nExample feature info (for width_multiplier=1.0):")
    # Need to set width_multiplier attribute for get_feature_info, or pass it
    temp_model_for_info = MobileNetV4ConvSmallPico(width_multiplier=1.0)
    # A bit hacky, get_feature_info should ideally take width_multiplier as arg or use self.
    temp_model_for_info.width_multiplier = 1.0 # ensure it's set for the method
    feature_info = temp_model_for_info.get_feature_info()
    for info_item in feature_info:
        print(info_item)


    # NEW CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    # --- Test MNv4-Conv-S (Small) ---
    print("--- Testing MNv4-Conv-S (Small) ---")
    
    # 1. Classification Mode
    print("\n1. Classification Mode:")
    model_s_cls = MobileNetV4(variant='conv_s', num_classes=100, width_multiplier=0.75).to(device)
    output_s_cls = model_s_cls(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output classification shape: {output_s_cls.shape}")

    # 2. Object Detection Feature Mode
    print("\n2. Object Detection Mode (multi-scale features):")
    model_s_det = MobileNetV4(variant='conv_s', features_only=True, out_features_names=['p3_s8', 'p4_s16', 'p5_s32']).to(device)
    features_s = model_s_det(dummy_input)
    print(f"Number of feature maps: {len(features_s)}")
    for i, feat in enumerate(features_s):
        print(f"Feature map {i} shape: {feat.shape}")

    # 3. VLM Backbone Mode
    print("\n3. VLM Backbone Mode (single final feature map):")
    model_s_vlm = MobileNetV4(variant='conv_s', features_only=True).to(device)
    feature_s_vlm = model_s_vlm(dummy_input)
    print(f"Final feature map shape: {feature_s_vlm.shape}")
    print(f"Number of output features for projector: {model_s_vlm.num_features}")

    # --- Test MNv4-Conv-M (Medium) ---
    print("\n--- Testing MNv4-Conv-M (Medium) ---")

    # 1. Classification Mode
    print("\n1. Classification Mode:")
    # Using 256x256 input as per Table 12
    dummy_input_256 = torch.randn(2, 3, 256, 256).to(device)
    model_m_cls = MobileNetV4(variant='conv_m', num_classes=1000).to(device)
    output_m_cls = model_m_cls(dummy_input_256)
    print(f"Input shape: {dummy_input_256.shape}")
    print(f"Output classification shape: {output_m_cls.shape}")

    # 2. Object Detection Feature Mode
    print("\n2. Object Detection Mode (multi-scale features):")
    model_m_det = MobileNetV4(variant='conv_m', features_only=True, out_features_names=['p3_s8', 'p4_s16', 'p5_s32']).to(device)
    features_m = model_m_det(dummy_input_256)
    print(f"Number of feature maps: {len(features_m)}")
    for i, feat in enumerate(features_m):
        print(f"Feature map {i} shape: {feat.shape}")

    # 3. Get Feature Info
    print("\nFeature info for MNv4-Conv-M (width_multiplier=1.0):")
    info_m = model_m_det.get_feature_info()
    for info_item in info_m:
        print(info_item)
