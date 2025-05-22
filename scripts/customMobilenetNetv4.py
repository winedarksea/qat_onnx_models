import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Type, Union, Dict

try:
    from timm.models.layers import DropPath
except ImportError:
    # Fallback DropPath if timm is not available
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        """
        def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor.div_(keep_prob)
            return x * random_tensor

        def extra_repr(self) -> str:
            return f'drop_prob={round(self.drop_prob,3):0.3f}'


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
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
        bias: bool = False, # Usually False if BN is used
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
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MNV4_FusedInvertedBottleneck(nn.Module):
    """Fused Inverted Bottleneck block for MobileNetV4."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        kernel_size: int, # This is dw_k2 from table, used for the main 3x3 conv
        stride: int,
        act_layer: Type[nn.Module],
        bn_layer: Type[nn.Module],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # Fused part: Conv-BN-Act
        self.conv_fused = ConvBNAct(
            in_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
            act_layer=act_layer, bn_layer=bn_layer
        )
        # Projection: Conv-BN
        self.conv_proj = ConvBNAct(
            expanded_channels, out_channels, kernel_size=1, stride=1,
            act_layer=None, bn_layer=bn_layer # No activation before residual add
        )
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        dw_kernel_size: int, # This is dw_k2 from table
        stride: int,
        act_layer: Type[nn.Module],
        bn_layer: Type[nn.Module],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion: 1x1 Conv-BN-Act
        self.conv_pw_exp = ConvBNAct(
            in_channels, expanded_channels, kernel_size=1, stride=1,
            act_layer=act_layer, bn_layer=bn_layer
        )
        # Depthwise: DWConv-BN-Act
        self.conv_dw = ConvBNAct(
            expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=stride,
            groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer
        )
        # Projection: 1x1 Conv-BN
        self.conv_pw_proj = ConvBNAct(
            expanded_channels, out_channels, kernel_size=1, stride=1,
            act_layer=None, bn_layer=bn_layer # No activation before residual add
        )
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        dw_kernel_size1: int,
        dw_kernel_size2: int,
        stride: int,
        act_layer: Type[nn.Module],
        bn_layer: Type[nn.Module],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion: 1x1 Conv-BN-Act
        self.conv_pw_exp = ConvBNAct(
            in_channels, expanded_channels, kernel_size=1, stride=1,
            act_layer=act_layer, bn_layer=bn_layer
        )
        # Depthwise 1: DWConv-BN-Act
        self.conv_dw1 = ConvBNAct(
            expanded_channels, expanded_channels, kernel_size=dw_kernel_size1, stride=stride, # Stride applied here
            groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer
        )
        # Depthwise 2: DWConv-BN-Act
        self.conv_dw2 = ConvBNAct(
            expanded_channels, expanded_channels, kernel_size=dw_kernel_size2, stride=1,
            groups=expanded_channels, act_layer=act_layer, bn_layer=bn_layer
        )
        # Projection: 1x1 Conv-BN
        self.conv_pw_proj = ConvBNAct(
            expanded_channels, out_channels, kernel_size=1, stride=1,
            act_layer=None, bn_layer=bn_layer # No activation before residual add
        )
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
    """ConvNeXt-like block for MobileNetV4 (simplified)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        dw_kernel_size: int, # This is dw_k1 from table
        stride: int,
        act_layer: Type[nn.Module],
        bn_layer: Type[nn.Module],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Depthwise conv part (stride is applied here)
        self.conv_dw = ConvBNAct(
            in_channels, in_channels, kernel_size=dw_kernel_size, stride=stride,
            groups=in_channels, act_layer=act_layer, bn_layer=bn_layer
        )
        # Pointwise expansion (MLP-like part)
        self.conv_pw_exp = ConvBNAct(
            in_channels, expanded_channels, kernel_size=1, stride=1,
            act_layer=act_layer, bn_layer=bn_layer
        )
        # Pointwise projection
        self.conv_pw_proj = ConvBNAct(
            expanded_channels, out_channels, kernel_size=1, stride=1,
            act_layer=None, bn_layer=bn_layer # No activation before residual add
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        # Note: In original ConvNeXt, the input to PW convs is the output of DW.
        # If DW changes channel count (not here), then PW needs to match.
        # Here, in_channels for conv_pw_exp is same as input to conv_dw.
        x_mlp = self.conv_pw_exp(x) # Taking output of DW Conv as input to MLP part
        x_mlp = self.conv_pw_proj(x_mlp)

        if self.use_residual:
            # If DW conv changes spatial res (stride > 1), shortcut is typically identity mapped
            # or downsampled. Here, residual is only if stride=1.
            x = self.drop_path(x_mlp) + shortcut
        else:
            # If not using residual (e.g. stride != 1 or channels changed in a way that disallows it)
            # the output is just the MLP part.
            # However, standard ConvNeXt block structure *always* has a residual.
            # If stride > 1, the shortcut path is typically downsampled.
            # For MNv4, it seems residual is only for S=1, C_in=C_out blocks.
            # If the DW conv has stride > 1, the output `x` from conv_dw is already downsampled.
            # The residual connection logic for blocks with stride > 1 is typically handled by not having one,
            # or by having a parallel path that also downsamples the shortcut.
            # For simplicity and adherence to typical MobileNet patterns, residual only if S=1, C_in=C_out.
            x = x_mlp 
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

        # Determine which features to output
        self.return_features_indices: Optional[Tuple[int, ...]] = None
        if out_features_indices:
            self.return_features_indices = tuple(sorted(list(set(out_features_indices))))
        elif out_features_names:
            self.return_features_indices = tuple(sorted(list(set(
                self._DEFAULT_FEATURE_INDICES[name] for name in out_features_names
                if name in self.DEFAULT_FEATURE_INDICES
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
