import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from ..backbone import build_backbone
from ..roi_heads.roi_heads_3d import build_3d_roi_heads
from ..stm_decoder.stm_decoder import build_stm_decoder


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, t, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ActionDetector(nn.Module):
    def __init__(self, cfg):
        super(ActionDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self._construct_space(cfg)
        self.stm_head = build_stm_decoder(cfg)
        self.roi_heads = build_3d_roi_heads(cfg, self.backbone.dim_out)
        self.device = torch.device("cuda")

    def _construct_space(self, cfg):
        out_channel = cfg.MODEL.STM.HIDDEN_DIM
        if "vit" in cfg.MODEL.BACKBONE.CONV_BODY.lower():
            in_channels = [cfg.ViT.EMBED_DIM] * 4
            self.lateral_convs = nn.ModuleList()

            for idx, scale in enumerate([4, 2, 1, 0.5]):
                dim = in_channels[idx]
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        LayerNorm(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                    out_dim = dim
                elif scale == 0.5:
                    layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                layers.extend(
                    [
                        nn.Conv3d(
                            out_dim,
                            out_channel,
                            kernel_size=1,
                            bias=False,
                        ),
                        LayerNorm(out_channel),
                        nn.Conv3d(
                            out_channel,
                            out_channel,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
                layers = nn.Sequential(*layers)

                self.lateral_convs.append(layers)
        else:
            in_channels = [256 + 64, 512 + 128, 1024 + 256, 2048 + 256]
            self.lateral_convs = nn.ModuleList()
            for idx, in_channel in enumerate(in_channels):
                lateral_conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)
                weight_init.c2_xavier_fill(lateral_conv)
                self.lateral_convs.append(lateral_conv)

    def space_forward(self, features):
        mapped_features = []
        for i, feature in enumerate(features):
            mapped_features.append(self.lateral_convs[i](feature))
        return mapped_features

    def forward(self, slow_video, fast_video, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):

        if part_forward == 1:
            slow_features = fast_features, features = None
        else:
            slow_features, fast_features, features = self.backbone(slow_video, fast_video)
        mapped_features = self.space_forward(features)

        result, detector_losses, loss_weight, detector_metrics = self.roi_heads(
            slow_features, fast_features, boxes, objects, keypoints, extras, part_forward
        )

        if self.training:
            return detector_losses, loss_weight, detector_metrics, result

        return result

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        new_key = name + "." + key
                        weight_map[new_key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping


def build_detection_model(cfg):
    return ActionDetector(cfg)
