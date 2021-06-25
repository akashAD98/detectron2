#!/usr/bin/env python

from typing import List

import torch.nn as nn
import torch.nn.functional as F

# from detectron2.modeling import BACKBONE_REGISTRY, Backbone, FPN, ShapeSpec

from ...modeling import BACKBONE_REGISTRY, Backbone, FPN 
from ...layers import ShapeSpec

#from .efficient_net_v2 import EfficientNetV2
#from ..layers import ConvBNA, FusedMBConv, MBConv
from ...config.detectron2_config import get_cfg

##
from ...layers import ConvBNA,MBConv,FusedMBConv
### added


__all__ = [
	"EfficientNet",
	"EfficientNetV2",
	"LastLevelMaxPool",
    "build_effnet_backbone",
    "build_effnet_fpn_backbone",
]



class EfficientNet(EfficientNetV2, Backbone):
    def __init__(self, cfg, out_features: List[str] = None):
        super(EfficientNet, self).__init__(cfg)

        self.out_features = ['s6'] if out_features is None else out_features

        self.strides = {}
        self.channels = {}
        for key in self.out_features:
            index = self.stage_indices[key]
            stride, channel = 1, 0
            for child in self.backbone[:index].children():
                if isinstance(child, (ConvBNA, FusedMBConv, MBConv)):
                    stride, channel = (
                        stride * child.stride,
                        child.out_channels,
                    )
            self.strides[key] = stride
            self.channels[key] = channel

        assert len(self.channels) > 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels[name], stride=self.strides[name]
            )
            for name in self.out_features
        }

    def forward(self, x):
        features = self.stage_forward(x)
        return {name: features[name] for name in self.out_features}

    def freeze(self, at: int = 0):
        # TODO: freeze at selected layer
        for parameters in self.parameters():
            parameters.requires_grad = False
        return self


@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg=None, input_shape=None):
    config = get_cfg()
    return EfficientNet(config, ['s2', 's3', 's4', 's5', 's6'])  # .freeze(0)


@BACKBONE_REGISTRY.register()
def build_effnet_fpn_backbone(cfg=None, input_shape=None):
    backbone = build_effnet_backbone()
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    return FPN(
        bottom_up=backbone,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


###########
class EfficientNetV2(nn.Module):
    def __init__(self, cfg: CN, in_channels: int = 3):
        super(EfficientNetV2, self).__init__()

        # input_shape = cfg.get('INPUTS').get('SHAPE')
        backbone = cfg['BACKBONE']
        # assert len(input_shape) == 3
        # in_channels = input_shape[0]

        layers, in_channels = self.build(backbone, in_channels)
        self.backbone = nn.Sequential(*layers)

        try:
            head = cfg['HEAD']
            layers, in_channels = self.build(head, in_channels)
            self.head = nn.Sequential(*layers)
        except KeyError:
            self.head = None

        self.out_channels = in_channels

    def build(self, nodes, in_channels):
        layers = []
        for index, (stage, node) in enumerate(nodes.items()):
            for i in range(node.pop('LAYERS', 1)):
                stride = node.get('STRIDE', 1) if i == 0 else 1
                assert stride
                layers.append(self.create_layer(node, in_channels, stride))
                in_channels = node.get('CHANNELS')

        return layers, in_channels

    def create_layer(self, node: CN, in_channels: int, stride: int):
        node = deepcopy(node)

        ops = node.pop('OPS')
        out_channels = node.pop('CHANNELS', None)
        kernel_size = node.get('KERNEL')
        expansion = node.get('EXPANSION')
        se = node.get('SE', 0)
        padding = node.get('PADDING', 0)

        if ops == 'conv':
            layer = ConvBNA(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif ops == 'mbconv':
            layer = MBConv(
                in_channels=in_channels,
                expansion=expansion,
                out_channels=out_channels,
                knxn=kernel_size,
                stride=stride,
                reduction=se,
            )
        elif ops == 'fused_mbconv':
            layer = FusedMBConv(
                in_channels=in_channels,
                expansion=expansion,
                out_channels=out_channels,
                knxn=kernel_size,
                stride=stride,
                reduction=se,
            )
        else:
            layer = getattr(nn, ops)
            if not issubclass(layer, nn.Module):
                raise ValueError(f'Unknown layer type {ops}')
            layer = layer(**node)

        return layer

    def forward(self, x):
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def stage_forward(self, x):
        s0 = self.backbone[0](x)
        s1 = self.backbone[1:3](s0)
        s2 = self.backbone[3:7](s1)
        s3 = self.backbone[7:11](s2)
        s4 = self.backbone[11:17](s3)
        s5 = self.backbone[17:26](s4)
        s6 = self.backbone[26:](s5)

        return {
            # 's0': s0,
            # 's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            's5': s5,
            's6': s6,
        }

    @property
    def stage_indices(self):
        return {
            's0': 1,
            's1': 3,
            's2': 7,
            's3': 11,
            's4': 17,
            's5': 26,
            's6': 41,
        }
