"""
Common modules for BEVFormer
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


"""
Classes in BEVFormer
"""


class ResnetEncoder(nn.Module):
    """
    ResNet as the image backbone to retrieve features.

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """
    def __init__(self, params):
        super(ResnetEncoder, self).__init__()

        self.num_layers = params['depth']
        self.pretrained = params['pretrained']

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet "
                "layers".format(self.num_layers))

        self.encoder = resnets[self.num_layers](self.pretrained)

    def forward(self, input_images):
        """
        Compute deep features from input images.

        Parameters
        ----------
        input_images : torch.Tensor
            The input images have shape of (B,C,H,W)

        Returns
        -------
        features: LIST
            The multi-scale features
        """

        x = self.encoder.conv1(input_images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        # the BEVFormer tiny only return the last stage feature
        return [x]


class FPN(nn.Module):
    """
    FPN as image neck in BEVFormer Tiny to augment feature. Bevformer tiny
    only has a single scale,

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """
    def __init__(self, params):
        super().__init__()

        self.in_channels = params['in_channels']
        self.out_channels = params['out_channels']

        # lateral convolution in FPN
        self.lateral_convs = nn.ModuleList()
        # fpn convolutions
        self.fpn_convs = nn.ModuleList()

        l_conv = nn.Conv2d(self.in_channels[0], self.out_channels, 1,
                           stride=1, padding=0)
        fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, 3,
                             stride=1, padding=1)

        self.lateral_convs.append(l_conv)
        self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        # inputs: [(b, c, h, w)]

        # build laterals. We only have one scale for bevformer tiny
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        # we will actually go into this for loop since it is single scale
        # for bevformer tiny. But it is still good to show
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # for bevformer bass version there should be extra convs. But here
        # miny version just directly the output
        return tuple(outs)


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.,
                 add_identity=True,
                 **kwargs):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels

        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))

        self.layers = nn.Sequential(*layers)
        self.dropout_layer = torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


