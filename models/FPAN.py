import torch.nn.functional as F
from torch import nn
from models.MultiScaleAttMod import MSAM


class FPAN(nn.Module):

    def __init__(
            self, in_channels_list, out_channels, top_blocks=None
    ):
        super(FPAN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpan_inner{}".format(idx)
            layer_block = "fpan_layer{}".format(idx)
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks
        self.MSAM3 = MSAM(channels_high=in_channels_list[1], channels_output=out_channels)
        self.MSAM4 = MSAM(channels_high=in_channels_list[2], channels_output=out_channels)
        self.MSAM5 = MSAM(channels_high=out_channels, channels_output=out_channels)

    def forward(self, x):
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        features = []
        middles = []
        results = []
        middles.append(last_inner)
        features.append(last_inner)
        for feature, inner_block in zip(x[:-1][::-1], self.inner_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(last_inner,
                                           size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                                           mode='nearest')
            last_inner = inner_lateral + inner_top_down
            features.insert(0, inner_lateral)
            middles.insert(0, last_inner)
        last_results = self.top_blocks(x[-1])
        att5 = self.MSAM5(features[2], last_results[0])
        att4 = self.MSAM4(features[1], x[2])
        att3 = self.MSAM3(features[0], x[1])
        middles[2] = middles[2] + att5
        middles[1] = middles[1] + att4
        middles[0] = middles[0] + att3
        for middle, layer_block in zip(middles, self.layer_blocks):
            results.append(getattr(self, layer_block)(middle))
        results.extend(last_results)
        return tuple(results)


class LastLevelP6P7(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5):
        x = c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
