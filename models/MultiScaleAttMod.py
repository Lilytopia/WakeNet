import torch
import torch.nn as nn


class MSAM(nn.Module):
    def __init__(self, channels_high, channels_output=256, divchannel=4):
        super(MSAM, self).__init__()
        self.conv_high = nn.Sequential(nn.Conv2d(channels_high, channels_output, kernel_size=1, padding=0, bias=False),
                                       nn.BatchNorm2d(channels_output),
                                       nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(channels_output, channels_output//divchannel)
        self.fc2 = nn.Linear(channels_output, channels_output//divchannel)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.output_channels = channels_output

    def forward(self, fms_low, fms_high):
        b, c1, h1, w1 = fms_low.shape

        if c1 != self.output_channels:
            raise ValueError

        feat_low = fms_low.view(b, -1, self.output_channels)
        feat_high = self.conv_high(fms_high).view(b, -1, self.output_channels)

        mid_low = self.fc1(feat_low)
        mid_high = self.fc2(feat_high).permute(0, 2, 1)
        energy = torch.bmm(mid_low, mid_high)
        attention = self.softmax(energy)

        mid = torch.bmm(feat_high.permute(0, 2, 1), attention.permute(0, 2, 1)).view(b, self.output_channels, h1, w1)
        out = self.scale*mid

        return out
