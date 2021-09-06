import math
import torch
import torch.nn as nn
from models.RT import RT


class CLSHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_classes, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.sigmoid(self.head(x))
        x = x.permute(0, 2, 3, 1)
        n, w, h, c = x.shape
        x = x.reshape(n, w, h, self.num_anchors, self.num_classes)
        return x.reshape(x.shape[0], -1, self.num_classes)


class REGHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_regress, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, self.num_regress)


class LDMHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 num_landmarks,
                 level):
        super(LDMHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_landmarks = num_landmarks
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True))
        if level == 'p3':
            self.dht = RT(numAngle=384, numRho=96)
        elif level == 'p4':
            self.dht = RT(numAngle=192, numRho=48)
        elif level == 'p5':
            self.dht = RT(numAngle=96, numRho=24)
        elif level == 'p6':
            self.dht = RT(numAngle=48, numRho=12)
        elif level == 'p7':
            self.dht = RT(numAngle=24, numRho=6)
        else:
            raise NotImplementedError
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, (2, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, (2, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True))
        self.head1 = nn.Conv2d(feat_channels, num_anchors * 2, 3, 1, 1)
        self.head2 = nn.Conv2d(feat_channels, num_anchors * (num_landmarks - 2), 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.head1.weight.data.fill_(0)
        self.head1.bias.data.fill_(0)
        self.head2.weight.data.fill_(0)
        self.head2.bias.data.fill_(0)

    def forward(self, x):
        x1 = self.convs1(x)
        x1 = self.head1(x1)
        x2 = self.dht(x)
        x2 = self.convs2(x2)
        x2 = self.head2(x2)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = x1.reshape(x1.shape[0], -1, 2)
        x2 = x2.reshape(x2.shape[0], -1, (self.num_landmarks - 2))
        return torch.cat([x1, x2], dim=-1)
