import torch.nn as nn
from utils._cdht.dht_func import C_dht


class RT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(RT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.rt = RT(numAngle=numAngle, numRho=numRho)
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.fist_conv(x)
        x = self.rt(x)
        x = self.convs(x)
        return x


class RT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(RT, self).__init__()
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        accum = self.line_agg(x)
        return accum
