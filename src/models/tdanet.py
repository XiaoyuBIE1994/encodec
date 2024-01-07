import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tda_base import UConvBlock, GlobLN

class Recurrent(nn.Module):
    def __init__(self, out_channels=128, inter_channels=512, upsampling_depth=5, _iter=4):
        super().__init__()
        self.unet = UConvBlock(out_channels, inter_channels, upsampling_depth)
        self.iter = _iter
        # self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        _x = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(_x + x))
        return x


class NetTDA(nn.Module):
    """
    Code modified from TDANet
    Paper: Kai Li et al. "An efficient encoder-decoder architecture with top-down attention for speech separation", ICLR 2023
    Github: https://github.com/JusperLee/TDANet
    """
    def __init__(self,
        out_channels=128,
        inter_channels=512,
        num_blocks=16,
        upsampling_depth=5,
    ):

        super().__init__()

        self.sm = Recurrent(out_channels, inter_channels, upsampling_depth, num_blocks)
        mask_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv, nn.ReLU())

        self.weights = nn.Parameter(torch.ones(32))
    
    # predict mask
    # def forward(self, x):
    #     s = x.clone()
    #     x = self.sm(x)
    #     x = self.mask_net(x)
    #     x = x * s

    #     return x
    
    # predict mask with weighted sum
    def forward(self, x, feats):
        s = x.clone()

        weights = torch.softmax(self.weights, dim=0)
        weighted_sum = torch.sum(feats * weights.view(-1, 1, 1, 1), dim=0)
        x = self.sm(weighted_sum)
        x = self.mask_net(x)
        x = x * s

        return x
        

if __name__ == '__main__':

    net = NetTDA()
    x = torch.randn(16, 128, 150)

    with torch.no_grad():
        y = net(x)

    print(y.shape)


    




