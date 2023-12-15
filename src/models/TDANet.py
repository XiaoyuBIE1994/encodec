
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger

from .tda_base import UConvBlock, GlobLN

logger = get_logger(__name__)



class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.unet = UConvBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        # self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
        return x
    


class TDANet(nn.Module):
    """
    Code modified from TDANet
    Paper: Kai Li et al. "An efficient encoder-decoder architecture with top-down attention for speech separation", ICLR 2023
    Github: https://github.com/JusperLee/TDANet
    """
    def __init__(self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=4,
        num_sources=2,
        sample_rate=8000,
    ):
        super().__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * num_sources,
            out_channels=num_sources,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
    

    def forward(self, input_wav):
        # input shape B, T
        assert input_wav.dim() == 2; "input wav dimension should be B x T, get {} instead".format(input_wav.shape)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )

        # Front end
        x = self.encoder(x.unsqueeze(1))

        # Split paths
        s = x.clone() # bs, enc_num_basis, T'
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x) # bs, out_channels, T' -> bs, out_channels, T'
        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()

        return estimated_waveforms

if __name__ == '__main__':

    from accelerate import Accelerator
    from omegaconf import OmegaConf
    cfg = OmegaConf.create()
    accelerator = Accelerator()

    model = TDANet().cuda()

    tot_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters()) \
                + sum(p.numel() for p in model.ln.parameters()) \
                + sum(p.numel() for p in model.bottleneck.parameters()) \
                
    sep_params = sum(p.numel() for p in model.sm.parameters()) \
                + sum(p.numel() for p in model.mask_net.parameters())

    dec_params =  sum(p.numel() for p in model.decoder.parameters())

    print(
        'Total params: {:.2f} Mb, Enc: {:.2f} Kb, Sep: {:.2f} Mb, Dec: {:.2f} Kb'.format(tot_params/1024**2, enc_params/1024, sep_params/1024**2, dec_params/1024)
    )

    x = torch.rand(4, 32000, device='cuda')

    y = model(x)

    print('Input shape: {}'.format(x.shape))
    print('Output shape: {}'.format(y.shape))
