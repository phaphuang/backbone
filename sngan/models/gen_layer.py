import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor

from disc_layer import snconv2d

class sn_block(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True, snorm=False):
        super(sn_block, self).__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(0.02),

            nn.UpsamplingNearest2d(scale_factor=2),
            snconv2d(nch_in=nch_in, nch_out=nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm),

            nn.LeakyReLU(0.02),

            snconv2d(nch_in=nch_out, nch_out=nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            snconv2d(nch_in=nch_in, nch_out=nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm),
        )
    
    def forward(self, x):
        x_res = self.conv(x)
        x = self.upsample(x)
    
        return x + x_res