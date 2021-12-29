import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .gen_layer import *
from .disc_layer import Self_Attn
        

class ResnetGenerator(nn.Module):
    def __init__(self):
        super(ResnetGenerator, self).__init__()
        
        self.snfc = spectral_norm(nn.Linear(128, 1*8*1536))

        self.layers = nn.Sequential(
            sn_block(nch_in=1536, nch_out=1536, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=1536, nch_out=768, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=768, nch_out=384, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=384, nch_out=192, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=192, nch_out=96, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),

            Self_Attn(in_dim=96),

            sn_block(nch_in=96, nch_out=48, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
        )

        self.bn = nn.BatchNorm2d(48)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.last = snconv2d(nch_in=48, nch_out=21, kernel_size=(1,1), padding=0, snorm=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.snfc(x)
        x = x.view(-1, 1536, 8, 1)
        x = self.layers(x)
        x = self.leakyrelu(self.bn(x))

        out = self.tanh(self.last(x))

        return out

if __name__ == '__main__':

    x = torch.randn(16, 128)

    net = ResnetGenerator()
    out = net(x)
    print(out.shape)