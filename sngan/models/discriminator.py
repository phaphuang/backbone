import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from disc_layer import *
        

class ResnetDiscriminator(nn.Module):
    def __init__(self):
        super(ResnetDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            sn_block(nch_in=21, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),

            Self_Attn(in_dim=36),

            sn_block(nch_in=36, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=36, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=36, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=36, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),
            sn_block(nch_in=36, nch_out=36, kernel_size=(3,3), stride=(1,2), padding=1, snorm=True),

            nn.ReLU(inplace=True),
        )

        self.minidev = MinibatchStdDev()
        self.snfc = spectral_norm(nn.Linear(36, 1))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 1, 8, 36)

        h_std = self.minidev(x)
        h_final_flattened = torch.sum(h_std, (1,2))
        output = self.snfc(h_final_flattened)

        return output

if __name__ == '__main__':

    x = torch.randn(16, 21, 512, 1)

    net = ResnetDiscriminator()
    out = net(x)
    print(out.shape)