
import torch.nn as nn

def conv_bn_lrelu(in_c, out_c, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.LeakyReLU(0.2, inplace=True)
    )

class PatchDiscriminator(nn.Module):
    """Simple PatchGAN-style discriminator."""
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_lrelu(3, base, 4, 2, 1),
            conv_bn_lrelu(base, base*2, 4, 2, 1),
            conv_bn_lrelu(base*2, base*4, 4, 2, 1),
            conv_bn_lrelu(base*4, base*8, 4, 1, 1),
            nn.Conv2d(base*8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
