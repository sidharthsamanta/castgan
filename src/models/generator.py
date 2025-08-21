
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, style_mu, style_sigma):
        # x: (B,C,H,W), style_*: (B,C,1,1)
        mu = x.mean(dim=[2,3], keepdim=True)
        std = x.std(dim=[2,3], keepdim=True) + self.eps
        x_norm = (x - mu) / std
        return x_norm * style_sigma + style_mu

class ResBlock(nn.Module):
    def __init__(self, c, adain=True):
        super().__init__()
        self.adain = adain
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        if not adain:
            self.norm1 = nn.InstanceNorm2d(c, affine=True)
            self.norm2 = nn.InstanceNorm2d(c, affine=True)

    def forward(self, x, style=None):
        y = x
        if self.adain:
            y = self.conv1(y)
            y = self.relu(y)
            y = self.conv2(y)
        else:
            y = self.conv1(y); y = self.norm1(y); y = self.relu(y)
            y = self.conv2(y); y = self.norm2(y)
        return x + y

class StyleEncoder(nn.Module):
    """Maps target image (or stats) to per-channel mu/sigma."""
    def __init__(self, c=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, c, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.to_mu = nn.Conv2d(c, c, 1)
        self.to_sigma = nn.Conv2d(c, c, 1)

    def forward(self, x):
        h = self.net(x)
        return self.to_mu(h), torch.abs(self.to_sigma(h)) + 1e-4

class Generator(nn.Module):
    def __init__(self, res_blocks=6, adain=True, base=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.res = nn.ModuleList([ResBlock(base*4, adain=adain) for _ in range(res_blocks)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 7, 1, 3), nn.Tanh()
        )
        self.adain = adain
        if adain:
            self.adain_layer = AdaIN()

    def forward(self, x, style_mu=None, style_sigma=None):
        h = self.enc(x)
        for rb in self.res:
            if self.adain and style_mu is not None and style_sigma is not None:
                h = self.adain_layer(h, style_mu, style_sigma)
            h = rb(h)
        out = self.dec(h)
        return out
