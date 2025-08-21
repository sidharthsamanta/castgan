
import torch

def rbf_mmd(x, y, sigma=1.0):
    # x: (n,d), y: (m,d)
    x2 = (x**2).sum(1, keepdim=True)
    y2 = (y**2).sum(1, keepdim=True)
    xy = x @ y.t()
    k_xx = torch.exp(-(x2 - 2*xy + y2.t())/(2*sigma**2))
    k_yy = torch.exp(-(y2 - 2*xy.t() + x2.t())/(2*sigma**2))
    k_xy = torch.exp(-(x2 - 2*xy + y2.t())/(2*sigma**2))
    mmd2 = k_xx.mean() + k_yy.mean() - 2*k_xy.mean()
    return mmd2
