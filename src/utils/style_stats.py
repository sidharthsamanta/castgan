
import torch

def batch_channel_stats(x):
    # x: (B,C,H,W)
    mu = x.mean(dim=[2,3], keepdim=True)
    sigma = x.std(dim=[2,3], keepdim=True) + 1e-5
    return mu, sigma

def global_channel_stats(x_list, max_batches=64):
    # naive running average over a few batches
    mus, sigmas = [], []
    for x in x_list[:max_batches]:
        mu, sigma = batch_channel_stats(x)
        mus.append(mu.mean(dim=0, keepdim=True))
        sigmas.append(sigma.mean(dim=0, keepdim=True))
    mu = torch.mean(torch.stack(mus, dim=0), dim=0)
    sigma = torch.mean(torch.stack(sigmas, dim=0), dim=0)
    return mu, sigma
