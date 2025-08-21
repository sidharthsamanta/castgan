
import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(real_logits, fake_logits):
    return torch.mean(F.relu(1. - real_logits)) + torch.mean(F.relu(1. + fake_logits))

def hinge_g_loss(fake_logits):
    return -torch.mean(fake_logits)

class IdentityLoss(nn.Module):
    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))

class FeatureConsistency(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feat_real, feat_fake):
        return torch.mean(torch.norm(feat_real - feat_fake, dim=1))

class EntropyMinimization(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__(); self.T = temperature
    def forward(self, logits):
        p = F.softmax(logits / self.T, dim=1)
        return torch.mean(torch.sum(-p * torch.log(p + 1e-8), dim=1))

class SourceLabelPreservation(nn.Module):
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)

class StyleStatRegularizer(nn.Module):
    def forward(self, mu_pred, sigma_pred, mu_tgt, sigma_tgt):
        return torch.mean(torch.abs(mu_pred - mu_tgt)) + torch.mean(torch.abs(sigma_pred - sigma_tgt))
