
import torch
import torch.nn.functional as F

def temperature_scale(logits, T=1.0):
    return logits / T

def max_confidence(logits, T=1.0):
    p = F.softmax(logits / T, dim=1)
    return p.max(dim=1).values
