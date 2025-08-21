
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        # Placeholder: fill with sampled triplets or use batch-hard logic
        return emb.sum()*0.0
