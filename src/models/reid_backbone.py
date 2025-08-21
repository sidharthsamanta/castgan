
import torch.nn as nn
import torchvision.models as tv

class ReIDBackbone(nn.Module):
    def __init__(self, name='resnet50', embedding_dim=2048, num_classes=0):
        super().__init__()
        assert name == 'resnet50', 'Skeleton supports resnet50; extend as needed.'
        m = tv.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # pool
        self.embedding_dim = 2048
        self.classifier = nn.Linear(self.embedding_dim, num_classes) if num_classes>0 else None

    def forward(self, x, return_feat=False):
        h = self.backbone(x)  # (B,2048,1,1)
        h = h.flatten(1)
        if return_feat or self.classifier is None:
            return h
        return self.classifier(h)
