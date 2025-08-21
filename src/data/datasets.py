
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ReIDImageFolder(Dataset):
    """Placeholder dataset. Replace with real Market-1501/Duke loader."""
    def __init__(self, root: str, transform=None, with_labels: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.with_labels = with_labels
        self.samples = []  # fill: List[(path, label)] or [path]
        # TODO: scan folders and build samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        if self.with_labels:
            path, label = entry
        else:
            path = entry; label = -1
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
