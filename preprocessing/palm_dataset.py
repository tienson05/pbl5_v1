import os
import random

from PIL import Image

from torch.utils.data import Dataset

class PalmDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        images = os.listdir(root)
        for image in images:
            label = image.split('_')
            self.image_paths.append(os.path.join(root, image))
            self.labels.append(label[0] + "_" + label[2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ===== Anchor =====
        anchor_path = self.image_paths[idx]
        anchor = Image.open(anchor_path).convert("RGB")
        anchor_label = self.labels[idx]

        # ===== Positive (cùng label, khác idx) =====
        same_class_indices = [
            i for i, label in enumerate(self.labels)
            if label == self.labels[idx] and i != idx
        ]
        positive_idx = random.choice(same_class_indices)
        positive_path = self.image_paths[positive_idx]
        positive = Image.open(positive_path).convert("RGB")

        # ===== Negative (khác label) =====
        diff_class_indices = [
            i for i, label in enumerate(self.labels)
            if label != anchor_label
        ]
        negative_idx = random.choice(diff_class_indices)
        negative = Image.open(self.image_paths[negative_idx]).convert("RGB")

        # ===== Transform =====
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative