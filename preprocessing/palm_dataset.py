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
        self.label_to_indices = {}

        images = os.listdir(root)

        for idx, image in enumerate(images):

            path = os.path.join(root, image)

            name = image.split(".")[0]
            parts = name.split("_")

            label = parts[0] + "_" + parts[2]

            self.image_paths.append(path)
            self.labels.append(label)

            if label not in self.label_to_indices:
                self.label_to_indices[label] = []

            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):

        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        anchor = Image.open(anchor_path).convert("RGB")

        # positive
        positive_idx = random.choice(
            [i for i in self.label_to_indices[anchor_label] if i != idx]
        )

        positive = Image.open(self.image_paths[positive_idx]).convert("RGB")

        # negative
        negative_label = random.choice(
            [l for l in self.unique_labels if l != anchor_label]
        )

        negative_idx = random.choice(self.label_to_indices[negative_label])

        negative = Image.open(self.image_paths[negative_idx]).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative