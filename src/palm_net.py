from torch import nn
import torch.nn.functional as F
from src.res_block import ResBlock


class PalmNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 224x224
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), # 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 56
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2), # 28
            ResBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2), # 14
            ResBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2), # 7
            ResBlock(512, 512),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(512, 128),
        )
        self.embedding = nn.Linear(512, 128)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head(out)
        out = self.embedding(out)
        out = F.normalize(out, p=2, dim=1)
        return out
