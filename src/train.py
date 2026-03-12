import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import numpy as np
from torch.utils.data import DataLoader

from preprocessing.palm_dataset import PalmDataset
from src.palm_net import PalmNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class SharpenTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img)
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            img_np = cv2.filter2D(img_np, -1, kernel)
            return F.to_pil_image(img_np)
        return img

if "__main__" == __name__:
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transforms
    train_transform = transforms.Compose([
        # resize ROI
        transforms.Resize((224, 224)),
        # small geometric transform
        transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        # brightness / contrast ±15%
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        # sharpening filter
        SharpenTransform(p=0.5),
        # convert to grayscale
        transforms.Grayscale(num_output_channels=1),
        # tensor
        transforms.ToTensor(),
        # normalize [-1,1]
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    # dataloader
    train_dataset = PalmDataset("../dataset/train")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)

    # model config
    model = PalmNet().to(device)
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # bộ lập lịch learning rate ReduceLROnPlateau được tích hợp vào quy trình huấn luyện
    # theo dõi validation loss, ko cải thiện trong 5 epoch, lr giảm đi 10 lần
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # minimize validation loss
        factor=0.1,  # giảm LR xuống 1/10
        patience=5,  # chờ 5 epoch không cải thiện
        threshold=1e-4,  # (tùy chọn) ngưỡng thay đổi nhỏ để coi là "không cải thiện"
        min_lr=1e-6,  # (tùy chọn) LR nhỏ nhất
    )
    # cơ chế Early Stopping, dừng sau 10 epoch validation loss ko cải thiện
    patience = 10
    best_val_loss = float("inf") # dương vô cực
    epochs_no_improve = 0
    num_epochs = 1000
    num_iter = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for i, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            writer.add_scalar(
                "Loss/train",
                loss.item(),
                epoch * len(train_loader) + i
            )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (anchor, positive, negative) in enumerate(val_loader):
                emb_a = model(anchor).to(device)
                emb_p = model(positive).to(device)
                emb_n = model(negative).to(device)

                loss = criterion(emb_a, emb_p, emb_n)

                val_loss += loss.item()
                writer.add_scalar("Loss/val", loss.item(), epoch * num_iter + i)

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        # ===== LR Scheduler =====
        scheduler.step(val_loss)
        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break