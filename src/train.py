import os
import shutil
import time
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from preprocessing.tongji_dataset import TongjiDataset
from src.palm_net import PalmNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

"""
Note: train_transform và (valid_transform, test_transform) khác nhau
best_model_tongji: valid và train dùng chung train_transform
best_model_tongji_v1: so sánh giữa train_transform và valid_transform
best_model_tongji_v2: so sánh khi thêm clahe + sharpen
best_model_tongji_v3: so sánh khi thêm sharpen + clahe
best_model_tongji_v4: so sánh khi chỉ thêm clahe, bỏ sharpen
v5: random positive img dùng v3
"""

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(5,5)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)

        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        img_np = clahe.apply(img_np)

        return Image.fromarray(img_np)

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
            return TF.to_pil_image(img_np)
        return img

if "__main__" == __name__:
    if os.path.exists("../runs/"):
        shutil.rmtree("../runs/")
    log_dir = f"../runs/exp_{time.strftime('%Y%m%d-%H%M%S')}"

    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # set transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.98, 1.02)), # small geometric transform
        transforms.ColorJitter(brightness=0.15, contrast=0.15), # brightness/contrast ±15%
        transforms.Grayscale(num_output_channels=1),
        SharpenTransform(p=0.5),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # dataloader
    train_dataset = TongjiDataset(root="../Tongji/train", transform=train_transform)
    val_dataset = TongjiDataset(root="../Tongji/valid", transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)

    # model config
    model = PalmNet().to(device)
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
    # Cơ chế Early Stopping, dừng sau 10 epoch validation loss ko cải thiện
    patience = 10
    best_val_loss = float("inf") # dương vô cực
    epochs_no_improve = 0
    num_epochs = 200
    num_iter = len(train_loader)
    pos_sum = 0
    neg_sum = 0
    count = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, colour="blue", file=sys.stdout)
        for i, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            emb_a = F.normalize(emb_a, dim=1)
            emb_p = F.normalize(emb_p, dim=1)
            emb_n = F.normalize(emb_n, dim=1)

            dist_ap = torch.norm(emb_a - emb_p, dim=1)
            dist_an = torch.norm(emb_a - emb_n, dim=1)

            pos_sum += dist_ap.mean().item()
            neg_sum += dist_an.mean().item()
            count += 1

            loss = criterion(emb_a, emb_p, emb_n)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), epoch * num_iter + i)
            pbar.set_description(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.4f}")

        train_loss /= num_iter

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)

                emb_a = F.normalize(emb_a, dim=1) # normalize lần 2 ấy, cân nhắc bỏ
                emb_p = F.normalize(emb_p, dim=1)
                emb_n = F.normalize(emb_n, dim=1)

                loss = criterion(emb_a, emb_p, emb_n)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        tqdm.write(f"Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}, Pos mean: {pos_sum / count:.4f}, Neg mean: {neg_sum / count:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        # LR Scheduler
        scheduler.step(val_loss)
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_tongji_v5.pth")
            tqdm.write("Best model saved")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break