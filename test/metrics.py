import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F


# =============================
# COMPUTE EMBEDDINGS
def compute_embeddings(model, dataloader, device):
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            emb = model(img)
            emb = F.normalize(emb, dim=1)

            embeddings.append(emb.cpu())
            labels.extend(label.numpy())

    embeddings = torch.cat(embeddings)

    return embeddings, np.array(labels)


# =============================
# BUILD GALLERY
def build_gallery(embeddings, labels):

    gallery = {}
    label_dict = defaultdict(list)

    for emb, label in zip(embeddings, labels):
        label_dict[label].append(emb)

    for label in label_dict:
        gallery[label] = torch.stack(label_dict[label]).mean(0)

    return gallery


# =============================
# CREATE PAIRS
def create_pairs(probe_embs, probe_labels, gallery):

    distances = []
    labels = []

    for emb, label in zip(probe_embs, probe_labels):

        for g_label in gallery:

            d = torch.norm(emb - gallery[g_label]).item()

            distances.append(d)

            labels.append(1 if label == g_label else 0)

    return np.array(distances), np.array(labels)


# =============================
# FAR / FRR
def compute_far_frr(distances, labels, threshold):

    preds = distances < threshold

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (FN + TP + 1e-8)

    return FAR, FRR, TP, TN, FP, FN


# =============================
# EER
def compute_eer(distances, labels):

    thresholds = np.linspace(distances.min(), distances.max(), 1000)

    fars = []
    frrs = []

    for t in thresholds:

        FAR, FRR, *_ = compute_far_frr(distances, labels, t)

        fars.append(FAR)
        frrs.append(FRR)

    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = np.argmin(np.abs(fars - frrs))

    eer = (fars[idx] + frrs[idx]) / 2
    threshold = thresholds[idx]

    return eer, threshold


# =============================
# ACCURACY
def compute_accuracy(TP, TN, FP, FN):

    return (TP + TN) / (TP + TN + FP + FN + 1e-8)


# =============================
# TPR @ FPR
def compute_tpr_at_fpr(distances, labels, target_fpr=1e-3):

    thresholds = np.linspace(distances.min(), distances.max(), 2000)

    best_tpr = 0

    for t in thresholds:

        FAR, FRR, TP, TN, FP, FN = compute_far_frr(distances, labels, t)

        if FAR <= target_fpr:

            TPR = TP / (TP + FN + 1e-8)

            best_tpr = max(best_tpr, TPR)

    return best_tpr