import cv2
from PIL import Image
from torch.utils.data import DataLoader
from preprocessing.palm_img_dataset import PalmImageDataset
from src.palm_net import PalmNet
from metrics import *
import torchvision.transforms as transforms

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================
    # TRANSFORM
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # =============================
    # DATASET
    gallery_dataset = PalmImageDataset(
        "../Tongji/test/session1",
        transform
    )

    probe_dataset = PalmImageDataset(
        "../Tongji/test/session2",
        transform
    )

    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False)
    probe_loader = DataLoader(probe_dataset, batch_size=64, shuffle=False)

    # =============================
    # MODEL
    model = PalmNet().to(device)

    model.load_state_dict(torch.load("../src/best_model_tongji_v5.pth"))

    model.eval()

    # =============================
    # COMPUTE EMBEDDINGS
    gallery_emb, gallery_labels = compute_embeddings(
        model,
        gallery_loader,
        device
    )

    gallery = build_gallery(gallery_emb, gallery_labels)

    probe_emb, probe_labels = compute_embeddings(
        model,
        probe_loader,
        device
    )
    print("gallery labels:", list(gallery.keys())[:10])
    print("probe labels:", np.unique(probe_labels)[:10])
    # =============================
    # CREATE PAIRS
    distances, pair_labels = create_pairs(
        probe_emb,
        probe_labels,
        gallery
    )
    pos = distances[pair_labels == 1]
    neg = distances[pair_labels == 0]

    print("pos mean:", pos.mean())
    print("neg mean:", neg.mean())

    # =============================
    # METRICS
    eer, threshold = compute_eer(distances, pair_labels)

    FAR, FRR, TP, TN, FP, FN = compute_far_frr(
        distances,
        pair_labels,
        threshold
    )

    acc = compute_accuracy(TP, TN, FP, FN)

    tpr = compute_tpr_at_fpr(distances, pair_labels)

    # =============================
    # PRINT RESULTS
    print("\n===== TEST RESULTS =====")

    print(f"EER: {eer:.4f}")
    print(f"Threshold: {threshold:.4f}")

    print(f"FAR: {FAR:.4f}")
    print(f"FRR: {FRR:.4f}")

    print(f"Accuracy: {acc:.4f}")

    print(f"TPR@FPR=1e-3: {tpr:.4f}")

"""
Tongji dataset
===== TEST RESULTS =====
EER: 0.0175
Threshold: 0.7050
FAR: 0.0175
FRR: 0.0175
Accuracy: 0.9825
TPR@FPR=1e-3: 0.7108

===== TEST RESULTS ===== v1
EER: 0.0125
Threshold: 0.7468
FAR: 0.0125
FRR: 0.0125
Accuracy: 0.9875
TPR@FPR=1e-3: 0.8642

===== TEST RESULTS ===== v2
EER: 0.0108
Threshold: 0.6539
FAR: 0.0109
FRR: 0.0108
Accuracy: 0.9891
TPR@FPR=1e-3: 0.8200
"""

