import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from src.train import SharpenTransform

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(5,5)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.clahe.apply(img)
        return Image.fromarray(img)

CLAHE_CLIP_LIMIT = 2.0              # Clip limit for CLAHE contrast enhancement.
CLAHE_TILE_GRID_SIZE = (5, 5)       # Tile grid size for CLAHE.
contrast = cv2.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
def enhance_contrast_clahe(gray_image: np.ndarray) -> np.ndarray:
    return contrast.apply(gray_image)

def detect_hand(image_path, name):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            roi = crop_palm_roi(image_rgb, hand.landmark, 224)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imshow(name, roi_bgr)
            cv2.waitKey(0)
            # i = name.split("_")[1]
            # if i == "1":
            #     cv2.imwrite(f"../dataset/val/{name}.jpg", roi_bgr)
            # if i == "2":
            #     cv2.imwrite(f"../dataset/test/{name}.jpg", roi_bgr)
            # else:
            #     cv2.imwrite(f"../dataset/train/{name}.jpg", roi_bgr)
    else:
        print("No hand detected")
        return None

def crop_palm_roi(image, landmarks, roi_size=224):
    h, w, _ = image.shape

    def to_xy(lm):
        return np.array([lm.x * w, lm.y * h])

    L0 = to_xy(landmarks[0])
    L5 = to_xy(landmarks[5])
    L9 = to_xy(landmarks[9])
    L17 = to_xy(landmarks[17])

    # STEP 1: vector L5L17
    v = L17 - L5
    s = np.linalg.norm(v)

    # STEP 2: normals
    n1 = np.array([-v[1], v[0]])
    n2 = np.array([v[1], -v[0]])

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # STEP 3: reference point
    Lref = (L0 + L9) / 2

    # STEP 4: midpoint
    M = (L5 + L17) / 2

    P1 = M + s * n1
    P2 = M + s * n2

    if np.linalg.norm(P1 - Lref) < np.linalg.norm(P2 - Lref):
        n = n1
    else:
        n = n2

    # STEP 5: ROI corners
    Qsrc = np.array([
        L5,
        L17,
        L17 + s * n,
        L5 + s * n
    ], dtype=np.float32)

    # STEP 6: destination square
    S = roi_size
    Qdst = np.array([
        [0, 0],
        [S-1, 0],
        [S-1, S-1],
        [0, S-1]
    ], dtype=np.float32)

    # STEP 7: perspective transform
    M = cv2.getPerspectiveTransform(Qsrc, Qdst)

    # STEP 8: warp ROI
    roi = cv2.warpPerspective(
        image,
        M,
        (S, S),
        flags=cv2.INTER_CUBIC
    )
    return roi

def stack_folder(images_path1, images_path2):
    left_image = os.listdir(images_path1)
    right_image = os.listdir(images_path2)
    images = []
    labels = []
    for image in left_image:
        label = image.split(".")[0] + "_l"
        images.append(os.path.join(images_path1, image))
        labels.append(label)
    for image in right_image:
        label = image.split(".")[0] + "_r"
        images.append(os.path.join(images_path2, image))
        labels.append(label)
    return images, labels

if "__main__" == __name__:
    # image_path = "D://Projects//Study//SelfStudy//CV//IITD Palmprint V1//Test/007_1.jpg"
    image_path = "D://Projects//Study//School//Nam3_Ky2//PBL5//pbl_v3//storage//2026//03//17//session_1773761220235//img_01.jpg"
    img_original = Image.open(image_path)


    # augmentation chung
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]

    # 3 pipeline
    pipeline_A = base_transforms + [
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
    ]

    pipeline_B = base_transforms + [
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        SharpenTransform(p=1.0)
    ]

    pipeline_C = base_transforms + [
        transforms.Grayscale(num_output_channels=1),
        SharpenTransform(p=1.0),
        CLAHETransform(),
    ]

    pipelines = [
        ("Grayscale → CLAHE", pipeline_A),
        ("Grayscale → CLAHE → Sharpen", pipeline_B),
        ("Grayscale → Sharpen → CLAHE", pipeline_C)
    ]

    plt.figure(figsize=(12, 10))

    for row, (title, pipeline) in enumerate(pipelines):

        img = img_original.copy()

        plt.subplot(3, len(pipeline) + 1, row * (len(pipeline) + 1) + 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        for i, t in enumerate(pipeline):

            img = t(img)

            plt.subplot(3, len(pipeline) + 1, row * (len(pipeline) + 1) + i + 2)

            if isinstance(img, torch.Tensor):
                show_img = img.permute(1, 2, 0).numpy()
                if show_img.shape[2] == 1:
                    show_img = show_img[:, :, 0]
                    plt.imshow(show_img, cmap="gray")
                else:
                    plt.imshow(show_img)
            else:
                plt.imshow(img, cmap="gray")

            plt.title(type(t).__name__)
            plt.axis("off")

        plt.text(0, -40, title, fontsize=12)

    plt.tight_layout()
    plt.show()
