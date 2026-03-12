import os
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

def detect_hand(image_path, name):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            roi = crop_palm_roi(image_rgb, hand.landmark, 224)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            i = name.split("_")[1]
            if i == "1":
                cv2.imwrite(f"../dataset/val/{name}.jpg", roi_bgr)
            if i == "2":
                cv2.imwrite(f"../dataset/test/{name}.jpg", roi_bgr)
            else:
                cv2.imwrite(f"../dataset/train/{name}.jpg", roi_bgr)
    else:
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
    os.makedirs("../dataset/train", exist_ok=True)
    os.makedirs("../dataset/val", exist_ok=True)
    os.makedirs("../dataset/test", exist_ok=True)
    paths, lb = stack_folder(
        images_path1="D:\\Projects\\Study\\SelfStudy\\CV\\IITD Palmprint V1\\Left Hand",
        images_path2="D:\\Projects\\Study\\SelfStudy\\CV\\IITD Palmprint V1\\Right Hand"
    )
    for path, label in zip(paths, lb):
        detect_hand(path, label)
    print("done n")