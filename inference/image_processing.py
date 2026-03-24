import cv2
import numpy as np
from PIL import Image

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

def is_palm_open(hand_landmarks, threshold_ratio=0.15):
    lm = hand_landmarks.landmark

    # Định nghĩa mapping cho các ngón
    fingers = {
        'thumb': {'tip': 4, 'pip': 3, 'mcp': 2, 'type': 'x'},
        'index': {'tip': 8, 'pip': 6, 'mcp': 5, 'type': 'y'},
        'middle': {'tip': 12, 'pip': 10, 'mcp': 9, 'type': 'y'},
        'ring': {'tip': 16, 'pip': 14, 'mcp': 13, 'type': 'y'},
        'pinky': {'tip': 20, 'pip': 18, 'mcp': 17, 'type': 'y'}
    }

    # Lấy tọa độ cổ tay để tham chiếu
    wrist_y = lm[0].y
    wrist_x = lm[0].x

    # Tính khoảng cách tham chiếu (bề rộng lòng bàn tay)
    palm_width = abs(lm[17].x - lm[5].x)

    for finger_name, config in fingers.items():
        tip = lm[config['tip']]
        pip = lm[config['pip']]
        mcp = lm[config['mcp']]

        if config['type'] == 'x':  # Ngón cái
            # Kiểm tra ngón cái mở (dang ra ngoài)
            if abs(tip.x - mcp.x) < palm_width * threshold_ratio:
                return False, f"{finger_name} not open"
        else:  # Các ngón khác
            # Kiểm tra ngón tay duỗi thẳng
            finger_length = abs(tip.y - mcp.y)
            if finger_length < palm_width * threshold_ratio:
                return False, f"{finger_name} not open"

            # Kiểm tra độ cong
            if tip.y > pip.y - 0.02:  # Ngón tay cong lên
                return False, f"{finger_name} bent"
    return True, "Palm is open"

def is_palm_large_enough(hand_landmarks, image_shape, min_ratio=0.4, min_pixel_height=180):
    h, w = image_shape[:2]
    pts = np.array([
        (lm.x * w, lm.y * h)
        for lm in hand_landmarks.landmark
    ])

    wrist = pts[0]
    middle_base = pts[9]

    palm_height = np.linalg.norm(wrist - middle_base)

    ratio = palm_height / h

    if ratio < min_ratio:
        return False, "Palm too small (ratio)"

    if palm_height < min_pixel_height:
        return False, "Palm too small (pixel)"

    return True, "OK"

def crop_palm_roi(image, hand_landmarks, roi_size=224):
    h, w, _ = image.shape
    def to_xy(lm):
        return np.array([lm.x * w, lm.y * h])

    landmarks = hand_landmarks.landmark

    L0 = to_xy(landmarks[0])
    L5 = to_xy(landmarks[5])
    L9 = to_xy(landmarks[9])
    L17 = to_xy(landmarks[17])

    v = L17 - L5
    s = np.linalg.norm(v)

    n1 = np.array([-v[1], v[0]])
    n2 = np.array([v[1], -v[0]])

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    Lref = (L0 + L9) / 2
    M = (L5 + L17) / 2

    P1 = M + s * n1
    P2 = M + s * n2

    if np.linalg.norm(P1 - Lref) < np.linalg.norm(P2 - Lref):
        n = n1
    else:
        n = n2

    Qsrc = np.array([
        L5,
        L17,
        L17 + s * n,
        L5 + s * n
    ], dtype=np.float32)

    S = roi_size

    Qdst = np.array([
        [0, 0],
        [S-1, 0],
        [S-1, S-1],
        [0, S-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(Qsrc, Qdst)

    roi = cv2.warpPerspective(
        image,
        M,
        (S, S),
        flags=cv2.INTER_CUBIC
    )

    return roi
