import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
)

def detect_hand(image):
    if image is None:
        return False, "No image", None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return False, "No hand detected", None

    hand = results.multi_hand_landmarks[0]
    return True, "OK", hand