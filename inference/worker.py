import json

import cv2
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from .image_processing import CLAHETransform
from .dao import connect_database, deactivate_active_sessions, add_session, get_available_locker, get_active_session
from .locker import send_locker
from src.palm_net import PalmNet
from .config import MODEL_PATH

conn = connect_database()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_model():
    print("[WORKER] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[WORKER] Model loaded")
    return model, device

def run_model(model, device, image_input):
    # nếu là path thì đọc ảnh
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Cannot read image: {image_input}")
    else:
        image = image_input

    # preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = inference_transform(image)

    # thêm batch dimension
    image = image.unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        embedding = model(image)

    return embedding.cpu().numpy().flatten()

def save_to_db(session_id, embedding):
    lockers = get_available_locker(conn)
    print("[WORKER] Available lockers: ", lockers)
    if not lockers:
        return None
    locker_id = lockers[0][0]
    embedding = embedding.tolist()
    add_session(conn, session_id, locker_id, palm_hash=json.dumps(embedding))
    return locker_id

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_embeddings(query_embedding):
    best_score = -1
    best_locker = None
    mean_embeddings = get_active_session(conn)

    for item in mean_embeddings:
        emd = json.loads(item[2])
        emd = np.array(emd)
        locker_id = item[4]
        score = cosine_similarity(query_embedding, emd)

        if score > best_score:
            best_score = score
            best_locker = locker_id

    if best_score < 0.8:
        return None, -1
    if best_locker is not None:
        deactivate_active_sessions(conn, best_locker)
    return best_locker, best_score


def worker_loop(send_queue, take_queue):
    print("[WORKER] starting...")
    model, device = load_model()
    print("[WORKER] Ready")

    session_embeddings = {}
    take_embeddings = []

    while True:
        # ===== REGISTER (send) =====
        if not send_queue.empty():
            image_path = send_queue.get()
            if image_path is None:
                take_embeddings = []
                session_embeddings = {}
                continue
            session_id = os.path.basename(os.path.dirname(image_path))
            embedding = run_model(model, device, image_path)
            if session_id not in session_embeddings:
                session_embeddings[session_id] = []

            session_embeddings[session_id].append(embedding)

            if len(session_embeddings[session_id]) == 5:
                emb = np.array(session_embeddings[session_id])
                mean_embedding = emb.mean(axis=0)
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                lock_id = save_to_db(session_id, mean_embedding)
                send_locker(lock_id)
                print(f"[WORKER] Lock: {lock_id}")
                del session_embeddings[session_id]
                print(f"[WORKER] Registered {session_id}")
                print(session_embeddings)

        # ===== VERIFY (take) =====
        if not take_queue.empty():
            image = take_queue.get()
            if image is None:
                take_embeddings = []
                session_embeddings = {}
                continue
            embedding = run_model(model, device, image)
            take_embeddings.append(embedding)
            print(f"[WORKER] collected {len(take_embeddings)}/2 embeddings")
            if len(take_embeddings) == 2:
                embeddings = np.array(take_embeddings)
                mean_embedding = embeddings.mean(axis=0)
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                lock_id, best_score = compare_embeddings(mean_embedding)
                if lock_id is not None:
                    send_locker(lock_id)
                print(f"[WORKER] Locker, Best_score: {lock_id}, {best_score}")
                take_embeddings.clear()