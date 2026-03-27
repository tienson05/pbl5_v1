from queue import Empty
import cv2
import numpy as np
from flask import Flask, request
from flask_sock import Sock
import os
import time
import threading
from datetime import datetime
from inference.dao import dao
from inference.ws_manager import ws_manager
from inference.image_processing import is_palm_open, is_palm_large_enough, crop_palm_roi
from .detect import detect_hand
from .worker import worker_loop
from multiprocessing import Process, Queue

app = Flask(__name__)
sock = Sock(app)

#config
UPLOAD_FOLDER = "storage"
ALLOWED_COMMANDS = {"take", "send"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
mode = None

TIMEOUT = 15  # giây
start_time = None

MAX_FRAMES = 5
frame_queue = Queue(maxsize=30)
send_queue = Queue(maxsize=30)
take_queue = Queue(maxsize=30)
ws_queue = Queue(maxsize=10)
counter = 0
take_counter = 0
invalid_counter = 0
MAX_TAKE_FRAMES = 2
current_session_dir = None
session_timestamp = None

"""
Save embeddings into database -> trừ trường hợp mất điện
Sau 15 giây không chụp xong 5 ảnh thì send fall xuống esp32 cam (fail)
Thay đổi trong file locker_manager trước khi chạy
"""

def keyboard_loop():
    print("=== Keyboard control ready ===")
    print("Commands:", ", ".join(ALLOWED_COMMANDS))
    while True:
        try:
            cmd = input(">>> ").strip().lower()
        except EOFError:
            break
        if cmd not in ALLOWED_COMMANDS:
            print("Invalid command")
            continue

        if ws_manager.get() is None:
            print("ESP32CAM not connected")
            continue
        if cmd == "send":
            start_new_session()
        ws_manager.send(cmd)
        global mode
        mode = cmd
        print(f"[KEYBOARD → ESP32CAM] {cmd}")

def ws_sender():
    while True:
        data = ws_queue.get()

        ws = ws_manager.get()
        if not ws:
            continue

        msg = "done" if data == 1 else "fail"
        ws_manager.send(msg)
        print(f"[WS_Sender] Sent: {msg}")

def start_new_session():
    global current_session_dir, session_timestamp, counter

    now = datetime.now()

    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    session_timestamp = int(time.time()*1000)
    counter = 0

    # tạo các path
    year_path = os.path.join(UPLOAD_FOLDER, year)
    month_path = os.path.join(year_path, month)
    day_path = os.path.join(month_path, day)

    # đảm bảo từng cấp thư mục tồn tại
    os.makedirs(year_path, exist_ok=True)
    os.makedirs(month_path, exist_ok=True)
    os.makedirs(day_path, exist_ok=True)

    # tạo thư mục session
    session_dir = f"session_{session_timestamp}"
    session_path = os.path.join(day_path, session_dir)

    os.makedirs(session_path, exist_ok=True)
    current_session_dir = session_path
    print(f"[SESSION] New session created: {session_path}")

@app.route("/event", methods=["GET"])
def http_command():
    global start_time
    cmd = request.args.get("command").strip().lower()
    if cmd not in ALLOWED_COMMANDS:
        return {"error": "invalid command"}, 400

    if ws_manager.get() is None:
        return {"error": "ESP32CAM not connected"}, 503
    if cmd == "send":
        lockers = dao.get_available_locker()
        if not lockers:
            ws_manager.send("full")
            return {"locker": None}, 400
        start_new_session()
    if cmd == "take":
        lockers = dao.get_active_session()
        if not lockers:
            ws_manager.send("fail")
            return {"locker": None}, 400
    ws_manager.send(cmd)
    global mode
    mode = cmd
    start_time = time.time()
    print(f"[Server → ESP32CAM] {cmd}")
    return {"status": "ok"}

def writer_worker():
    global counter, take_counter, current_session_dir, invalid_counter, start_time, mode
    while True:
        jpeg = frame_queue.get()
        np_arr = np.frombuffer(jpeg, np.uint8) # convert bytes -> numpy array
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode jpeg -> BGR image
        ok, msg1, hand = detect_hand(frame) #ktra là bàn tay

        # Time vượt quá TIMEOUT mà chưa phát hiện được bàn tay nào hợp lệ thì dừng
        current_time = time.time()
        if start_time is not None and (current_time - start_time) >= TIMEOUT:
            if invalid_counter > 30:
                if ws_manager.get() is not None:
                    ws_manager.send("fail")
                start_time = None
                invalid_counter = 0

        if not ok:
            invalid_counter += 1
            continue
        is_open, msg2 = is_palm_open(hand) # ktra tay mở
        if not is_open:
            invalid_counter += 1
            continue
        is_enough, msg3 = is_palm_large_enough(hand, frame.shape) # ktra tay đủ lớn
        if not is_enough:
            invalid_counter += 1
            continue
        roi = crop_palm_roi(frame, hand, roi_size=224)

        invalid_counter = 0

        if mode == "take":
            take_queue.put(roi)
            take_counter += 1
            print(f"[TAKE]: received {take_counter} frames")
            if take_counter >= MAX_TAKE_FRAMES:
                if ws_manager.get() is not None:
                    ws_manager.send("wait")
                take_queue.put(None)
                mode = None
                print("[TAKE] Sent 'done' after 2 frames")
                take_counter = 0
        elif mode == "send":
            if current_session_dir is None:
                continue

            counter += 1
            filename = os.path.join(current_session_dir, f"img_{counter:02d}.jpg")

            try:
                cv2.imwrite(filename, roi)
                send_queue.put(filename) # cân nhắc put roi như take luôn
                print(f"[WRITER] Saved {filename} ({len(roi)} bytes)")
            except Exception as e:
                print(f"[WRITER] Error saving {filename}: {e}")

            if counter >= MAX_FRAMES:
                if ws_manager.get() is not None:
                    ws_manager.send("done")
                    counter = 0
                    print(f"[SESSION] Completed {current_session_dir}")
                    current_session_dir = None
                    print("[WRITER] Sent 'done' after 5 frames")
                send_queue.put(None)
                mode = None

def reset_all_state(reason=""):
    """Reset tất cả state về ban đầu"""
    global frame_queue, send_queue, take_queue

    print(f"\n{'=' * 50}")
    print(f"Reason: {reason}")

    # Reset frame queue
    frame_dropped = 0
    try:
        while True:
            frame_queue.get_nowait()
            frame_dropped += 1
    except Empty:
        pass
    print(f"Xóa {frame_dropped} frames trong queue")

    # Reset send queue
    send_dropped = 0
    try:
        while True:
            send_queue.get_nowait()
            send_dropped += 1
    except Empty:
        pass
    print(f"Xóa {send_dropped} items trong send queue")

    # Reset take queue
    take_dropped = 0
    try:
        while True:
            take_queue.get_nowait()
            take_dropped += 1
    except Empty:
        pass
    print(f"Xóa {take_dropped} items trong take queue")
    print(f"Hoàn tất! Queue sizes: frame={frame_queue.qsize()}, "f"send={send_queue.qsize()}, take={take_queue.qsize()}")
    print(f"{'=' * 50}\n")

@sock.route("/ws")
def esp32_socket(ws):
    ws_manager.set(ws)

    print(f"[Server] ESP32 connected (new) | Queue size: {frame_queue.qsize()}")

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            if isinstance(data, str):
                print(f"[ESP32CAM → TEXT] {data}")
                continue

            if isinstance(data, bytes):
                jpeg = data
                dropped = False
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        dropped = True
                        print("[Server] Queue full → dropped oldest frame")
                    except:
                        pass
                frame_queue.put(jpeg)
                print(f"[Server] Received JPEG: {len(jpeg)} bytes | Queue size: {frame_queue.qsize()}" + (" (dropped old)" if dropped else ""))
            else:
                print(f"[WS] Unknown data type: {type(data)}")

    except Exception as e:
        print(f"[WS] Error in receive loop: {e}")
    finally:
        ws_manager.clear(ws)
        print("[Server] ESP32CAM disconnected")
        reset_all_state("client disconnected")

if __name__ == "__main__":
    dao.connect_database()
    worker = Process(target=worker_loop, args=(send_queue, take_queue, ws_queue,))
    worker.daemon = True
    worker.start()
    print("worker pid:", worker.pid)

    threading.Thread(target=keyboard_loop, daemon=True).start()
    threading.Thread(target=writer_worker, daemon=True).start()
    threading.Thread(target=ws_sender, daemon=True).start()
    print("Server running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)