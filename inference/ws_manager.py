import threading

class WSManager:
    def __init__(self):
        self.ws = None
        self.lock = threading.Lock()

    def set(self, ws):
        with self.lock: # chỉ 1 thread được vào tại 1 thời điểm, các thread khác phải chờ
            if self.ws and self.ws != ws:
                print("[WSManager] Existing connection found -> closing old one")
                try:
                    self.ws.close() # đóng kết nối trước
                except Exception as e:
                    print(f"[WSManager] Error closing old: {e}")
            self.ws = ws

    def get(self):
        return self.ws

    def send(self, data):
        with self.lock:
            ws = self.ws # copy ra để tránh locking quá lâu
        if ws:
            try:
                ws.send(data)
            except:
                self.ws = None

    def clear(self, ws):
        with self.lock:
            if self.ws == ws:  # check đúng connection
                self.ws = None

ws_manager = WSManager()