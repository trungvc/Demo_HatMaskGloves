import threading
import cv2
import time
import logging
from queue import Empty

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class StreamThread(threading.Thread):
    def __init__(self, queue_stream):
        super().__init__()
        self.queue_stream = queue_stream
        self.running = True
        self.prev_time = time.time()
        self.fps = 0.0

    def run(self):
        while self.running:
            try:
                frame = self.queue_stream.get(timeout=0.5)
                if frame is None:
                    logging.warning("[StreamThread] Nhận frame rỗng, bỏ qua.")
                    continue
                self.display_frame(frame)
            except Empty:
                continue
            time.sleep(0.01)

    def display_frame(self, frame):
        cv2.imshow("Segmented Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        logging.info("[StreamThread] Đã dừng.")
