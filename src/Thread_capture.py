import cv2
import time
import logging
import threading
from queue import  Full

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ThreadCapture(threading.Thread):
    def __init__(self, video_path, output_queue, max_buffer_size=12):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        self.output_queue = output_queue
        self.timestamps = []
        self.max_buffer_size = max_buffer_size

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("[ThreadCapture] Video hết, quay lại từ đầu.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                if self.output_queue.qsize() < self.max_buffer_size:
                    self.output_queue.put((frame))
                else:
                    logging.warning("[ThreadCapture] Queue đầy, bỏ frame để tránh lag.")
            except Full:
                pass

            time.sleep(0.02)

        self.cap.release()

    def stop(self):
        self.running = False
        logging.info("[ThreadCapture] Dừng luồng ghi hình.")