import logging
from queue import Queue
from PyQt5.QtWidgets import QMainWindow
from config import model_path, video_path, model_path1
from Thread_stream import StreamThread
from Thread_capture import ThreadCapture
from Thread_Segment import Segment
from Thread_detect import Detect_P

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.capture_queue = Queue(maxsize=10)
        self.segment_queue = Queue(maxsize=10)

        # Khởi tạo các luồng
        self.capture = ThreadCapture(video_path, self.capture_queue)
        self.detect = Detect_P(self.capture_queue, model_path)
        self.segment = Segment(self.detect.output_stream(), model_path1)
        self.stream_thread = StreamThread(self.segment.output_stream())
        
        # Khởi động luồng
        self.capture.start()
        self.detect.start()
        self.segment.start()
        self.stream_thread.start()

    def closeEvent(self, event):
        # Dừng tất cả các luồng khi đóng ứng dụng
        self.capture.stop()
        self.detect.stop()
        self.segment.stop()
        self.stream_thread.stop()
        event.accept()
