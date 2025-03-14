import cv2
import time
import logging
import threading
from queue import Queue, Empty
from ultralytics import YOLO
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Segment(threading.Thread):
    def __init__(self, input_queue, model_path, process_every_n_frames=1):
        super().__init__()
        self.model = YOLO(model_path)
        self.running = True
        self.input_queue = input_queue
        self.output_queue = Queue()
        self.frame_counter = 0
        self.process_every_n_frames = process_every_n_frames

    def run(self):
        while self.running:
            try:
                frame, objects_info = self.input_queue.get(timeout=0.5)
                if not objects_info:
                    continue

                overlay = frame.copy()
                self.frame_counter += 1

                for (x1, y1, x2, y2, track_id) in objects_info:
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        logging.warning(f"Bounding box ngoài phạm vi: {x1, y1, x2, y2}")
                        continue

                    obj_crop = frame[y1:y2, x1:x2]
                    if obj_crop.size == 0:
                        logging.warning(f"Ảnh crop bị lỗi: {x1, y1, x2, y2}")
                        continue

                    if self.frame_counter % self.process_every_n_frames == 0:
                        results = self.model(obj_crop)
                        frame = self.draw_results(frame, results, (x1, y1, x2, y2, track_id), overlay)
                self.output_queue.put(frame)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Lỗi trong Segment: {e}")
            time.sleep(0.01)

    def draw_results(self, frame, results, bbox, overlay):
        x1, y1, x2, y2, track_id = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label_y = y1 - 10 if y1 - 10 > 0 else y2 + 20  
        cv2.putText(frame, f"ID: {track_id}", (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 2, cv2.LINE_AA)

        if results[0].masks is None:
            return frame

        for mask in results[0].masks.xy:
            points = [np.array(mask, dtype=np.int32) + (x1, y1)]
            cv2.fillPoly(overlay, points, color=(0, 255, 0))
            cv2.polylines(frame, points, isClosed=True, color=(0, 0, 255), thickness=2)
        
        cv2.addWeighted(overlay, 0, frame, 1, 0, frame)
        return frame

    def output_stream(self):
        return self.output_queue

    def stop(self):
        self.running = False
        logging.info("Dừng xử lý video.")
        self.model = None
