from ultralytics.trackers.bot_sort import BOTSORT
import threading
import time
import logging
import numpy as np
from queue import Queue, Empty
from types import SimpleNamespace as Namespace
from ultralytics import YOLO

class Detect_P(threading.Thread):
    def __init__(self, input_queue, model_path, process_every_n_frames=5):
        super().__init__()
        self.model = YOLO(model_path, task="detect")
        self.running = True
        self.input_queue = input_queue
        self.output_queue = Queue()
        self.frame_counter = 0
        self.process_every_n_frames = process_every_n_frames
        self.lock = threading.Lock()
        
        args = Namespace(
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            gmc_method='sparseOptFlow',
            with_reid=True,
            reid_model='osnet_x1_0',
            fuse_score=True,
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.7,
            track_buffer=400,
            match_thresh=0.95,
            aspect_ratio_thresh=1.6,
            min_box_area=10,
            mot20=False
        )
        self.tracker = BOTSORT(args)

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.5)
                self.frame_counter += 1
                
                if self.frame_counter % self.process_every_n_frames == 0:
                    results = self.model(frame)
                    
                    if not results or len(results) == 0:
                        continue
                    
                    objects_info = self.extract_objects(results)
                    self.output_queue.put((frame, objects_info))

            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in Detect_P: {e}", exc_info=True)

            time.sleep(0.01)

    def extract_objects(self, results):
        """Extracts bounding box and tracking ID information."""
        objects_info = []
        boxes, confs, clss = [], [], []

        for r in results:
            if r.boxes:
                boxes.extend([box.cpu().numpy() for box in r.boxes.xywh])
                confs.extend([conf.cpu().numpy() for conf in r.boxes.conf])
                clss.extend([cls.cpu().numpy() for cls in r.boxes.cls])

        if len(boxes) == 0:
            return objects_info  

        boxes = np.array(boxes)
        confs = np.array(confs)
        clss = np.array(clss)

        keep = (clss == 0) & (confs > 0.6)  
        if not np.any(keep):
            return objects_info

        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]

        detections = Namespace(
            xywh=boxes,
            conf=confs,
            cls=clss
        )

        tracks = self.tracker.update(detections)  

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            objects_info.append((x1, y1, x2, y2, track_id))

        return objects_info

    def output_stream(self):
        return self.output_queue

    def stop(self):
        self.running = False
        logging.info("Stopping video processing.")
        self.join()
        self.model = None
