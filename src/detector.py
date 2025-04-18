import cv2
import imutils
import datetime
from ultralytics import YOLO

class MotionDetector:
    def __init__(self, config):
        self.min_area = config['detector']['min_contour_area']
        self.blur_size = config['detector']['blur_size']
        self.threshold_value = config['detector']['threshold_value']
        self.dilate_iterations = config['detector']['dilate_iterations']
        self.first_frame = None
        self.prev_frame = None
        self.yolo_model = YOLO(config['yolo']['model_path'])
        self.motion_start_time = None
        self.motion_duration_threshold = config['detector']['motion_duration_threshold']

    def process_frame(self, frame, frame_width):
        """Xử lý khung hình để phát hiện chuyển động."""
        frame = imutils.resize(frame, width=frame_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self.first_frame is None:
            self.first_frame = gray
            self.prev_frame = gray
            return frame, "Normal", False, 0, []

        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=self.dilate_iterations)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        text = "Normal"
        motion_detected = False
        object_count = 0
        yolo_results = []

        # YOLO object detection
        results = self.yolo_model(frame, conf=0.5)
        for result in results:
            for box in result.boxes:
                x, y, w, h = map(int, box.xywh[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                yolo_results.append((class_name, (x - w//2, y - h//2, w, h)))
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 255), 2)
                cv2.putText(frame, class_name, (x - w//2, y - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving Object detected"
            motion_detected = True
            object_count += 1

        self.prev_frame = gray.copy()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, text, motion_detected, object_count, yolo_results

    def reset_background(self):
        """Reset nền để cập nhật lại."""
        self.first_frame = None
        self.prev_frame = None

