import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time
from src.utils import load_config, setup_logger, get_output_writer, initialize_capture
from src.detector import MotionDetector

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detection Software")
        self.config = load_config("config/settings.yaml")
        self.logger = setup_logger(self.config['log']['log_dir'])
        self.cap = None
        self.out = None
        self.detector = None
        self.running = False
        self.motion_detected = False
        self.frame_width = self.config['video']['frame_width']
        self.frame_height = None
        self.motion_start_time = None
        self.latest_snapshot = None

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg="black")
        self.canvas.pack(side=tk.LEFT)

        self.snapshot_label = tk.Label(self.main_frame, text="Latest Snapshot", width=20)
        self.snapshot_label.pack(side=tk.RIGHT, padx=10)
        self.snapshot_canvas = tk.Canvas(self.main_frame, width=160, height=120, bg="black")
        self.snapshot_canvas.pack(side=tk.RIGHT)

        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack()

        self.fps_label = tk.Label(root, text="FPS: 0.00", font=("Arial", 12))
        self.fps_label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.open_camera_btn = tk.Button(self.button_frame, text="Open Camera", command=self.open_camera)
        self.open_camera_btn.pack(side=tk.LEFT, padx=5)

        self.upload_video_btn = tk.Button(self.button_frame, text="Upload Video", command=self.upload_video)
        self.upload_video_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(self.button_frame, text="Reset Background", command=self.reset_background, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(self.button_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.start_time = time.time()
        self.frame_count = 0

    def open_camera(self):
        if not self.running:
            self.config['video']['source'] = 'camera'
            self.start_processing()

    def upload_video(self):
        if not self.running:
            file_path = filedialog.askopenfilename(
                initialdir="data/",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.wmv *.mkv *.vob *.flv"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.config['video']['source'] = 'video'
                try:
                    self.start_processing(video_path=file_path)
                    messagebox.showinfo("Success", f"Video {os.path.basename(file_path)} loaded successfully!")
                except ValueError as e:
                    messagebox.showerror("Error", str(e))
                    self.logger.error(str(e))

    def reset_background(self):
        if self.detector:
            self.detector.reset_background()
            self.logger.info("Background reset")

    def start_processing(self, video_path=None):
        try:
            self.cap, source_info = initialize_capture(self.config, video_path)
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.frame_width / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.detector = MotionDetector(self.config)
            self.running = True
            self.open_camera_btn.config(state=tk.DISABLED)
            self.upload_video_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.logger.info("Started processing: %s", source_info)
            self.status_label.config(text=f"Status: {source_info}")
            self.process_frames()
        except ValueError as e:
            self.logger.error(str(e))
            self.status_label.config(text=f"Error: {str(e)}")
            raise

    def process_frames(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.logger.info("Video ended")
            messagebox.showinfo("Info", "Video has ended")
            self.stop()
            return

        if self.config['video']['source'] == 'camera':
            frame = cv2.flip(frame, 1)

        processed_frame, text, motion_detected_now, object_count, yolo_results = self.detector.process_frame(frame, self.frame_width)

        if motion_detected_now:
            if self.motion_start_time is None:
                self.motion_start_time = time.time()
            elif time.time() - self.motion_start_time >= self.config['detector']['motion_duration_threshold'] and self.out is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.out, output_path = get_output_writer(self.config, self.frame_height, timestamp)
                self.logger.info("Bắt đầu ghi video: %s", output_path)
                self.motion_detected = True
        else:
            self.motion_start_time = None

        if motion_detected_now:
            snapshot_path = os.path.join(self.config['video']['output_dir'], f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_path, processed_frame)
            self.logger.info("Lưu ảnh chụp: %s", snapshot_path)
            self.update_snapshot(snapshot_path)

        if self.out is not None:
            self.out.write(processed_frame)

        if text == "Normal" and self.motion_detected:
            self.motion_detected = False
            if self.out is not None:
                self.out.release()
                self.out = None
                self.logger.info("Đã dừng ghi video")

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        yolo_text = ", ".join([name for name, _ in yolo_results]) if yolo_results else "None"
        self.status_label.config(text=f"Status: {text}, Objects: {object_count}, YOLO: {yolo_text}")
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {fps:.2f}")

        if self.running:
            self.root.after(10, self.process_frames)

    def update_snapshot(self, snapshot_path):
        """Hiển thị ảnh chụp mới nhất."""
        try:
            img = Image.open(snapshot_path)
            img = img.resize((160, 120), Image.Resampling.LANCZOS)
            self.latest_snapshot = ImageTk.PhotoImage(img)
            self.snapshot_canvas.create_image(0, 0, anchor=tk.NW, image=self.latest_snapshot)
        except Exception as e:
            self.logger.error("Không thể hiển thị ảnh chụp: %s", str(e))

    def stop(self):
        self.running = False
        if self.out is not None:
            self.out.release()
            self.out = None
        if self.cap is not None:
            self.cap.release()
        self.canvas.delete("all")
        self.snapshot_canvas.delete("all")
        self.status_label.config(text="Status: Idle")
        self.fps_label.config(text="FPS: 000")
        self.open_camera_btn.config(state=tk.NORMAL)
        self.upload_video_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.motion_start_time = None
        self.logger.info("Đã dừng xử lý")

    def on_closing(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()