import cv2
import os
import datetime
import yaml
import logging

def load_config(config_path):
    """Đọc file cấu hình YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logger(log_dir):
    """Thiết lập logger để ghi log."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"motion_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    return logging.getLogger()

def get_output_writer(config, frame_height, timestamp):
    """Tạo VideoWriter để ghi video định dạng AVI."""
    output_dir = config['video']['output_dir']
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError(f"Đường dẫn '{output_dir}' tồn tại nhưng không phải thư mục. Vui lòng xóa hoặc đổi tên tệp.")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"motion_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter(
        output_path,
        fourcc,
        config['video']['fps'],
        (config['video']['frame_width'], frame_height)
    ), output_path

def initialize_capture(config, video_path=None):
    """Khởi tạo nguồn video hoặc camera."""
    if config['video']['source'] == 'video' or video_path:
        path = video_path if video_path else config['video']['input_path']
        if not os.path.exists(path):
            raise ValueError(f"File video không tồn tại: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {path}. Kiểm tra định dạng (.mp4, .avi, .wmv, .mkv, .vob, .flv) hoặc cài FFmpeg.")
        return cap, f"Video: {os.path.basename(path)}"
    else:
        cap = cv2.VideoCapture(config['video']['camera_id'])
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                raise ValueError("Không thể mở camera với ID 0 hoặc 1")
        return cap, f"Camera ID: {config['video']['camera_id']}"