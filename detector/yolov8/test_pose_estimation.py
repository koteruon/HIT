import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# global variables
SAVE_DIR = "detector/yolov8/runs/detect/2023niag_0003"
VIDEO_PATH = "/home/chaoen/yoloNhit_calvin/yolov7/inference/videos_serve/2023niag_shot_one/2023niag_0003.mp4"
OUTPUT_VIDEO_NAME = os.path.basename(VIDEO_PATH)

# Set up save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load a model
model = YOLO("detector/yolov8/weights/yolov8x-pose.pt")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up video writer
output_path = os.path.join(SAVE_DIR, f"{OUTPUT_VIDEO_NAME}_with_keypoints.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process results generator
ret, frame = cap.read()
assert ret is not False and ret is not None

for _ in tqdm(range(frame_count), desc="Processing video", unit="frame"):
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(frame, verbose=False)[0]
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    frame_with_key_points = result.plot()
    frame_with_key_points = cv2.cvtColor(frame_with_key_points, cv2.COLOR_RGB2BGR)
    out.write(frame_with_key_points)
    ret, frame = cap.read()

cap.release()
out.release()
print("Video processing completed and saved to:", output_path)
