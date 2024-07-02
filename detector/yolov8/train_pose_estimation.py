import os

from ultralytics import YOLO

results = model.train(
    model="/home/chaoen/yoloNhit_calvin/HIT/detector/yolov8/weights/yolov8x-pose.pt",
    data="/home/chaoen/yoloNhit_calvin/HIT/detector/yolov8/config/coco8-pose.yaml",
    epochs=300,
    imgsz=960,
    project="detector/yolov8/runs/train",
    name="naig_pose",
)
