import os

import torch
from ultralytics import YOLO

from utils.general import bbox_iou

# Load a model
model = YOLO("detector/yolov8/weights/yolov8x-pose.pt")

dataset_path = "detector/yolov8/datasets/niag"

train_images_path = os.path.join(dataset_path, "train/images")
train_labels_path = os.path.join(dataset_path, "train/labels")
valid_images_path = os.path.join(dataset_path, "valid/images")
valid_labels_path = os.path.join(dataset_path, "valid/labels")

train_images = os.listdir(train_images_path)
train_labels = os.listdir(train_labels_path)
valid_images = os.listdir(valid_images_path)
valid_labels = os.listdir(valid_labels_path)

for image in train_images:
    image_filename, image_extension = os.path.splitext(image)[0]
    with open(os.path.join(train_images_path, image_filename + ".txt"), "r") as f:
        bboxs = f.readlines()
    target_line = []
    target_bbox = []
    for bbox in bboxs:
        clz, x, y, w, h, conf = map(float, bbox.split())
        if 1 == int(clz):
            target_bbox.append(" ".join([x, y, w, h]))
    target_bbox = torch.tensor(target_bbox)

    results = model.predict(image, verbose=False)
    for result in results:
        pose_bbox = torch.tensor(result.boxes)
        keypoints = result.keypoints
        iou = bbox_iou(pose_bbox, target_bbox, x1y1x2y2=True, CIoU=True)
