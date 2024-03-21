import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


class KeyPointDetection:
    def __init__(self, is_train=False):
        # 輸出的結果
        self.all_outputs = []

        # load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load("weights/yolov7-w6-pose.1.pt", map_location=self.device)
        self.model = weigths["model"]
        _ = self.model.float().eval()
        if torch.cuda.is_available():
            self.model.half().to(self.device)

        # 輸出檔案位置
        if is_train:
            self.root = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/keyframes/train/"
            self.json_path = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/annotations/table_tennis_train_person_bbox_kpts.json"
        else:
            self.root = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/keyframes/test/"
            self.json_path_pattern = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/annotations/table_tennis_test_person_bbox_kpts.json"

    def detect_one(self, timestamp, root_idx=0):
        im = cv2.imread(os.path.join(self.root, timestamp + ".jpg"))
        origin_height = im.shape[0]
        origin_width = im.shape[1]

        im = letterbox(im, 960, stride=64, auto=True)[0]
        # im_ = im.copy()
        im = transforms.ToTensor()(im)
        im = torch.tensor(np.array([im.numpy()]))

        if torch.cuda.is_available():
            im = im.half().to(self.device)

        yolo_height = im.shape[2]
        yolo_width = im.shape[3]

        output, _ = self.model(im)
        output = non_max_suppression_kpt(
            output, 0.25, 0.65, nc=self.model.yaml["nc"], nkpt=self.model.yaml["nkpt"], kpt_label=True
        )
        output = output_to_keypoint(output)

        # print(yolo_height, yolo_width)
        # print(origin_height, origin_width)
        h_ratio = origin_height / yolo_height
        w_ratio = origin_width / yolo_width

        for i in range(output.shape[0]):
            category_id = int(output[i][1])  ## cls

            bbox = output[i][2:6]
            bbox[0::2] = bbox[0::2] * w_ratio
            bbox[1::2] = bbox[1::2] * h_ratio

            keypoints = output[i][7:]
            keypoints[0::3] = keypoints[0::3] * w_ratio
            keypoints[1::3] = keypoints[1::3] * h_ratio

            self.all_outputs.append(
                {
                    "image_id": int("{}".format(timestamp)) + (100000 * root_idx),
                    "category_id": 1 if category_id == 0 else category_id,
                    "bbox": bbox.tolist(),
                    "keypoints": keypoints.reshape(-1, 3).tolist(),
                    "score": float(output[i][6]),
                }
            )
        return self.all_outputs

    def detect(self):
        root_path = sorted(os.listdir(self.root), key=lambda x: int(x.split("-")[1]))
        with torch.no_grad():
            for root_idx, root_dir in enumerate(root_path):
                path = os.path.join(self.root, root_dir)
                # if root_dir not in ["f-3","M-4"] :
                #     continue
                list_of_images = os.listdir(path)
                for image in tqdm(list_of_images):
                    self.detect_one(image.split(".")[0], root_idx)

    def dump(self):
        with open(self.json_path, "w") as fp:
            json.dump(self.all_outputs, fp)
        print(f"Write keypoints into json file {self.json_path} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation using yolov7")
    parser.add_argument(
        "--train",
        help="Build training dataset",
        action="store_true",
    )

    args = parser.parse_args()

    t1 = time.time()
    key_points_detector = KeyPointDetection(is_train=args.train)
    key_points_detector.detect()
    key_points_detector.dump()
    t2 = time.time()
    print("Total Time :", t2 - t1)
