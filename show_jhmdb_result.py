import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from hit.dataset.datasets import table_tennis_tools


def draw_skeleton(image, keypoints, model):
    keypoints = np.array(keypoints, dtype=np.int32)
    pairs = model.pairs.numpy()

    for idx, (p1, p2) in enumerate(pairs):
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]

        if idx < model.sections[0]:
            color = model.colors["head"]
        elif idx < model.sections[1]:
            color = model.colors["torso"]
        elif idx < model.sections[2]:
            color = model.colors["left_arm"]
        elif idx < model.sections[3]:
            color = model.colors["right_arm"]
        elif idx < model.sections[4]:
            color = model.colors["left_leg"]
        else:
            color = model.colors["right_leg"]

        cv2.line(image, (x1, y1), (x2, y2), color, 3)

    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image


# 動作對應表
action_map = {
    1: "brush hair",
    2: "catch",
    3: "clap",
    4: "climb stairs",
    5: "golf",
    6: "jump",
    7: "kick ball",
    8: "pick",
    9: "pour",
    10: "pullup",
    11: "push",
    12: "run",
    13: "shoot ball",
    14: "shoot bow",
    15: "shoot gun",
    16: "sit",
    17: "stand",
    18: "swing baseball",
    19: "throw",
    20: "walk",
    21: "wave",
}

font_height = 32
thickness = -1
color_gt = (0, 100, 0)
color_action = (0, 0, 255)

# 路徑設定
root_path = "data/draw/hitnet_pose_transformer_with_pretrain_skateformer_20250319_seed_0046/inference/jhmdb_val"
base_det_path = os.path.join(root_path, "detections")
output_dir = os.path.join(root_path, "output_videos")
image_root = "data/jhmdb/videos"
bbox_kpts_json = "data/jhmdb/annotations/jhmdb_test_person_bbox_kpts.json"

os.makedirs(output_dir, exist_ok=True)

# 載入 Ground Truth 與 Keypoints 資料
with open("data/jhmdb/annotations/jhmdb_test_gt_min.json", "r") as f:
    gt_data = json.load(f)

with open(bbox_kpts_json, "r") as f:
    kpts_data = json.load(f)

# 建立映射
gt_action_dict = {ann["image_id"]: ann["action_ids"][0] for ann in gt_data["annotations"]}
kpts_dict = {item["image_id"]: item["keypoints"] for item in kpts_data}

# 初始化骨架模型
model = table_tennis_tools.joint2bone()

# 收集影片影格資料
video_frames = defaultdict(list)

for img_info in gt_data["images"]:
    image_id = img_info["id"]
    movie = img_info["movie"]
    timestamp = img_info["timestamp"]
    img_path = img_info["img_path"]

    txt_path = os.path.join(base_det_path, f"{movie}_{timestamp}.txt")
    if not os.path.exists(txt_path):
        print(f"[警告] 找不到檔案：{txt_path}")
        continue

    # 取得最高分預測
    best_score = -1
    best_action_id = None
    with open(txt_path, "r") as ftxt:
        for line in ftxt:
            parts = line.strip().split()
            if len(parts) >= 2:
                action_id = int(parts[0])
                score = float(parts[1])
                if score > best_score:
                    best_score = score
                    best_action_id = action_id

    gt_action_id = gt_action_dict.get(image_id, -1)
    keypoints = kpts_dict.get(image_id, None)
    if keypoints:
        keypoints = [[x, y] for x, y, c in keypoints]

    video_frames[movie].append((int(timestamp), img_path, best_action_id, gt_action_id, keypoints))

# 輸出影片
for movie, frames in tqdm(video_frames.items(), desc="處理影片數量"):
    frames.sort()
    first_img_path = os.path.join(image_root, frames[0][1])
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        print(f"[錯誤] 無法讀取：{first_img_path}")
        continue

    height, width = first_frame.shape[:2]
    gt_ids = [f[3] for f in frames]
    most_common_gt_action_id = Counter(gt_ids).most_common(1)[0][0]
    action_folder = action_map[most_common_gt_action_id].replace(" ", "_")

    output_subdir = os.path.join(output_dir, action_folder)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, f"{movie}.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

    for _, img_rel_path, pred_action_id, gt_action_id, keypoints in tqdm(frames, desc=f"寫入 {movie}", leave=False):
        img_full_path = os.path.join(image_root, img_rel_path)
        frame = cv2.imread(img_full_path)
        if frame is None:
            print(f"[警告] 無法讀取圖片：{img_full_path}")
            continue

        color = color_gt if pred_action_id == gt_action_id else color_action
        action_name = action_map.get(pred_action_id, "unknown")
        cv2.putText(frame, action_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        if keypoints is not None:
            frame = draw_skeleton(frame, keypoints, model)

        writer.write(frame)

    writer.release()
