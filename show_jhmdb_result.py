import json
import os
from collections import Counter, defaultdict
from copy import deepcopy

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors

from hit.dataset.datasets import table_tennis_tools

colors.pose_palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [0, 204, 255],  # [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def plot_pose(frame, pose_result_names, pose_result_keypoints):
    img = frame  # 1080*960
    img_shape = img.shape[:2]
    annotator = Annotator(
        deepcopy(img),
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        example=pose_result_names,
    )

    annotator.limb_color = colors.pose_palette[[9, 9, 5, 5, 7, 7, 7, 7, 11, 0, 11, 0, 16, 16, 16, 16, 16, 16, 16]]
    annotator.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 7, 7, 11, 0, 11, 0, 7, 7, 9, 5, 9, 5]]

    # Plot Pose results (預期是 list of [17, 3])
    for k in reversed(pose_result_keypoints):
        annotator.kpts(
            k,
            img_shape,
            radius=3,
            kpt_line=True,
            conf_thres=0.5,
            kpt_color=None,
        )

    return annotator.result()


# IoU 計算 (xyxy 格式)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# 找到 IoU 最大的骨架關鍵點與對應 bbox
def find_best_keypoints(image_id, gt_bbox, kpts_data):
    best_iou = -1
    best_kpts, best_bbox = None, None
    for item in kpts_data:
        if item["image_id"] != image_id:
            continue
        iou = compute_iou(gt_bbox, item["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_kpts = item["keypoints"]
            best_bbox = item["bbox"]
    return best_kpts, best_bbox


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
freetype = cv2.freetype.createFreeType2()
freetype.loadFontData("ttf/CALIBRI.TTF", 0)

# ---------------------------------------------test-----------------------------------------

# 路徑設定
root_path = "data/draw/hitnet_pose_transformer_20250324_seed_0014/inference/jhmdb_val"
base_det_path = os.path.join(root_path, "detections")
output_dir = os.path.join(root_path, "output_test_videos")
image_root = "data/jhmdb/videos"
gt_json_path = "data/jhmdb/annotations/jhmdb_test_gt_min.json"
kpt_json_path = "data/jhmdb/annotations/jhmdb_test_person_bbox_kpts.json"
yowo_json_path = "data/jhmdb/annotations/jhmdb_test_yowo_det_score.json"

os.makedirs(output_dir, exist_ok=True)

# 載入資料
with open(gt_json_path, "r") as f:
    gt_data = json.load(f)

with open(kpt_json_path, "r") as f:
    kpts_data = json.load(f)

with open(yowo_json_path) as f:
    yowo_dets = json.load(f)

# 建立映射
gt_ann_dict = {ann["image_id"]: ann for ann in gt_data["annotations"]}
yowo_det_dict = defaultdict(list)
for d in yowo_dets:
    x, y, w, h = d["bbox"]
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    yowo_det_dict[d["image_id"]].append(((x1, y1, x2, y2), d["score"]))

# 初始化骨架模型
model = table_tennis_tools.joint2bone()

# 收集影片影格資料
video_frames = defaultdict(list)

for img_info in tqdm(gt_data["images"], desc="處理標註資料"):
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

    gt_ann = gt_ann_dict.get(image_id, None)
    gt_action_id = gt_ann["action_ids"][0] if gt_ann else -1
    gt_bbox = gt_ann["bbox"] if gt_ann else [0, 0, 0, 0]

    best_kpts, matched_bbox = find_best_keypoints(image_id, gt_bbox, kpts_data)
    if best_kpts:
        best_kpts = [[x, y] for x, y, c in best_kpts]

    video_frames[movie].append(
        (int(timestamp), img_path, best_action_id, gt_action_id, best_kpts, matched_bbox, image_id)
    )

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
    output_path = os.path.join(output_subdir, f"{movie}.avi")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"FFV1"), 30, (width, height))

    for _, img_rel_path, pred_action_id, gt_action_id, keypoints, matched_bbox, image_id in tqdm(
        frames, desc=f"寫入 {movie}", leave=False
    ):
        img_full_path = os.path.join(image_root, img_rel_path)
        frame = cv2.imread(img_full_path)
        if frame is None:
            print(f"[警告] 無法讀取圖片：{img_full_path}")
            continue

        color = color_gt if pred_action_id == gt_action_id else color_action
        action_name = action_map.get(pred_action_id, "unknown")

        # 計算文字大小
        (text_w, text_h), baseline = freetype.getTextSize(action_name, font_height, thickness)
        # 設定文字左上角座標
        x, y = 10, 10

        # 根據文字大小計算背景框位置
        box_x1 = x
        box_y1 = y + 23 - text_h
        box_x2 = x + text_w + 10  # 加一點 padding
        box_y2 = y + 23 + baseline + 10

        # 畫半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 畫上文字（略微下移以置中）
        freetype.putText(
            frame,
            action_name,
            (x + 5, y - 4),  # 微調置中
            font_height,
            color,
            thickness,
            cv2.LINE_AA,
            False,
        )

        # 畫出 person bbox（細黑框）
        # if matched_bbox:
        #     x1, y1, x2, y2 = map(int, matched_bbox)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 189), 1)

        # 畫出 YOWO 偵測框（細黑框）
        # top3 = sorted(yowo_det_dict[image_id], key=lambda x: -x[1])[:3]
        # for bbox, _ in top3:
        #     x1, y1, x2, y2 = map(int, bbox)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

        if keypoints is not None:
            # Step 1: 轉為 numpy 陣列 (17, 2)
            keypoints_xy = np.array(keypoints)
            # Step 2: 加上 conf=1.0，變成 (17, 3)
            keypoints_full = np.hstack([keypoints_xy, np.ones((17, 1), dtype=float)])
            # Step 3: 包成 list of [17, 3] 結構 (支援多人)
            pose_keypoints_batch = [keypoints_full]

            frame = plot_pose(frame, ["person"], pose_keypoints_batch)

        writer.write(frame)

    writer.release()


# ---------------------------------------------train-----------------------------------------

output_dir = os.path.join(root_path, "output_train_videos")
image_root = "data/jhmdb/videos"
gt_json_path = "data/jhmdb/annotations/jhmdb_train_gt_min.json"
kpt_json_path = "data/jhmdb/annotations/jhmdb_train_person_bbox_kpts.json"


os.makedirs(output_dir, exist_ok=True)

# 載入資料
with open(gt_json_path, "r") as f:
    gt_data = json.load(f)

with open(kpt_json_path, "r") as f:
    kpts_data = json.load(f)

# 建立映射
gt_ann_dict = {ann["image_id"]: ann for ann in gt_data["annotations"]}

# 收集影片影格資料
video_frames = defaultdict(list)

for img_info in tqdm(gt_data["images"], desc="處理標註資料"):
    image_id = img_info["id"]
    movie = img_info["movie"]
    timestamp = img_info["timestamp"]
    img_path = img_info["img_path"]

    gt_ann = gt_ann_dict.get(image_id, None)
    gt_action_id = gt_ann["action_ids"][0] if gt_ann else -1
    gt_bbox = gt_ann["bbox"] if gt_ann else [0, 0, 0, 0]

    best_kpts, matched_bbox = find_best_keypoints(image_id, gt_bbox, kpts_data)
    if best_kpts:
        best_kpts = [[x, y] for x, y, c in best_kpts]

    video_frames[movie].append((int(timestamp), img_path, -1, gt_action_id, best_kpts, matched_bbox, image_id))

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
    output_path = os.path.join(output_subdir, f"{movie}.avi")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"FFV1"), 30, (width, height))

    for _, img_rel_path, _, _, keypoints, _, _ in tqdm(frames, desc=f"寫入 {movie}", leave=False):
        img_full_path = os.path.join(image_root, img_rel_path)
        frame = cv2.imread(img_full_path)
        if frame is None:
            print(f"[警告] 無法讀取圖片：{img_full_path}")
            continue

        if keypoints is not None:
            # Step 1: 轉為 numpy 陣列 (17, 2)
            keypoints_xy = np.array(keypoints)
            # Step 2: 加上 conf=1.0，變成 (17, 3)
            keypoints_full = np.hstack([keypoints_xy, np.ones((17, 1), dtype=float)])
            # Step 3: 包成 list of [17, 3] 結構 (支援多人)
            pose_keypoints_batch = [keypoints_full]

            frame = plot_pose(frame, ["person"], pose_keypoints_batch)

        writer.write(frame)

    writer.release()
