import os

import cv2
import pandas as pd
from tqdm import tqdm

videos_path = "data/stroke_postures/videos"
output_path = "data/output/hitnet_pose_transformer_stroke_postures_20250325_06/inference/stroke_postures_val_1000"
result_path = os.path.join(output_path, "result_top1_action_by_frame_confusion_matrix_stroke_postures.csv")


with open(result_path, "r") as file:
    lines = file.readlines()
# 儲存處理過的資料
csv_data = []
# 逐行處理資料
for line in lines:
    if line == "":
        break
    csv_data.append(line.split(","))

# 把清理過的資料轉換為 pandas 的 DataFrame
df = pd.DataFrame(
    csv_data,
    columns=[
        "movie_name_with_dir",
        "timestamp",
        "box_str_x1",
        "box_str_y1",
        "box_str_x2",
        "box_str_y2",
        "action_id",
        "score_str",
        "gt_action_id",
    ],
)

# 設定文字顏色
color_gt = (0, 255, 0)  # 綠色
color_action = (0, 0, 255)  # 紅色
color_score = (0, 255, 255)  # 黃色
# 在圖片上顯示文字
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

videos_dir = os.listdir(videos_path)
for video_dir in videos_dir:
    frame_width = 1920
    frame_height = 1080
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(os.path.join(output_path, video_dir + ".mp4"), fourcc, fps, (frame_width, frame_height))
    movie_name_with_dir = os.path.join(videos_path, video_dir)
    file_names = sorted(os.listdir(movie_name_with_dir))
    started = False
    for file_name_with_extension in tqdm(file_names):
        file_path = os.path.join(movie_name_with_dir, file_name_with_extension)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        timestamp = str(int(file_name)).zfill(4)
        # 根據條件篩選 DataFrame 中的資料
        filtered_df = df[(df["movie_name_with_dir"] == movie_name_with_dir) & (df["timestamp"] == timestamp)]
        if not filtered_df.empty:
            started = True
        if not started:
            continue

        # 讀取圖片
        image = cv2.imread(file_path)

        if not filtered_df.empty:
            # 提取相應的欄位
            action_id = filtered_df["action_id"].iloc[0]
            score_str = filtered_df["score_str"].iloc[0]
            gt_action_id = filtered_df["gt_action_id"].iloc[0]

            # 顯示 gt_action_id (綠色)
            cv2.putText(image, f"GT: {gt_action_id}", (10, 40), font, font_scale, color_gt, thickness)

            # 顯示 action_id (根據是否相等來選擇顏色)
            if action_id == gt_action_id:
                action_color = color_gt  # 如果相等，顯示綠色
            else:
                action_color = color_action  # 如果不相等，顯示紅色
            cv2.putText(image, f"Action: {action_id}", (10, 80), font, font_scale, action_color, thickness)

            # 顯示 score_str (黃色)
            cv2.putText(image, f"Score: {score_str}", (10, 120), font, font_scale, color_score, thickness)

        video_out.write(image)
    video_out.release()
