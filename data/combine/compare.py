import os

import cv2
import numpy as np
from tqdm import tqdm

# 資料夾路徑
video_dir = "all_left"
output_path = os.path.join(video_dir, "combined_labeled.avi")

# 影片檔名
backhand_videos = ["backhand_chop_01.avi", "backhand_flick_01.avi", "backhand_push_01.avi", "backhand_topspin_01.avi"]
forehand_videos = ["forehand_chop_01.avi", "forehand_drive_01.avi", "forehand_smash_01.avi", "forehand_topspin_01.avi"]

labels = [
    "反手切球",  # "Backhand Chop",
    "反手擰球",  # "Backhand Flick",
    "反手推球",  # "Backhand Push",
    "反手拉球",  # "Backhand Topspin",
    "正手切球",  # "Forehand Chop",
    "正手平擊",  # "Forehand Drive",
    "正手殺球",  # "Forehand Smash",
    "正手拉球",  # "Forehand Topspin",
]


# 建立 freetype 字體物件
freetype = cv2.freetype.createFreeType2()
freetype.loadFontData("/home/siplab5/chaoen/yoloNhit_calvin/HIT_B2C/ttf/MSJH.TTC", 0)


def draw_label(frame, text, freetype, font_height=48):
    h, w = frame.shape[:2]
    thickness = -1
    color = (0, 215, 255)

    # 正確取得文字大小與 baseline
    (text_size, baseline) = freetype.getTextSize(text, font_height, thickness)
    text_w, text_h = text_size

    # 中心置中，靠下方顯示
    x = (w - text_w) // 2
    y = h - 40  # 預留底部空間

    # 背景框（需加上 baseline，才能包住整個文字區）
    box_margin_x, box_margin_y = 40, 20
    box_x1 = x - box_margin_x
    box_y1 = y - text_h - box_margin_y
    box_x2 = x + text_w + box_margin_x
    box_y2 = y + box_margin_y
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # 畫文字
    freetype.putText(frame, text, (x, y - text_h - 8), font_height, color, thickness, cv2.LINE_AA, False)


# 視訊參數
frame_width, frame_height = 960, 1080
fps = 60
fourcc = cv2.VideoWriter_fourcc(*"FFV1")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

# 設定每對影片最多處理多少幀
max_frames_per_pair = -1  # 若不想限制可以設為 float("inf")

# 處理每對影片
for i, (left_name, right_name) in enumerate(
    tqdm(zip(backhand_videos, forehand_videos), total=len(backhand_videos), desc="處理影片對")
):
    left_path = os.path.join(video_dir, left_name)
    right_path = os.path.join(video_dir, right_name)
    left_label = labels[i]
    right_label = labels[i + 4]  # forehand 從 index 4 開始

    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)

    left_last_frame = None
    right_last_frame = None

    left_len = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_len = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_len = max(left_len, right_len)

    # 限制最大幀數
    if max_frames_per_pair == -1:
        process_len = max_len
    else:
        process_len = min(max_len, max_frames_per_pair)

    for _ in tqdm(range(process_len), desc=f"{left_name} + {right_name}", leave=False):
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()

        if left_ret:
            draw_label(left_frame, left_label, freetype)
            left_last_frame = left_frame
        else:
            left_frame = left_last_frame

        if right_ret:
            draw_label(right_frame, right_label, freetype)
            right_last_frame = right_frame
        else:
            right_frame = right_last_frame

        if left_frame is None and right_frame is None:
            break

        if left_frame is None:
            left_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if right_frame is None:
            right_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        combined_frame = np.hstack((left_frame, right_frame))
        out.write(combined_frame)

    left_cap.release()
    right_cap.release()

out.release()
print(f"✅ 成功輸出：{output_path}")
