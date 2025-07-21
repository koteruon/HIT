import cv2
import numpy as np
from tqdm import tqdm

labels = [
    "反手切球",  # "Backhand Chop",
    "反手擰球",  # "Backhand Flick",
    "正手切球",  # "Forehand Chop",
    "正手平擊",  # "Forehand Drive",
    "反手推球",  # "Backhand Push",
    "反手拉球",  # "Backhand Topspin",
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


# 影片路徑與設定
prefix = "without_racket_info_left/"
video_paths = [
    "backhand_chop_01.avi",
    "backhand_flick_01.avi",
    "forehand_chop_01.avi",
    "forehand_drive_01.avi",
    "backhand_push_01.avi",
    "backhand_topspin_01.avi",
    "forehand_smash_01.avi",
    "forehand_topspin_01.avi",
]
video_paths = [prefix + path for path in video_paths]

video_w, video_h = 960, 1080
cols, rows = 4, 2
output_w = video_w * cols
output_h = video_h * rows

# 開啟影片
caps = [cv2.VideoCapture(path) for path in video_paths]
last_frames = [None] * len(caps)

# FPS 和 frame 數
fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps if cap.isOpened()]
fps = int(min(fps_list)) if fps_list else 25
frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps if cap.isOpened()]
min_frames = min(frame_counts) if frame_counts else 0
max_frames = max(frame_counts) if frame_counts else 0

# 輸出影片設定
fourcc = cv2.VideoWriter_fourcc(*"FFV1")
out_short = cv2.VideoWriter(prefix + "stitched_short.avi", fourcc, fps, (output_w, output_h))
out_long = cv2.VideoWriter(prefix + "stitched_long.avi", fourcc, fps, (output_w, output_h))

# 建立進度條
pbar = tqdm(total=max_frames, desc="處理影片", unit="frame")

short_done = False

for _ in range(max_frames):
    row_frames = [[] for _ in range(rows)]
    any_video_ended = False
    all_video_ended = True

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            last_frames[i] = frame.copy()
            all_video_ended = False
        else:
            frame = (
                last_frames[i].copy() if last_frames[i] is not None else np.zeros((video_h, video_w, 3), dtype=np.uint8)
            )
            any_video_ended = True

        draw_label(frame, labels[i], freetype)
        row = i // cols
        row_frames[row].append(frame)

    stitched_rows = [cv2.hconcat(frames) for frames in row_frames]
    final_frame = cv2.vconcat(stitched_rows)

    if not short_done and any_video_ended:
        short_done = True  # 停止寫入 short
    if not short_done:
        out_short.write(final_frame)

    out_long.write(final_frame)
    pbar.update(1)

    if all_video_ended:
        break  # long 也完成

pbar.close()
for cap in caps:
    cap.release()
out_short.release()
out_long.release()

print("✅ 短版：stitched_short.avi")
print("✅ 長版：stitched_long.avi")
