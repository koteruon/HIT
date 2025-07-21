import os

import cv2
from tqdm import tqdm


def crop_left_half_and_save_ffv1(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".avi"):
            continue

        input_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"❌ 無法讀取影片：{filename}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        half_width = width // 2

        output_path = os.path.join(output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (half_width, height))

        print(f"📁 處理影片：{filename}")
        for _ in tqdm(range(total_frames), desc=f"  處理中", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            left_half = frame[:, :half_width]
            out.write(left_half)

        cap.release()
        out.release()
        print(f"✅ 已儲存：{output_path}\n")


# ✅ 使用方式
names = ["all", "hit", "with_racket_info", "without_racket_info"]

for name in names:
    input_folder = name
    output_folder = name + "_left"
    crop_left_half_and_save_ffv1(input_folder, output_folder)
