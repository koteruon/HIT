import os
import tempfile

import cv2
from tqdm import tqdm

# ✅ 指定影片順序
video_files = [
    "PlayingBasketballGotWater_shoot_ball_f_nm_np1_ba_med_6.avi",
    "PlayingBasketballGotWater_shoot_ball_f_nm_np1_ba_med_8.avi",
    "ReggieMillerTakesonThreeAverageGuysinaShootout_shoot_ball_u_nm_np1_ba_med_9.avi",
    "Stairway_to_fitness_climb_stairs_f_nm_np1_fr_med_1.avi",
    "Stairway_to_fitness_climb_stairs_f_nm_np1_fr_med_0.avi",
    "Wayne_Rooney_At_Home_Funny_Must_See_kick_ball_f_cm_np1_le_bad_5.avi",
    "Shootingarecurve_shoot_bow_u_nm_np1_fr_med_2.avi",
    "TrumanShow_run_f_nm_np1_ba_med_21.avi",
    "Baby_Bob_kann_klatschen_!_clap_u_cm_np1_fr_med_0.avi",
    "Fellowship_4_stand_f_cm_np1_fr_med_3.avi",
]

input_dir_a = "data/draw/hitnet_pose_transformer_20250324_seed_0014/inference/jhmdb_val/output_test_videos"
input_dir_b = "data/draw/hitnet_pose_transformer_with_pretrain_skateformer_20250319_seed_0046/inference/jhmdb_val/output_test_videos"
output_path = "output_video/output_concat_slow.avi"
slow_factor = 2


def find_video_path(root_dir, target_filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None


def draw_label_left_right(frame, left_text="HIT Network", right_text="Our Proposed Method"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    half_w = w // 2
    (left_w, text_h), _ = cv2.getTextSize(left_text, font, font_scale, font_thickness)
    (right_w, _), _ = cv2.getTextSize(right_text, font, font_scale, font_thickness)

    x_left = (half_w - left_w) // 2
    y_left = h - 15
    x_right = half_w + (half_w - right_w) // 2
    y_right = h - 15

    def draw_bg_text(x, y, text, text_w):
        box_margin_x, box_margin_y = 10, 5
        box_x1 = x - box_margin_x
        box_y1 = y - text_h - box_margin_y
        box_x2 = x + text_w + box_margin_x
        box_y2 = y + box_margin_y
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 215, 255), font_thickness, cv2.LINE_AA)

    draw_bg_text(x_left, y_left, left_text, left_w)
    draw_bg_text(x_right, y_right, right_text, right_w)


def merge_two_videos(path_a, path_b):
    cap_a = cv2.VideoCapture(path_a)
    cap_b = cv2.VideoCapture(path_b)

    if not cap_a.isOpened() or not cap_b.isOpened():
        print(f"[錯誤] 開啟失敗：{path_a} 或 {path_b}")
        return None, None, None

    width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_a.get(cv2.CAP_PROP_FPS)
    frame_count = int(min(cap_a.get(cv2.CAP_PROP_FRAME_COUNT), cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))

    temp_fd, temp_path = tempfile.mkstemp(suffix=".avi")
    os.close(temp_fd)

    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width * 2, height))

    for _ in range(frame_count):
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        frame_a = cv2.resize(frame_a, (width, height))
        frame_b = cv2.resize(frame_b, (width, height))
        merged = cv2.hconcat([frame_a, frame_b])
        out.write(merged)

    cap_a.release()
    cap_b.release()
    out.release()
    return temp_path, fps, (width * 2, height)


def concat_merged_videos(video_list, output_path, slow_factor=2):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = None
    print("開始合併影片...")
    for name in tqdm(video_list, desc="處理影片"):

        path_a = find_video_path(input_dir_a, name)
        path_b = find_video_path(input_dir_b, name)

        if path_a is None or path_b is None:
            print(f"[警告] 找不到影片：{name}")
            continue

        merged_path, fps, size = merge_two_videos(path_a, path_b)
        if merged_path is None:
            continue

        cap = cv2.VideoCapture(merged_path)
        if writer is None:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"FFV1"), fps / slow_factor, size)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            draw_label_left_right(frame)
            writer.write(frame)
        cap.release()
        os.remove(merged_path)

    if writer:
        writer.release()
        print(f"✅ 輸出完成：{output_path}")
    else:
        print("⚠️ 無影片成功輸出")


if __name__ == "__main__":
    concat_merged_videos(video_files, output_path, slow_factor)
