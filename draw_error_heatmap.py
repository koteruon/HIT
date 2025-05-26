import csv
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from draw_upper_bound_cal import read_method_instances

root_path = "error_heatmap"
os.makedirs(root_path, exist_ok=True)


def generate_error_heatmap_from_single_method(annotation_dir, a_file, image_name):
    normalized_length = 30
    video_names = []
    video_normalized_errors = []

    per_video_lengths = {}  # 用來儲存每個 video 的 frame 長度總和
    per_video_counts = {}  # 用來儲存每個 video 有效段落數

    a_instances = read_method_instances(a_file, "a")

    csv_files = sorted(glob(os.path.join(annotation_dir, "*_annotations.csv")))
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        video_name = os.path.basename(file_path).replace("_left_01_annotations.csv", "")
        segments = []

        total_length = 0
        valid_segment_count = 0

        for _, row in df.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            length = end - start + 1
            if length < 2:
                continue

            # 真實錯誤序列（1 = error, 0 = correct）
            is_save = False
            error_seq = np.zeros(length)
            for i in range(length):
                frame = start + i
                key = f"data/stroke_postures/videos/{video_name}_01,{frame}"
                if key in a_instances:
                    is_save = True
                    pred = a_instances[key]["pred_a"]
                    gt = a_instances[key]["gt"]
                    error_seq[i] = 0 if pred == gt else 1

            # 差值為 30 格
            if is_save:
                # 計算並儲存該段長度
                total_length += length
                valid_segment_count += 1

                interp_x = np.linspace(0, length - 1, normalized_length)
                interpolated = np.interp(interp_x, np.arange(length), error_seq)
                segments.append(interpolated)

        if segments:
            per_video_lengths[video_name] = total_length
            per_video_counts[video_name] = valid_segment_count

            video_names.append(video_name)
            video_normalized_errors.append(np.mean(segments, axis=0))

    print("\n各影片平均 frame 數：")
    total_frames = 0
    total_segments = 0
    for vname in video_names:
        total = per_video_lengths.get(vname, 0)
        count = per_video_counts.get(vname, 0)
        avg = total / count if count > 0 else 0
        print(f"  {vname:25s} 平均長度: {avg:.2f} ({count} 段)")

        total_frames += total
        total_segments += count

    if total_segments > 0:
        overall_avg = total_frames / total_segments
        print(f"\n所有段落平均長度（frame）: {overall_avg:.2f}")
    else:
        print("\n無有效段落資料")

    # 繪圖
    error_matrix = np.array(video_normalized_errors)
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        error_matrix,
        cmap=sns.light_palette("steelblue", as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Average Error Rate"},
        xticklabels=[f"{i+1}" for i in range(normalized_length)],
        yticklabels=[v.replace("_", " ") for v in video_names],
    )
    plt.title("Average Per-Frame Error Rate Heatmap", fontsize=14)
    plt.xlabel("Normalized Frame Index (1–30)", fontsize=12)
    plt.ylabel("Action Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, image_name), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"已儲存至 {os.path.join(root_path, image_name)}")


if __name__ == "__main__":
    without_racket = "data/bast/hitnet_pose_transformer_stroke_postures_with_pretrain_skateformer_joint_20250424_seed_0018/inference/stroke_postures_val_450/result_top1_action_by_frame_confusion_matrix_stroke_postures.csv"
    generate_error_heatmap_from_single_method(
        annotation_dir="./data/stroke_postures/select_frame/20250331",
        a_file=without_racket,
        image_name="error_heatmap_without_racket.png",
    )

    with_racket = "data/bast/hitnet_pose_transformer_stroke_postures_with_pretrain_skateformer_and_racket_info_joint_20250514_seed_0008/inference/stroke_postures_val/result_top1_action_by_frame_confusion_matrix_stroke_postures.csv"
    generate_error_heatmap_from_single_method(
        annotation_dir="./data/stroke_postures/select_frame/20250331",
        a_file=with_racket,
        image_name="error_heatmap_with_racket.png",
    )
