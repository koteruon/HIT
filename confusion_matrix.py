import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 自訂類別名稱對照表（可修改）
class_names_dict = {
    0: "backhand chop",
    1: "backhand flick",
    2: "backhand push",
    3: "backhand topspin",
    4: "forehand chop",
    5: "forehand drive",
    6: "forehand smash",
    7: "forehand topspin",
    8: "background",
}

root_path = "confusion_matrix"
os.makedirs(root_path, exist_ok=True)


def generate_confusion_matrix(raw_text, save_file_name):
    # 轉換為 numpy 矩陣
    conf_mat = np.array([list(map(int, line.strip().split(","))) for line in raw_text.strip().splitlines()])

    # 計算 row-wise 比率（顏色用）
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    normalized_colors = conf_mat / row_sums

    # 根據 dict 排列標籤（確保順序正確）
    num_classes = conf_mat.shape[0]
    labels = [class_names_dict.get(i, f"Class {i+1}") for i in range(num_classes)]

    # 畫圖
    plt.figure(figsize=(1.2 * num_classes, 1.1 * num_classes))
    sns.heatmap(
        normalized_colors,
        annot=conf_mat,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12},
        vmin=0,
        vmax=1,
    )

    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title("Confusion Matrix (Color = Ratio, Value = Count)", fontsize=18)

    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # 計算 Precision / Recall / F1
    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    avg_all = {"Precision": np.mean(precision), "Recall": np.mean(recall), "F1-Score": np.mean(f1)}

    # 排除最後一類
    conf_mat_no_bg = conf_mat[:-1, :-1]

    TP_no_bg = np.diag(conf_mat_no_bg)
    FP_no_bg = np.sum(conf_mat_no_bg, axis=0) - TP_no_bg
    FN_no_bg = np.sum(conf_mat_no_bg, axis=1) - TP_no_bg

    precision_no_bg = TP_no_bg / (TP_no_bg + FP_no_bg + 1e-10)
    recall_no_bg = TP_no_bg / (TP_no_bg + FN_no_bg + 1e-10)
    f1_no_bg = 2 * precision_no_bg * recall_no_bg / (precision_no_bg + recall_no_bg + 1e-10)

    avg_no_bg = {
        "Precision": np.mean(precision_no_bg),
        "Recall": np.mean(recall_no_bg),
        "F1-Score": np.mean(f1_no_bg),
    }

    # 每一類別的 Precision / Recall / F1
    class_data = []
    for i in range(num_classes):
        class_data.append(
            [
                f"{i+1}",
                f"{precision[i]:.3f}",
                f"{recall[i]:.3f}",
                f"{f1[i]:.3f}",
            ]
        )
    class_column_labels = ["Class", "Precision", "Recall", "F1-Score"]

    # 建立新的表格放在主圖下方
    table = plt.table(
        cellText=class_data,
        colLabels=class_column_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[-0.25, -1.2, 1.4, 0.9],  # 視圖高度可調整
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)

    # 準備表格資料
    table_data = [
        ["Average (w/ bg)", f"{avg_all['Precision']:.3f}", f"{avg_all['Recall']:.3f}", f"{avg_all['F1-Score']:.3f}"],
        [
            "Average (w/o bg)",
            f"{avg_no_bg['Precision']:.3f}",
            f"{avg_no_bg['Recall']:.3f}",
            f"{avg_no_bg['F1-Score']:.3f}",
        ],
    ]
    column_labels = ["Range", "Precision", "Recall", "F1-Score"]

    # ---- 建立表格並放置在圖片下方 ----
    table2 = plt.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[-0.25, -1.5, 1.4, 0.25],  # [x, y, width, height]，視圖高度而定
        edges="closed",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(16)

    # 儲存圖片
    plt.savefig(os.path.join(root_path, save_file_name), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"已儲存混淆矩陣圖為 {os.path.join(root_path, save_file_name)}")


# ---------------------------------------HIT only rgb-------------------------------------------------

# 貼上你的混淆矩陣文字
raw_text = """
252,  0,  0,  0,  0,  0,  0,  0, 23
  0,250,  0,  0,  0,  0,  0,  0, 14
 66,  0,291,  0,  0,  0,  0,  0, 53
  1,  6, 54,166,  0,  0,  0,  2,  6
  0,  0,  0,  0,238,  0,  0,  0, 67
  0,  0,  0,  0,  5,122,  0,  0, 28
  0,  0,  0,  0,  4, 43, 54,  7,115
  0,  0,  0,  0,  0,  0,  0,352, 34
 37,202, 70, 27, 11, 29,  0,126,3418
"""

generate_confusion_matrix(raw_text, "hit_only_rgb.png")

# ---------------------------------------HIT add single frame pose-------------------------------------------------

# 貼上你的混淆矩陣文字
raw_text = """
270,  0,  0,  0,  0,  0,  0,  0,  5
  0,257,  0,  2,  0,  0,  0,  0,  5
 79,  0,253,  0,  0,  0,  0,  0, 78
  0,  0,  0,234,  0,  0,  0,  0,  1
  0,  0,  0,  0,283,  0,  0,  0, 22
  0,  0,  0,  0, 88, 60,  1,  0,  6
  0,  0,  0,  0,  3,  0,138,  0, 82
  0,  0,  0,  0,  0,  0,  0,367, 19
 71,152, 26,118,119,  8,  1, 58,3367
"""

generate_confusion_matrix(raw_text, "hit_add_single_frame_pose.png")


# --------------------------------------- skateformer -------------------------------------------------

# 貼上你的混淆矩陣文字
raw_text = """
162,  0,  0,  0,  0,  0,  0,  0,113
  0,233,  0,  0,  0,  0,  0,  0, 31
  0,  0,232,  0,  0,  0,  0,  0,178
  0,  0,  0,224,  0,  0,  0,  0, 11
  0,  0,  0,  0,291,  5,  0,  0,  9
  0,  0,  0,  0,  6,147,  0,  0,  2
  0,  0,  0,  0,  0,  0,189,  0, 34
  0,  0,  0,  0,  0,  0,  0,370, 16
  0, 33,  0,  2, 27, 22,  2, 43,3791
"""
generate_confusion_matrix(raw_text, "skateformer.png")


# ---------------------------------------HIT add skateformer -------------------------------------------------

# 貼上你的混淆矩陣文字
raw_text = """
259,  0,  0,  0,  0,  0,  0,  0, 16
  0,257,  0,  0,  0,  0,  0,  0,  7
  0,  0,342,  0,  0,  0,  0,  0, 68
  0,  0,  0,226,  0,  0,  0,  0,  9
  0,  0,  0,  0,279,  0,  0,  0, 26
  0,  0,  0,  0,  0,149,  0,  0,  6
  0,  0,  0,  0,  0,  1,164,  0, 58
  0,  0,  0,  0,  0,  0,  0,364, 22
  9, 59,  8, 30,  5,  7,  0, 16,3786
"""

generate_confusion_matrix(raw_text, "hit_add_skateformer.png")
