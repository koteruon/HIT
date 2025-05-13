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

# 儲存圖片
plt.savefig(os.path.join(root_path, "hit_add_single_frame_pose.png"), dpi=300)
plt.close()

print(f"已儲存混淆矩陣圖為 {os.path.join(root_path, 'hit_add_single_frame_pose.png')}")

# --------------------------------------- skateformer -------------------------------------------------

# 貼上你的混淆矩陣文字
raw_text = """
179,  0,  0,  0,  0,  0,  0,  0, 96
  0,239,  0,  0,  0,  0,  0,  0, 25
  0,  0,196,  0,  0,  0,  0,  0,214
  0,  0,  0,229,  0,  0,  0,  0,  6
  0,  0,  0,  0,295,  2,  0,  0,  8
  0,  0,  0,  0,  0,152,  0,  0,  3
  0,  0,  0,  0,  0,  0,189,  0, 34
  0,  0,  0,  0,  0,  0,  0,369, 17
  8, 53,  0,  4, 20, 17,  2, 50,3766
"""

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

# 儲存圖片
plt.savefig(os.path.join(root_path, "skateformer.png"), dpi=300)
plt.close()

print(f"已儲存混淆矩陣圖為 {os.path.join(root_path, 'skateformer.png')}")

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

# 儲存圖片
plt.savefig(os.path.join(root_path, "hit_add_skateformer.png"), dpi=300)
plt.close()

print(f"已儲存混淆矩陣圖為 {os.path.join(root_path, 'hit_add_skateformer.png')}")
