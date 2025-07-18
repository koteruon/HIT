import io
import os

import matplotlib.pyplot as plt
import pandas as pd

# === 貼上第一組結果 ===
data1_text = """
Class  ,Precision,Recall   ,F1-Score
      1,    1.000,    0.998,    0.999
      2,    0.692,    0.865,    0.769
      3,    1.000,    0.907,    0.951
      4,    0.927,    0.711,    0.805
      5,    0.938,    1.000,    0.968
      6,    0.706,    0.675,    0.690
      7,    0.716,    0.705,    0.710
      8,    0.862,    0.889,    0.875
      9,    0.988,    0.998,    0.993
     10,    0.977,    1.000,    0.988
     11,    1.000,    0.857,    0.923
     12,    0.491,    0.668,    0.566
     13,    0.910,    0.399,    0.555
     14,    1.000,    0.962,    0.981
     15,    0.996,    0.970,    0.983
     16,    0.603,    0.659,    0.630
     17,    0.535,    0.513,    0.524
     18,    1.000,    0.990,    0.995
     19,    0.781,    0.813,    0.797
     20,    0.668,    0.768,    0.714
     21,    0.829,    0.955,    0.887
"""

# === 貼上第二組結果 ===
data2_text = """
Class  ,Precision,Recall   ,F1-Score
      1,    0.994,    1.000,    0.997
      2,    0.677,    0.768,    0.719
      3,    0.971,    0.932,    0.951
      4,    0.990,    0.896,    0.941
      5,    0.998,    1.000,    0.999
      6,    0.771,    0.594,    0.671
      7,    0.842,    0.812,    0.826
      8,    0.865,    0.918,    0.891
      9,    1.000,    1.000,    1.000
     10,    0.957,    1.000,    0.978
     11,    0.909,    1.000,    0.952
     12,    0.544,    0.620,    0.580
     13,    0.736,    0.683,    0.709
     14,    1.000,    1.000,    1.000
     15,    0.939,    0.972,    0.955
     16,    0.554,    0.608,    0.579
     17,    0.521,    0.431,    0.472
     18,    0.998,    1.000,    0.999
     19,    0.793,    0.698,    0.742
     20,    0.751,    0.796,    0.773
     21,    0.867,    0.867,    0.867
"""

# === 貼上動作對照表 ===
action_map_text = """
action_id,action_name
1,brush_hair
2,catch
3,clap
4,climb_stairs
5,golf
6,jump
7,kick_ball
8,pick
9,pour
10,pullup
11,push
12,run
13,shoot_ball
14,shoot_bow
15,shoot_gun
16,sit
17,stand
18,swing_baseball
19,throw
20,walk
21,wave
"""


# === 轉換為 DataFrame ===
def load_text_data(text, suffix):
    df = pd.read_csv(io.StringIO(text.strip()))
    df.columns = ["Class", f"Precision_{suffix}", f"Recall_{suffix}", f"F1_{suffix}"]
    return df


df1 = load_text_data(data1_text, "1")
df2 = load_text_data(data2_text, "2")
action_map = pd.read_csv(io.StringIO(action_map_text.strip()))

# 合併資料並加上動作名稱
df = pd.merge(df1, df2, on="Class")
df = pd.merge(df, action_map, left_on="Class", right_on="action_id")
df["Delta_F1"] = df["F1_2"] - df["F1_1"]
df_sorted = df.sort_values("Delta_F1", ascending=False)

# === 畫圖 ===
bar_colors = ["green" if d >= 0 else "red" for d in df_sorted["Delta_F1"]]

plt.figure(figsize=(12, 6))
plt.bar(df_sorted["action_name"], df_sorted["Delta_F1"], color=bar_colors)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.ylabel("ΔF1-Score (F1_2 - F1_1)")
plt.title("ΔF1-Score per Action Class")
plt.tight_layout()


# 確保資料夾存在
output_dir = "jhmdb_analyze"
os.makedirs(output_dir, exist_ok=True)

# 儲存圖像
output_path = os.path.join(output_dir, "delta_f1_comparison.png")
plt.savefig(output_path, dpi=600)
print(f"圖已儲存至：{output_path}")
