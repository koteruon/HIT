import csv
import glob
import os
from collections import defaultdict

import numpy as np


def read_method_instances(filepath, method):
    instances = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].startswith("data/"):
                continue
            key = f"{row[0]},{row[1]}"
            frame_number = int(row[1])
            pred = int(row[-3])
            gt = int(row[-1])
            instances[key] = {"frame": frame_number, f"pred_{method}": pred, "gt": gt}
    return instances


def evaluate_fusion_upper_bound(a_file, b_file):
    a_instances = read_method_instances(a_file, "a")
    b_instances = read_method_instances(b_file, "b")

    inconsistent_gt = []
    common_keys = set(a_instances.keys()) & set(b_instances.keys())
    for key in common_keys:
        gt_a = a_instances[key]["gt"]
        gt_b = b_instances[key]["gt"]
        if gt_a != gt_b:
            inconsistent_gt.append((key, gt_a, gt_b))

    print(f"總共有 {len(common_keys)} 筆共同 instance")
    print(f"GT 不一致的筆數：{len(inconsistent_gt)}")
    if inconsistent_gt:
        print("以下為前幾筆不一致 GT 的資料：")
        for i, (key, a_gt, b_gt) in enumerate(inconsistent_gt[:10]):
            print(f"{key} → A: {a_gt}, B: {b_gt}")

    merged = []
    for key in common_keys:
        if a_instances[key]["gt"] == b_instances[key]["gt"]:
            merged.append(
                {
                    "key": key,
                    "gt": a_instances[key]["gt"],
                    "pred_a": a_instances[key]["pred_a"],
                    "pred_b": b_instances[key]["pred_b"],
                }
            )

    class_tp_a = defaultdict(int)
    class_tp_b_rescue = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for row in merged:
        gt = row["gt"]
        pred_a = row["pred_a"]
        pred_b = row["pred_b"]
        a_correct = pred_a == gt
        b_correct = pred_b == gt

        if a_correct:
            class_tp_a[gt] += 1
        elif b_correct:
            class_tp_b_rescue[gt] += 1
        else:
            class_fn[gt] += 1
            class_fp[pred_a] += 1

    all_classes = sorted(set(class_tp_a) | set(class_tp_b_rescue) | set(class_fp) | set(class_fn))

    # 顯示表格
    print("\n【Per-Class Fusion Upper Bound Metrics】")
    print(
        f"{'Class':>7} | {'TP(A)':>6} | {'TP(A or B)':>11} | {'FP':>4} | {'FN':>4} | {'Precision ↑':>11} | {'Recall ↑':>9} | {'F1-score ↑':>11}"
    )
    print("-" * 90)

    result = []
    precisions, recalls, f1_scores = [], [], []
    for cls in all_classes:
        tp_a = class_tp_a.get(cls, 0)
        tp_b = class_tp_b_rescue.get(cls, 0)
        tp_total = tp_a + tp_b
        fp = class_fp.get(cls, 0)
        fn = class_fn.get(cls, 0)

        prec = tp_total / (tp_total + fp) if (tp_total + fp) > 0 else 0
        rec = tp_total / (tp_total + fn) if (tp_total + fn) > 0 else 0
        f1_cls = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(
            f"{cls:>7} | {tp_a:>6} | {tp_total:>11} | {fp:>4} | {fn:>4} | {prec:>11.3f} | {rec:>9.3f} | {f1_cls:>11.3f}"
        )

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1_cls)

        result.append(
            {
                "class": cls,
                "tp_a": tp_a,
                "tp_total": tp_total,
                "fp": fp,
                "fn": fn,
                "precision_upper": prec,
                "recall_upper": rec,
                "f1_upper": f1_cls,
            }
        )

    # 顯示平均值
    print("-" * 90)
    print(
        f"{'Average':>7} | {'-':>6} | {'-':>11} | {'-':>4} | {'-':>4} | {np.mean(precisions):>11.3f} | {np.mean(recalls):>9.3f} | {np.mean(f1_scores):>11.3f}"
    )

    return result


def count_exceeded_metrics(result):
    print("\n【Compare with Original Results — Exceeded Fusion Upper Bound】")

    original_table = """
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
    Average,    0.842,    0.838,    0.838
    """.strip().splitlines()[
        1:
    ]

    # 轉成 dict
    original_result = {}
    for row in original_table:
        parts = row.strip().split(",")
        cls = int(parts[0]) if parts[0] != "Average" else "Average"
        original_result[cls] = {
            "precision": float(parts[1]),
            "recall": float(parts[2]),
            "f1": float(parts[3]),
        }

    exceeded_classes = []
    exceed_count = 0

    precisions = []
    recalls = []
    f1_scores = []

    for row in result:
        cls = row["class"]
        fusion_prec = row["precision_upper"]
        fusion_rec = row["recall_upper"]
        fusion_f1 = row["f1_upper"]

        precisions.append(fusion_prec)
        recalls.append(fusion_rec)
        f1_scores.append(fusion_f1)

        orig = original_result.get(cls, {})
        if not orig:
            continue

        exceeded = []
        if orig["precision"] > fusion_prec:
            exceeded.append("Precision")
            exceed_count += 1
        if orig["recall"] > fusion_rec:
            exceeded.append("Recall")
            exceed_count += 1
        if orig["f1"] > fusion_f1:
            exceeded.append("F1-score")
            exceed_count += 1

        if exceeded:
            exceeded_classes.append((cls, exceeded))

    # 顯示 class 列表
    if exceeded_classes:
        print(f"{'Class':>7} | {'Exceeded Metrics'}")
        print("-" * 40)
        for cls, items in exceeded_classes:
            print(f"{cls:>7} | {', '.join(items)}")
    else:
        print("✅ No class in the original result exceeded the fusion upper bound.")

    # 處理 Average
    avg_fusion_prec = np.mean(precisions)
    avg_fusion_rec = np.mean(recalls)
    avg_fusion_f1 = np.mean(f1_scores)
    avg_exceeded = []
    if original_result["Average"]["precision"] > avg_fusion_prec:
        avg_exceeded.append("Precision")
        exceed_count += 1
    if original_result["Average"]["recall"] > avg_fusion_rec:
        avg_exceeded.append("Recall")
        exceed_count += 1
    if original_result["Average"]["f1"] > avg_fusion_f1:
        avg_exceeded.append("F1-score")
        exceed_count += 1

    print("-" * 40)
    print(f"{'Average':>7} | {', '.join(avg_exceeded) if avg_exceeded else 'OK'}")

    print(f"\n🔢 總共超過 fusion 上限的 precision / recall / f1 筆數：{exceed_count}")

    return exceed_count


if __name__ == "__main__":

    # 輸入檔案路徑
    # a_file = "data/bast/hitnet_pose_transformer_stroke_postures_joint_only_rgb_20250511_seed_0004/inference/stroke_postures_val_450/result_top1_action_by_frame_confusion_matrix_stroke_postures.csv"
    # b_file = "data/bast/stroke_postures/SkateFormer_j_2D_20250423/runs-180-16380_top1f.csv"
    # result = evaluate_fusion_upper_bound(a_file, b_file)
    # exceed_count = count_exceeded_metrics(result)

    # 目前141選擇
    # a_dir = "data/output/232"
    # a_dir_prefixes = ["hitnet_pose_transformer_only_rgb_20250530_"]
    # b_dir = "data/bast/jhmdb"
    # b_dir_prefixes = ["SkateFormer_j_2D_20250521_"]

    # 來自232
    a_dir = "data/jhmdb/top1f"
    a_dir_prefixes = ["hitnet_pose_transformer_only_rgb_20250530_"]

    # 來自222
    b_dir = "data/skateformer/top1f"
    b_dir_prefixes = ["SkateFormer_j_2D_20250521_", "SkateFormer_j_2D_20250531_", "SkateFormer_j_2D_20250603_"]

    # === 搜尋符合 prefix 的 a_files ===
    a_files = []
    for subfolder in os.listdir(a_dir):
        full_path = os.path.join(a_dir, subfolder)
        if os.path.isdir(full_path) and any(subfolder.startswith(p) for p in a_dir_prefixes):
            match = glob.glob(
                os.path.join(full_path, "inference/jhmdb_val/result_top1_action_by_frame_confusion_matrix_jhmdb.csv")
            )
            a_files.extend(match)
    a_files = sorted(a_files)

    # === 搜尋符合 prefix 的 b_files ===
    b_files = []
    for subfolder in os.listdir(b_dir):
        full_path = os.path.join(b_dir, subfolder)
        if os.path.isdir(full_path) and any(subfolder.startswith(p) for p in b_dir_prefixes):
            matched = glob.glob(os.path.join(full_path, "*top1f.csv"))
            b_files.extend(matched)
    b_files = sorted(b_files)

    print(f"🔍 找到 {len(a_files)} 個 a_file，{len(b_files)} 個 b_file")

    # === 比對所有 a × b 的組合，找出最小 exceed_count ===
    min_exceed_count = float("inf")
    best_pairs = []

    # === 統計每個 b_file 對應所有 a_file 的平均 F1-score ===
    b_file_f1_scores = defaultdict(list)

    for b_file in b_files:
        b_name = b_file.split(os.sep)[-2]
        for a_file in a_files:
            a_name = a_file.split(os.sep)[-4]  # 例如 hitnet_pose_transformer_only_rgb_*
            print("\n" + "*" * 40)
            print(f"📊 比較 A: {a_name}")
            print(f"        B: {b_name}")
            try:
                result = evaluate_fusion_upper_bound(a_file, b_file)
                f1_scores = [row["f1_upper"] for row in result]
                avg_f1 = np.mean(f1_scores)
                b_file_f1_scores[b_file].append(avg_f1)
                exceed_count = count_exceeded_metrics(result)
                print(f"→ 超過 fusion 上限的項目數：{exceed_count}")
            except Exception as e:
                print(f"❌ 錯誤跳過：{e}")
                continue

            if exceed_count < min_exceed_count:
                min_exceed_count = exceed_count
                best_pairs = [(a_file, b_file)]  # 重新開始紀錄
            elif exceed_count == min_exceed_count:
                best_pairs.append((a_file, b_file))  # 加入同樣最佳者

    # === 最佳結果輸出 ===
    print("\n" + "-" * 40)
    print(f"✅ 所有最佳組合（共 {len(best_pairs)} 組），min exceed count = {min_exceed_count}：")
    for a_file, b_file in best_pairs:
        print(f"A: {a_file}")
        print(f"B: {b_file}")
        print("-" * 20)

    # === 計算每個 b_file 的平均 F1-score ===
    b_file_avg_f1 = {b_file: np.mean(f1_list) for b_file, f1_list in b_file_f1_scores.items() if f1_list}

    # === 額外列出所有最佳組合中 b_file 的平均 F1-score ===
    print("\n" + "=" * 40)
    print("📌 所有最佳組合中的 b_file 之平均 F1-score：")
    for _, b_file in best_pairs:
        avg_f1 = b_file_avg_f1.get(b_file, None)
        if avg_f1 is not None:
            print(f"B: {b_file} → 平均 F1-score: {avg_f1:.3f}")
        else:
            print(f"B: {b_file} → 無 F1-score 資料")

    # === 額外印出 Top 5 b_file（依照平均 F1-score 排序） ===
    print("\n" + "=" * 40)
    print("🏆 平均 F1-score 前五名的 B files：")
    top_5 = sorted(b_file_avg_f1.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (b_file, avg_f1) in enumerate(top_5, 1):
        print(f"{i:>2}. {b_file} → 平均 F1-score: {avg_f1:.3f}")
