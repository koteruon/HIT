import csv
from collections import defaultdict
import numpy as np


def evaluate_fusion_upper_bound(a_file, b_file):
    def read_method_instances(filepath, method):
        instances = {}
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].startswith("data/"):
                    continue
                key = f"{row[0]},{row[1]}"
                pred = int(row[-3])
                gt = int(row[-1])
                instances[key] = {f"pred_{method}": pred, "gt": gt}
        return instances

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


if __name__ == "__main__":

    # 輸入檔案路徑
    # a_file = "data/bast/hitnet_pose_transformer_stroke_postures_joint_only_rgb_20250511_seed_0004/inference/stroke_postures_val_450/result_top1_action_by_frame_confusion_matrix_stroke_postures.csv"
    # b_file = "data/bast/stroke_postures/SkateFormer_j_2D_20250423/runs-180-16380_top1f.csv"

    a_file = "data/bast/hitnet_pose_transformer_only_rgb_20250518_seed_0009/inference/jhmdb_val/result_top1_action_by_frame_confusion_matrix_jhmdb.csv"
    b_file = "data/bast/jhmdb/SkateFormer_j_2D_20250518_seed_09/runs-92-16284_top1f.csv"

    evaluate_fusion_upper_bound(a_file, b_file)
