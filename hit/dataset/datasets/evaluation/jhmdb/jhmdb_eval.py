import csv
import operator
import os
import tempfile
import time
from collections import defaultdict
from pprint import pformat

import numpy as np

from .pascal_evaluation import object_detection_evaluation, standard_fields


def save_jhmdb_results(dataset, predictions, output_folder, logger):
    logger.info("Preparing results for AVA format")
    ava_results = prepare_for_jhmdb_detection(predictions, dataset)
    logger.info("Evaluating predictions")
    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "result_jhmdb.csv")
            top1_action_by_frame_confusion_matrix_file_path = os.path.join(
                output_folder, "result_top1_action_by_frame_confusion_matrix_jhmdb.csv"
            )
            top1_action_by_video_confusion_matrix_file_path = os.path.join(
                output_folder, "result_top1_action_by_video_confusion_matrix_jhmdb.csv"
            )
        write_csv(ava_results, file_path, logger)
        write_top1_action_by_frame_confusion_matrix_csv(
            ava_results, top1_action_by_frame_confusion_matrix_file_path, logger, dataset
        )
        avg_precision = write_top1_action_by_video_confusion_matrix_csv(
            ava_results, top1_action_by_video_confusion_matrix_file_path, logger, dataset
        )
        write_files(ava_results, output_folder, logger)
        return avg_precision


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def decode_image_key(image_key):
    return image_key[:-5], image_key[-4:]


def prepare_for_jhmdb_detection(predictions, dataset):
    ava_results = {}
    score_thresh = dataset.action_thresh
    for video_id, prediction in enumerate(predictions):
        video_info = dataset.get_video_info(video_id)
        if len(prediction) == 0:
            continue
        video_width = video_info["width"]
        video_height = video_info["height"]
        prediction = prediction.resize((video_width, video_height))
        prediction = prediction.convert("xyxy")

        prediction = prediction.to("cpu")
        boxes = prediction.bbox.numpy()

        # No background class.
        scores = prediction.get_field("scores").numpy()
        box_ids, action_ids = np.where(scores >= score_thresh)
        boxes = boxes[box_ids, :]
        scores = scores[box_ids, action_ids]
        action_ids = action_ids + 1

        movie_name = video_info["movie"]
        timestamp = video_info["timestamp"]

        clip_key = make_image_key(movie_name, timestamp)

        ava_results[clip_key] = {"boxes": boxes, "scores": scores, "action_ids": action_ids}
    return ava_results


def testlist_to_dict(base_path):
    testlist_path = os.path.join(base_path, "testlist.txt")
    with open(testlist_path) as f:
        data = f.readlines()
    data = [i.strip() for i in data]

    dict_data = {}
    for i in data:
        split_index = i.rfind("/")
        v = i[:split_index]
        k = i[split_index + 1 :]
        # v, k = i.split("/")[:2]
        dict_data[k] = v

    return dict_data


def write_csv(ava_results, csv_result_file, logger):
    print(csv_result_file)
    dict_data = testlist_to_dict(csv_result_file.split("/")[0] + "/jhmdb/annotations")
    start = time.time()
    with open(csv_result_file, "w") as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=",")
        for clip_key in ava_results:
            movie_name, timestamp = decode_image_key(clip_key)
            cur_result = ava_results[clip_key]
            boxes = cur_result["boxes"]
            scores = cur_result["scores"]
            action_ids = cur_result["action_ids"]
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
            for box, score, action_id in zip(boxes, scores, action_ids):
                box_str = ["{:.5f}".format(cord) for cord in box]
                score_str = "{:.5f}".format(score)
                movie_name_with_dir = dict_data[movie_name] + "/" + movie_name
                spamwriter.writerow(
                    [
                        movie_name_with_dir,
                        timestamp,
                    ]
                    + box_str
                    + [action_id, score_str]
                )
    print_time(logger, "write file " + csv_result_file, start)


def write_top1_action_by_frame_confusion_matrix_csv(ava_results, csv_result_file, logger, dataset):
    print(csv_result_file)
    dict_data = testlist_to_dict(csv_result_file.split("/")[0] + "/jhmdb/annotations")

    # 提取資料集中的 distinct 類別數量
    num_classes = len(
        np.unique(dataset.movies_action_gt.val_arr.flatten())
    )  # 類別數，假設 val_arr 存儲所有 ground truth 的 action_id
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)  # 初始化混淆矩陣

    start = time.time()
    with open(csv_result_file, "w") as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=",")
        for clip_key in ava_results:
            movie_name, timestamp = decode_image_key(clip_key)
            cur_result = ava_results[clip_key]
            boxes = cur_result["boxes"]
            scores = cur_result["scores"]
            action_ids = cur_result["action_ids"]
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]

            # 找到最高分的索引
            max_score_index = scores.argmax()

            # 取出對應的資料
            box = boxes[max_score_index]
            score = scores[max_score_index]
            action_id = action_ids[max_score_index]

            # 格式化數據
            box_str = ["{:.5f}".format(cord) for cord in box]
            score_str = "{:.5f}".format(score)
            movie_name_with_dir = dict_data[movie_name] + "/" + movie_name

            # 寫入最高分的行
            spamwriter.writerow([movie_name_with_dir, timestamp] + box_str + [action_id, score_str])

            # 取得 ground truth action_id
            ground_truth_action_id = dataset.movies_action_gt.val_arr[dataset.movies_action_gt.convert_key(movie_name)]

            # 更新混淆矩陣
            confusion_matrix[ground_truth_action_id - 1, action_id - 1] += 1

        # 換行
        spamwriter.writerow([])

        # 計算 Precision 和 Recall
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)

        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FP = confusion_matrix[:, i].sum() - TP
            FN = confusion_matrix[i, :].sum() - TP
            TN = confusion_matrix.sum() - (TP + FP + FN)

            # Precision, Recall, F1-Score
            precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score[i] = (
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
            )

        # 計算所有類別的平均 Precision, Recall, F1-Score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1_score = np.mean(f1_score)

        for i in range(num_classes):
            row = list(confusion_matrix[i])
            row_with_padding = [f"{val:>3}" for val in row]
            spamwriter.writerow(row_with_padding)

        # Precision, Recall, F1-Score 寫入
        spamwriter.writerow([])  # 空行分隔混淆矩陣和指標
        spamwriter.writerow(["Class  ", "Precision", "Recall   ", "F1-Score "])
        for i in range(num_classes):
            spamwriter.writerow([f"{i+1:>7}", f"{precision[i]:9.3f}", f"{recall[i]:9.3f}", f"{f1_score[i]:9.3f}"])

        # 寫入平均 Precision, Recall, F1-Score
        spamwriter.writerow([])  # 空行分隔各類別和總體平均
        spamwriter.writerow(["Average", f"{avg_precision:9.3f}", f"{avg_recall:9.3f}", f"{avg_f1_score:9.3f}"])

    print_time(logger, "write file " + csv_result_file, start)


def write_top1_action_by_video_confusion_matrix_csv(ava_results, csv_result_file, logger, dataset):
    # 提取資料集中的 distinct 類別數量
    num_classes = len(
        np.unique(dataset.movies_action_gt.val_arr.flatten())
    )  # 類別數，假設 val_arr 存儲所有 ground truth 的 action_id
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)  # 初始化混淆矩陣
    start = time.time()

    # 用來儲存每個 movie_name 的最高分資料
    top_results = {}

    for clip_key in ava_results:
        movie_name, timestamp = decode_image_key(clip_key)
        cur_result = ava_results[clip_key]
        boxes = cur_result["boxes"]
        scores = cur_result["scores"]
        action_ids = cur_result["action_ids"]
        assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]

        # 找到當前 clip_key 中的最高分數及對應索引
        max_score_index = scores.argmax()
        max_score = scores[max_score_index]
        box = boxes[max_score_index]
        action_id = action_ids[max_score_index]

        # 如果該 movie_name 還沒有記錄，或者當前分數更高，則更新
        if movie_name not in top_results or max_score > top_results[movie_name]["score"]:
            top_results[movie_name] = {
                "timestamp": timestamp,
                "box": box,
                "score": max_score,
                "action_id": action_id,
            }

        # 取得 ground truth action_id
        ground_truth_action_id = dataset.movies_action_gt.val_arr[dataset.movies_action_gt.convert_key(movie_name)]

        # 更新混淆矩陣
        confusion_matrix[ground_truth_action_id - 1, action_id - 1] += 1

    # 計算 Precision 和 Recall
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        TN = confusion_matrix.sum() - (TP + FP + FN)

        # Precision, Recall, F1-Score
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score[i] = (
            2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        )

    # 計算所有類別的平均 Precision, Recall, F1-Score
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1_score = np.mean(f1_score)

    # 保存混淆矩陣為 CSV
    with open(csv_result_file, "w", newline="") as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=",")
        for i in range(num_classes):
            row = list(confusion_matrix[i])
            row_with_padding = [f"{val:>3}" for val in row]
            spamwriter.writerow(row_with_padding)

        # Precision, Recall, F1-Score 寫入
        spamwriter.writerow([])  # 空行分隔混淆矩陣和指標
        spamwriter.writerow(["Class  ", "Precision", "Recall   ", "F1-Score "])
        for i in range(num_classes):
            spamwriter.writerow([f"{i+1:>7}", f"{precision[i]:9.3f}", f"{recall[i]:9.3f}", f"{f1_score[i]:9.3f}"])

        # 寫入平均 Precision, Recall, F1-Score
        spamwriter.writerow([])  # 空行分隔各類別和總體平均
        spamwriter.writerow(["Average", f"{avg_precision:9.3f}", f"{avg_recall:9.3f}", f"{avg_f1_score:9.3f}"])

    print_time(logger, "write confusion matrix CSV file " + csv_result_file, start)
    return avg_precision


def write_files(ava_results, csv_result_file, logger):
    start = time.time()
    if not os.path.exists(csv_result_file + "/detections"):
        os.mkdir(csv_result_file + "/detections")

    dict_data = testlist_to_dict(csv_result_file.split("/")[0] + "/jhmdb/annotations")

    for clip_key in ava_results:
        movie_name, timestamp = decode_image_key(clip_key)
        filename = movie_name + "_" + timestamp.zfill(5) + ".txt"
        cur_result = ava_results[clip_key]
        boxes = cur_result["boxes"]
        scores = cur_result["scores"]
        action_ids = cur_result["action_ids"]
        assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]

        detection_path = os.path.join(csv_result_file + "/detections", filename)

        with open(detection_path, "w+") as f_detect:
            for box, score, action_id in zip(boxes, scores, action_ids):
                if float(score) > 0:
                    box_str = [str(round(cord)) for cord in box]
                    score_str = "{:.5f}".format(score)
                    f_detect.write(
                        str(int(action_id))
                        + " "
                        + score_str
                        + " "
                        + box_str[0]
                        + " "
                        + box_str[1]
                        + " "
                        + box_str[2]
                        + " "
                        + box_str[3]
                        + "\n"
                    )


def print_time(logger, message, start):
    logger.info("==> %g seconds to %s", time.time() - start, message)
