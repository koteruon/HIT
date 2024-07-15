import numpy as np
import pandas as pd


def calculate_tiou(frame_range1, frame_range2):
    intersection = max(0, min(frame_range1[1], frame_range2[1]) - max(frame_range1[0], frame_range2[0]) + 1)
    union = max(frame_range1[1], frame_range2[1]) - min(frame_range1[0], frame_range2[0])
    return intersection / union if union > 0 else 0

def get_action_segments(df):
    df = df.sort_values(by=["video_id", "entity_id", "frame_stamp"])
    segments = []
    current_segment = None
    for idx, row in df.iterrows():
        if current_segment is None:
            current_segment = {'video_id': row['video_id'], 'action_id': row['action_id'], 'entity_id': row['entity_id'], 'start_frame': row['frame_stamp'], 'end_frame': row['frame_stamp']}
        elif row['video_id'] == current_segment['video_id'] and row['action_id'] == current_segment['action_id'] and row['entity_id'] == current_segment['entity_id']:
            current_segment['end_frame'] = row['frame_stamp']
        else:
            segments.append(current_segment)
            current_segment = {'video_id': row['video_id'], 'action_id': row['action_id'], 'entity_id': row['entity_id'], 'start_frame': row['frame_stamp'], 'end_frame': row['frame_stamp']}
    if current_segment:
        segments.append(current_segment)
    segments = only_server_and_stroke(segments)
    return segments, pd.DataFrame(segments)

def only_server_and_stroke(segments):
    only_server_and_stroke_segments = []
    for segment in segments:
        if segment["action_id"] == 1 or segment["action_id"] == 2:
            only_server_and_stroke_segments.append(segment)
    return only_server_and_stroke_segments

def get_prediction_action_segments(df, image_width = 640, act_th = 0.5):
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['entity_id'] = df['center_x'].apply(lambda x: 1 if x < image_width / 2 else 2)

    df = df[df['conf'] >= act_th]

    df = df.sort_values(by=["video_id", "entity_id", "frame_stamp", "conf"], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=["video_id", "entity_id", "frame_stamp"], keep='first')
    return get_action_segments(df)

def calculate_metrics(ground_truth_segments, prediction_segments, tiou_threshold=0.5):
    TP, FP, FN = 0, 0, 0

    for idx, gt in ground_truth_segments.iterrows():
        match = False
        gt_frame_range = (gt['start_frame'], gt['end_frame'])
        for _, pred in prediction_segments[(prediction_segments['video_id'] == gt['video_id']) &
                                           (prediction_segments['entity_id'] == gt['entity_id']) &
                                           (prediction_segments['action_id'] == gt['action_id'])].iterrows():
            pred_frame_range = (pred['start_frame'], pred['end_frame'])
            if calculate_tiou(gt_frame_range, pred_frame_range) >= tiou_threshold:
                match = True
                break
        if match:
            TP += 1
        else:
            FN += 1

    FP = max(0, len(prediction_segments) - TP)

    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

# 讀取 CSV 文件
def calculate_action_metrics(logger = None):
    columns = ["video_id", "time_stamp", "action_id", "entity_id", "frame_stamp"]
    ground_truth_df = pd.read_csv('data/table_tennis/annotations/action_timestamp_test.csv')
    prediction_df = pd.read_csv('data/output/hitnet/model_R50_M32_FPS30_4classes/inference/table_tennis_test/result_table_tennis.csv')

    # 獲取動作片段
    for act_th in np.arange(0.35, 0.75, 0.01):
    # for act_th in np.arange(0.55, 0.56, 0.01):
        ground_truth_list, ground_truth_segments = get_action_segments(ground_truth_df)
        prediction_list, prediction_segments = get_prediction_action_segments(prediction_df, act_th = act_th)
        ground_truth_segments.to_csv("gt.csv")
        prediction_segments.to_csv("pred.csv")

        # 計算指標
        accuracy, precision, recall, f1 = calculate_metrics(ground_truth_segments, prediction_segments, tiou_threshold=0.00001)

        print(f"-------------act_th: {act_th:.2f}--------------")
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-Measure: {f1:.2f}')

        if logger:
            logger.info(f"-------------act_th: {act_th:.2f}--------------")
            logger.info(f'Accuracy: {accuracy:.2f}')
            logger.info(f'Precision: {precision:.2f}')
            logger.info(f'Recall: {recall:.2f}')
            logger.info(f'F1-Measure: {f1:.2f}')

if __name__ == '__main__':
    calculate_action_metrics()