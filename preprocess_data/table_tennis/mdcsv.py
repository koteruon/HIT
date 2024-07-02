import os

import cv2
import pandas as pd


class MDCSV:
    """
    ==========
    actions
    ==========
    serve 1
    stroke 2
    others 3
    stand by 4
    """

    def __init__(self) -> None:
        self.csv_path = r"data/table_tennis/annotations/action_timestamp_tmp.csv"
        self.movie_root = r"data/table_tennis/videos/train"
        self.columns = ["video_id", "time_stamp", "action_id", "entity_id", "frame_stamp"]
        self.movie_path_list = []
        self.step = 1
        self.next_step = self.step - 1
        self.privious_step = -self.step - 1

        # 讀取影片列表
        movie_names = os.listdir(self.movie_root)
        for movie_name in movie_names:
            movie_path = os.path.join(self.movie_root, movie_name)
            self.movie_path_list.append(movie_path)

        # 創建檔案
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as file:
                file.write(",".join(self.columns))
        self.csv_df = pd.read_csv(self.csv_path)

    def get_df_action_id(self, csv_df, video_id, frame_stamp, entity_id):
        left_action_id_series = csv_df[(csv_df["video_id"] == video_id) & (csv_df["frame_stamp"] == frame_stamp) & (csv_df["entity_id"] == entity_id)][
            "action_id"
        ]
        if not left_action_id_series.empty:
            return str(left_action_id_series.iloc[0])
        else:
            return None

    def show_frame(self, frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2):
        if action_1 != 0:
            left_action_id = str(action_1)
        else:
            left_action_id = self.get_df_action_id(self.csv_df, video_id, frame_stamp, 1)

        if action_2 != 0:
            right_action_id = str(action_2)
        else:
            right_action_id = self.get_df_action_id(self.csv_df, video_id, frame_stamp, 2)

        hight = frame.shape[0]
        width = frame.shape[1]
        frame_clone = frame.copy()
        if action_1 != 0:
            color = (0, 0, 255)  # Red color
        else:
            color = (0, 255, 0)  # Green color
        if left_action_id:
            cv2.putText(
                frame_clone, left_action_id, (width // 5, hight // 2), cv2.FONT_HERSHEY_SIMPLEX, 16, color, 4
            )

        if action_2 != 0:
            color = (0, 0, 255)  # Red color
        else:
            color = (0, 255, 0)  # Green color
        if right_action_id:
            cv2.putText(
                frame_clone,
                right_action_id,
                ((width * 3) // 5, hight // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                16,
                color,
                4,
            )

        if is_recording:
            status_text = "RECORD"
            color = (0, 0, 255)  # Red color
        else:
            status_text = "STANDBY"
            color = (0, 255, 0)  # Green color
        cv2.putText(
            frame_clone,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        # 添加當前幀號和總幀數的顯示
        frame_info_text = f"{int(frame_stamp)} / {int(total_frames)}"
        cv2.putText(
            frame_clone,
            frame_info_text,
            (width - 350, 30),  # 調整這裡的坐標來控制文字的位置
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White color
            2,
        )

        frame_s = cv2.resize(frame_clone, (width // 2, hight // 2))
        cv2.imshow("Frame", frame_s)

    def mark_action_timestamps(self, video_id, time_stamp, action_1, action_2, frame_stamp_start, frame_stamp_end):
        # time_stamp
        time_stamp_series = self.csv_df[(self.csv_df["video_id"] == video_id) & (self.csv_df["frame_stamp"] == frame_stamp_start)]["time_stamp"]
        if not time_stamp_series.empty:
            time_stamp = time_stamp_series.iloc[0]

        if action_1 == 0 and action_2 == 0:
            return time_stamp

        for frame_stamp in range(frame_stamp_start, frame_stamp_end + 1):
            # 如果timestamp比原本的還早，就移動整個時間軸
            self.csv_df.loc[(self.csv_df["video_id"] == video_id) & (self.csv_df["time_stamp"] >= time_stamp), "time_stamp"] += 1

            # 如果已經有紀錄了，就直接修改動作值，沒有紀錄則增加一行
            if action_1 != 0:
                new_data = {
                    "video_id": video_id,
                    "time_stamp": time_stamp,
                    "action_id": action_1,
                    "entity_id": 1,
                    "frame_stamp": frame_stamp,
                }
                left_action_id = self.get_df_action_id(self.csv_df, video_id, frame_stamp, 1)
                if left_action_id != None:
                    self.csv_df.loc[
                        (self.csv_df["video_id"] == video_id) & (self.csv_df["frame_stamp"] == frame_stamp) & (self.csv_df["entity_id"] == 1), "action_id"
                    ] = action_1
                else:
                    self.csv_df = self.csv_df.append(new_data, ignore_index=True)

            if action_2 != 0:
                new_data = {
                    "video_id": video_id,
                    "time_stamp": time_stamp,
                    "action_id": action_2,
                    "entity_id": 2,
                    "frame_stamp": frame_stamp,
                }
                right_action_id = self.get_df_action_id(self.csv_df, video_id, frame_stamp, 2)
                if right_action_id != None:
                    self.csv_df.loc[
                        (self.csv_df["video_id"] == video_id) & (self.csv_df["frame_stamp"] == frame_stamp) & (self.csv_df["entity_id"] == 2), "action_id"
                    ] = action_2
                else:
                    self.csv_df = self.csv_df.append(new_data, ignore_index=True)

            self.csv_df = self.csv_df.sort_values(by=["video_id", "time_stamp", "entity_id"])
            time_stamp += 1

        return time_stamp

    def mark_action(self):
        movie_names_idx = 0
        read_video_capture_flag = True
        while 0 <= movie_names_idx < len(self.movie_path_list):
            movie_path = self.movie_path_list[movie_names_idx]
            movie_name = os.path.basename(movie_path)
            video_id = os.path.splitext(movie_name)[0]
            if read_video_capture_flag:
                cap = cv2.VideoCapture(movie_path)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                read_video_capture_flag = False
                time_stamp = 0
                # 是否正在紀錄動作中
                is_recording = False
                action_1 = 0
                action_2 = 0

            ret, frame = cap.read()

            # 检查是否成功读取帧
            if not ret:
                print(f"無法讀取{movie_name}影片")
                movie_names_idx += 1
                read_video_capture_flag = True
                continue

            # 取得當前的FrameNumber
            frame_stamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 顯示畫面
            self.show_frame(frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2)

            # initial values
            entity_id = 1

            # 按下按鍵
            break_flag = False
            entity_one_is_zero = False
            recode_frame_stamp = 0
            while True:
                key = cv2.waitKey(0) & 0xFF
                # 0代表null
                if key == ord("0"):
                    if entity_id == 1:
                        entity_one_is_zero = True
                        action_1 = 0
                        entity_id = 2
                        self.show_frame(frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2)
                        continue
                    elif entity_id == 2:
                        action_2 = 0
                        if entity_one_is_zero:
                            entity_one_is_zero = False
                        else:
                            is_recording = True
                        self.show_frame(frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2)
                # 按下1234
                if key == ord("1") or key == ord("2") or key == ord("3") or key == ord("4"):
                    if entity_id == 1:
                        if key == ord("1"):
                            action_1 = 1
                        elif key == ord("2"):
                            action_1 = 2
                        elif key == ord("3"):
                            action_1 = 3
                        elif key == ord("4"):
                            action_1 = 4
                        entity_id = 2
                        self.show_frame(frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2)
                        continue
                    elif entity_id == 2:
                        if key == ord("1"):
                            action_2 = 1
                        elif key == ord("2"):
                            action_2 = 2
                        elif key == ord("3"):
                            action_2 = 3
                        elif key == ord("4"):
                            action_2 = 4
                        is_recording = True
                        recode_frame_stamp = frame_stamp
                        self.show_frame(frame, video_id, frame_stamp, total_frames, is_recording, action_1, action_2)
                # 空白見結束record
                if key == ord(" ") and is_recording:
                    time_stamp = self.mark_action_timestamps(video_id, time_stamp, action_1, action_2, recode_frame_stamp, frame_stamp)
                    action_1 = 0
                    action_2 = 0
                    is_recording = False
                    key = 83
                # 按下 'q' 键退出循环
                if key == ord("q"):
                    break_flag = True
                # 按下左箭头键（81）跳到上一帧
                elif key == 81:
                    frame_stamp += self.privious_step
                    print(frame_stamp)
                    if frame_stamp < 0:
                        movie_names_idx -= 1
                        read_video_capture_flag = True
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_stamp)
                # 按下右箭头键（83）跳到下一帧
                elif key == 83:
                    frame_stamp += self.next_step
                    print(frame_stamp)
                    if frame_stamp >= total_frames:
                        movie_names_idx += 1
                        read_video_capture_flag = True
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_stamp)
                # 按下 Page Up 键（85）跳到上一帧
                elif key == 85:
                    movie_names_idx -= 1
                    read_video_capture_flag = True
                # 按下 Page Down 键（86）跳到下一帧
                elif key == 86:
                    movie_names_idx += 1
                    read_video_capture_flag = True
                else:
                    continue
                break
            if break_flag:
                break

    def dump(self):
        self.csv_df.to_csv(self.csv_path, encoding="utf8", index=False, header=True)


if __name__ == "__main__":
    mdcsv = MDCSV()
    mdcsv.mark_action()
    mdcsv.dump()
