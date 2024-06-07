import os

import cv2
import pandas as pd


class MDCSV:
    def __init__(self) -> None:
        self.csv_path = r"data/table_tennis/annotations/action_timestamp_tmp.csv"
        self.movie_root = r"data/table_tennis/videos/train"
        self.columns = ["video_id", "time_stamp", "action_id", "entity_id", "frame_stamp"]
        self.movie_path_list = []
        self.frame_csv_tmp = pd.DataFrame(columns=self.columns)
        self.step = 10
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

    def get_df_action_id(self, csv_df, frame_stamp, entity_id):
        left_action_id_series = csv_df[(csv_df["frame_stamp"] == frame_stamp) & (csv_df["entity_id"] == entity_id)][
            "action_id"
        ]
        if not left_action_id_series.empty:
            return str(left_action_id_series.iloc[0])
        else:
            return None

    def show_frame(self, frame, frame_stamp):
        left_action_id = self.get_df_action_id(self.frame_csv_tmp, frame_stamp, 1)
        right_action_id = self.get_df_action_id(self.frame_csv_tmp, frame_stamp, 2)

        if left_action_id == None:
            left_action_id = self.get_df_action_id(self.csv_df, frame_stamp, 1)
        if right_action_id == None:
            right_action_id = self.get_df_action_id(self.csv_df, frame_stamp, 2)

        hight = frame.shape[0]
        width = frame.shape[1]
        frame_clone = frame.copy()
        if left_action_id:
            cv2.putText(
                frame_clone, left_action_id, (width // 5, hight // 2), cv2.FONT_HERSHEY_SIMPLEX, 16, (0, 255, 0), 4
            )
        if right_action_id:
            cv2.putText(
                frame_clone,
                right_action_id,
                ((width * 3) // 5, hight // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                16,
                (0, 255, 0),
                4,
            )
        frame_s = cv2.resize(frame_clone, (width // 2, hight // 2))
        cv2.imshow("Frame", frame_s)

    def mark_action_timestamps(self, frame, video_id, time_stamp, key, entity_id, frame_stamp):
        if key == ord("1"):
            action_id = 1
        elif key == ord("2"):
            action_id = 2
        elif key == ord("3"):
            action_id = 3
        elif key == ord("4"):
            action_id = 4

        # time_stamp
        time_stamp_series = self.csv_df[
            (self.csv_df["frame_stamp"] == frame_stamp) & (self.csv_df["entity_id"] == entity_id)
        ]["time_stamp"]
        if not time_stamp_series.empty:
            time_stamp = time_stamp_series.iloc[0]

        new_data = {
            "video_id": video_id,
            "time_stamp": time_stamp,
            "action_id": action_id,
            "entity_id": entity_id,
            "frame_stamp": frame_stamp,
        }
        self.frame_csv_tmp = self.frame_csv_tmp.append(new_data, ignore_index=True)
        if entity_id == 2:
            self.csv_df.loc[self.csv_df["time_stamp"] >= time_stamp, "time_stamp"] += 1

            left_action_id = self.get_df_action_id(self.csv_df, frame_stamp, 1)
            right_action_id = self.get_df_action_id(self.csv_df, frame_stamp, 2)
            if left_action_id != None:
                self.csv_df.loc[
                    (self.csv_df["frame_stamp"] == frame_stamp) & (self.csv_df["entity_id"] == 1), "action_id"
                ] = self.frame_csv_tmp[self.frame_csv_tmp["entity_id"] == 1]["action_id"].iloc[0]
            else:
                self.csv_df = pd.concat([self.csv_df, self.frame_csv_tmp[self.frame_csv_tmp["entity_id"] == 1]])
            if right_action_id != None:
                self.csv_df.loc[
                    (self.csv_df["frame_stamp"] == frame_stamp) & (self.csv_df["entity_id"] == 2), "action_id"
                ] = self.frame_csv_tmp[self.frame_csv_tmp["entity_id"] == 2]["action_id"].iloc[0]
            else:
                self.csv_df = pd.concat([self.csv_df, self.frame_csv_tmp[self.frame_csv_tmp["entity_id"] == 2]])

            self.csv_df = self.csv_df.sort_values(by=["video_id", "time_stamp", "entity_id"])
            time_stamp += 1

        self.show_frame(frame, frame_stamp)

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
            self.show_frame(frame, frame_stamp)

            # initial values
            entity_id = 1
            self.frame_csv_tmp = pd.DataFrame(columns=self.columns)

            # 按下按鍵
            break_flag = False
            while True:
                key = cv2.waitKey(0) & 0xFF
                # 按下1234
                if key == ord("1") or key == ord("2") or key == ord("3") or key == ord("4"):
                    time_stamp = self.mark_action_timestamps(frame, video_id, time_stamp, key, entity_id, frame_stamp)
                    if entity_id == 1:
                        entity_id = 2
                        continue
                    elif entity_id == 2:
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
