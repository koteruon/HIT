import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class ShowResult:
    def __init__(self) -> None:
        self.HEIGHT = 1080
        self.WIDTH = 1920

        self.act_th = 0.5
        self.object_th = 0.5
        self.person_th = 0.8
        self.target_fps = 5

        self.act_dict = {
            1: "Serve",
            2: "Stroke",
            3: "Others",
            4: "Stand By",
        }

        self.csv_path = f"/home/chaoen/yoloNhit_calvin/HIT/data/output/hitnet/model_R50_M32_FPS30_4classes/inference/table_tennis_test/result_table_tennis.csv"
        self.v_root = f"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/yolov7_kp_videos/"
        self.lbl_root = f"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/test/"
        self.result_root = f"/home/chaoen/yoloNhit_calvin/HIT/data/output/action_videos/"

        self.frame_span = 30

        self.ball_track_list = [(-1, -1, -1, -1) for _ in range(self.frame_span)]  ## (x, y, w, h)

    def writeActionText(self, frame, data, color, person_list):
        status = 0
        item = data.iloc[0]
        if item.x1 < 320:
            person = 0
        else:
            person = 1
        if len(person_list) >= 2:
            status = 1
            x, y, w, h, score = person_list[person][1:]
            x1, y1 = x - (w / 2), y - (h / 2)
            x_, y_ = x1 * self.WIDTH, y1 * self.HEIGHT
            cv2.putText(
                frame,
                f"{str(self.act_dict[item.action_id])}  {item.conf:.2f}",
                (int(x_), int(y_ - 25)),
                0,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame, status

    def subtask(self, csv_file, v_name, v_path, lbl_dir):
        cap = cv2.VideoCapture(v_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        new_v = cv2.VideoWriter(
            os.path.join(self.result_root, f"4classes_{v_name}_{self.act_th}_30.mp4"),
            fourcc,
            cap.get(cv2.CAP_PROP_FPS) / 2,
            (self.WIDTH, self.HEIGHT),
        )

        lbl_list = sorted(
            os.listdir(lbl_dir),
            key=lambda x: int(x.replace(f"{v_name}_", "").split(".")[0]),
        )

        track_length = int(cap.get(cv2.CAP_PROP_FPS)) // 4
        self.ball_track_list = [(-1, -1, -1, -1) for _ in range(track_length)]  ## (x, y, w, h)

        for count, lbl in enumerate(lbl_list):
            ret, frame = cap.read()
            if not ret:
                break

            with open(os.path.join(lbl_dir, lbl), "r") as txt_f:
                self.ball_track_list.pop(0)
                lines = txt_f.readlines()
                line_split = [x.split() for x in lines]
                paragraph = [
                    [float(word) if idx != 0 else int(word) for idx, word in enumerate(line)] for line in line_split
                ]

                objects_sorted = sorted(paragraph, key=lambda x: (x[0], x[1]))
                person_list = [p for p in objects_sorted if p[5] >= self.person_th and p[0] == 1]
                ball_list = [b for b in objects_sorted if b[5] >= self.object_th and b[0] == 0]
                table_list = [t for t in objects_sorted if t[5] >= self.object_th and t[0] == 2]

                if ball_list != []:
                    ball_list = sorted(ball_list, key=lambda x: x[5], reverse=True)
                    self.ball_track_list.append((ball_list[0][1], ball_list[0][2], ball_list[0][3], ball_list[0][4]))
                else:
                    self.ball_track_list.append((-1, -1, -1, -1))

                past_ = sorted(
                    [
                        b_idx
                        for b_idx, (b_x, b_y, b_w, b_h) in enumerate(self.ball_track_list)
                        if (b_x, b_y, b_w, b_h) != (-1, -1, -1, -1)
                    ],
                    reverse=True,
                )
                if past_ != []:
                    past_idx = max(past_)

                for b_idx in past_:
                    (b_x, b_y, b_w, b_h) = self.ball_track_list[b_idx]
                    b_cx, b_cy = int((b_x) * self.WIDTH), int((b_y) * self.HEIGHT)
                    cv2.circle(frame, (b_cx, b_cy), 5, (97, 220, max(200 - 8 * (len(past_) - b_idx), 0)), 2)

                    # (past_x, past_y, _, _) = self.ball_track_list[past_idx]
                    # past_cx, past_cy = int((past_x)*WIDTH), int((past_y)*HEIGHT)
                    # if np.abs(past_cx - b_cx) < 100 and np.abs(past_cy - b_cy) < 100:
                    #     cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (97, 220, max(200-8*(len(past_)-b_idx), 0)), 5)
                    #     past_idx = b_idx

                dataset = (
                    csv_file.loc[csv_file["frame_stamp"] >= count - self.target_fps]
                    .loc[csv_file["frame_stamp"] < count + self.target_fps]
                    .loc[csv_file["video_id"] == f"{v_name}"]
                    .loc[csv_file["conf"] >= self.act_th]
                )

                data = dataset.loc[csv_file["action_id"] == 1]

                if len(data) > 0:
                    frame, status = self.writeActionText(frame, data, [255, 0, 255], person_list)
                    if status == 1:
                        new_v.write(frame)
                        continue

                data = dataset.loc[csv_file["action_id"] == 2]

                if len(data) > 0:
                    frame, status = self.writeActionText(frame, data, [255, 0, 0], person_list)

                data = (
                    csv_file.loc[csv_file["frame_stamp"] >= count - self.target_fps * 3]
                    .loc[csv_file["frame_stamp"] < count + self.target_fps * 3]
                    .loc[csv_file["video_id"] == f"test/{v_name}"]
                    .loc[csv_file["conf"] >= self.act_th]
                )

                if len(data) >= 2 * 3 and all(pd.unique(data["action_id"]) == 3):
                    all_x = [int(self.ball_track_list[b_idx][0] * self.WIDTH) for b_idx in past_]
                    all_y = [int(self.ball_track_list[b_idx][1] * self.HEIGHT) for b_idx in past_]
                    if len(table_list) > 0:
                        (t_x, t_y, t_w, t_h, t_score) = table_list[0][1:]
                        t_lu_x, t_lu_y = int((t_x - t_w / 2) * self.WIDTH), int((t_y - t_h / 2) * self.HEIGHT)
                        t_rb_x, t_rb_y = int((t_x + t_w / 2) * self.WIDTH), int((t_y + t_h / 2) * self.HEIGHT)
                    else:
                        t_lu_x, t_lu_y = int(self.WIDTH * 3 / 10), int(self.HEIGHT * 3 / 10)
                        t_rb_x, t_rb_y = int(self.WIDTH * 7 / 10), int(self.HEIGHT * 7 / 10)

                    end_flag = False
                    if len(all_x) == 0 and len(all_y) == 0:
                        end_flag = True
                    elif np.mean(all_x) < t_lu_x or np.mean(all_x) > t_rb_x or np.var(all_y) <= 30:
                        end_flag = True

                    if end_flag:
                        color = [64, 255, 0]
                        cv2.rectangle(frame, (40, 20), (200, 60), (0, 0, 0), -1)
                        cv2.putText(
                            frame,
                            f"End !!",
                            (int(50), int(50)),
                            0,
                            1,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        end_flag = False

            new_v.write(frame)

        cap.release()
        new_v.release()

        return None

    def main(self, num_workers=4):
        csv_file = pd.read_csv(self.csv_path)
        csv_file.columns = ["video_id", "frame_stamp", "x1", "y1", "x2", "y2", "action_id", "conf"]
        cols_to_convert = ["frame_stamp", "x1", "y1", "x2", "y2", "action_id", "conf"]
        csv_file[cols_to_convert] = csv_file[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        v_name_list = [v_path.split(".")[0] for v_path in os.listdir(self.v_root)]
        v_path_list = [os.path.join(self.v_root, v_path) for v_path in os.listdir(self.v_root)]
        lbl_dir_list = [os.path.join(self.lbl_root, v_name) for v_name in v_name_list]
        csv_int_v = [csv_file.loc[csv_file["video_id"] == f"{v_name}"] for v_name in v_name_list]

        v_pair = list(zip(csv_int_v, v_name_list, v_path_list, lbl_dir_list))
        # v_pair = sorted(v_pair, key=lambda x: int(x[0].split('-')[-1]))

        mp_tasks_list = v_pair

        for i in tqdm(range(0, len(mp_tasks_list), num_workers)):
            next_list = v_name_list[i : i + num_workers]
            print(f"Now handling {next_list}")

            mp_pool = mp.Pool(processes=num_workers)
            mp_pool.starmap(self.subtask, mp_tasks_list[i : i + num_workers])
            mp_pool.close()
            mp_pool.join()
            # print(i, i+num_workers)
            # print(mp_tasks_list[i : i + num_workers])

        print(f"All videos are finished.")
        print(f"Results are stored at {self.result_root}")

        return None


if __name__ == "__main__":
    t1 = time.time()
    show_result = ShowResult()
    show_result.main()
    t2 = time.time()
    print("Total Time :", t2 - t1)
