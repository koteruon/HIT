import argparse
import json
import os
import time

import pandas as pd


class Yolo72coco:
    def __init__(self, is_train=False) -> None:
        """
        ==========
        objects
        ==========
        ball 0 : 2,
        person 1 : 1,
        table 2 : 3,
        """
        """
        ==========
        actions
        ==========
        serve 1
        stroke 2
        others 3
        stand by 4
        """

        self.category = {
            0: 2,
            1: 1,
            2: 3,
        }

        self.persons = [1]
        self.objects = [0, 2]

        self.HEIGHT = 1080
        self.WIDTH = 1920

        self.new_width = 640
        self.new_height = int(round(self.new_width * self.HEIGHT / self.WIDTH / 2) * 2)
        self.ava_dict = {
            "video_id": [],
            "time_stamp": [],
            "lt_x": [],
            "lt_y": [],
            "rb_x": [],
            "rb_y": [],
            "action_id": [],
            "entity_id": [],
        }
        self.person_list = []
        self.object_list = []
        self.flag = False

        if is_train:
            self.action_label_path = r"data/table_tennis/annotations/action_timestamp.csv"
            self.output_csv = r"data/table_tennis/annotations/table_tennis_train.csv"
            self.root_txt_path = r"data/table_tennis/train/"
            self.output_person_json = r"data/table_tennis/boxes/table_tennis_train_det_person_bbox.json"
            self.output_object_json = r"data/table_tennis/boxes/table_tennis_train_det_object_bbox.json"
        else:
            self.action_label_path = r""
            self.output_csv = r""
            self.root_txt_path = r"data/table_tennis/test/"
            self.output_person_json = r"data/table_tennis/boxes/table_tennis_test_det_person_bbox.json"
            self.output_object_json = r"data/table_tennis/boxes/table_tennis_test_det_object_bbox.json"

        self.txt_name_pattern = r"M-4_{}.txt"

    def read_txt_file(self, txt_path, txt_file):
        with open(os.path.join(txt_path, txt_file), "r") as txt_f:
            lines = txt_f.readlines()
            line_split = [x.split() for x in lines]
            paragraph = [
                [float(word) if idx != 0 else int(word) for idx, word in enumerate(line)] for line in line_split
            ]
        return paragraph

    def resize_letterboxd_and_tran_to_coco_format(self, sequence, root_idx, root_dir, txt_idx):
        if txt_idx > 100000:
            raise Exception("txt_idx greater than 100000")
        if len(sequence[1:]) == 5:
            x, y, w, h, score = sequence[1:]
        else:
            x, y, w, h = sequence[1:]
            score = 0.98
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        # lt_x, lt_y, rb_x, rb_y
        x_, y_, w_, h_ = (
            x1 * self.new_width,
            y1 * self.new_height,
            x2 * self.new_width,
            y2 * self.new_height,
        )
        dict_item = {
            "video_id": str(root_dir),
            "image_id": int(txt_idx + (100000 * root_idx)),
            "category_id": self.category[sequence[0]],
            "bbox": [x_, y_, w_, h_],
            "score": score,
        }
        return score, dict_item, x1, y1, x2, y2

    def transform(self, is_train=False, timestamp=None):
        for root_idx, root_dir in enumerate(sorted(os.listdir(self.root_txt_path))):
            txt_path = os.path.join(self.root_txt_path, root_dir)

            if timestamp is None:
                txt_list = sorted(
                    [x for x in os.listdir(txt_path)],
                    key=lambda x: int(x.replace(root_dir + "_", "").split(".")[0]),
                )
            else:
                frame_span = 60
                right_span = frame_span // 2
                left_span = frame_span - right_span
                txt_list = [
                    self.txt_name_pattern.format(x)
                    for x in range(int(timestamp) - left_span, int(timestamp) + right_span)
                    if os.path.exists(os.path.join(txt_path, self.txt_name_pattern.format(x)))
                ]

            if is_train:
                action_df = pd.read_csv(self.action_label_path)
                local_df = action_df.loc[action_df["video_id"] == root_dir]
                time_stamp = pd.unique(local_df["frame_stamp"].values.flatten())

                for txt_idx, txt_file in enumerate(txt_list):
                    if time_stamp.size > 0:
                        stamp_item = time_stamp.min()
                    else:
                        stamp_item = -1

                    paragraph = self.read_txt_file(txt_path, txt_file)
                    for sequence in paragraph:
                        score, dict_item, x1, y1, x2, y2 = self.resize_letterboxd_and_tran_to_coco_format(
                            sequence, root_idx, root_dir, txt_idx
                        )
                        if sequence[0] == 1:
                            if score >= 0.8:
                                self.person_list.append(dict_item)

                            if (x1 + x2) / 2 <= 0.5:
                                entity_id = 1  ## left player
                            else:
                                entity_id = 2  ## right player

                            ## 取stamp
                            if stamp_item - 15 <= txt_idx < stamp_item + 15:
                                stamp_df = local_df.loc[local_df["frame_stamp"] == stamp_item]
                                if stamp_df.values.size > 0:
                                    items = stamp_df.loc[stamp_df["entity_id"] == entity_id].values.flatten().tolist()
                                    action_id = items[2] if items != [] else 3

                                    self.ava_dict["video_id"].append(str(root_dir))
                                    self.ava_dict["time_stamp"].append(int(txt_idx))
                                    self.ava_dict["lt_x"].append(float(x1))
                                    self.ava_dict["lt_y"].append(float(y1))
                                    self.ava_dict["rb_x"].append(float(x2))
                                    self.ava_dict["rb_y"].append(float(y2))
                                    self.ava_dict["action_id"].append(int(action_id))
                                    self.ava_dict["entity_id"].append(int(entity_id))
                            else:
                                action_id = 3

                        else:
                            if score >= 0.5:
                                self.object_list.append(dict_item)

                    if txt_idx >= stamp_item + 15:
                        time_stamp = time_stamp[1:]
            else:
                for txt_idx, txt_file in enumerate(txt_list):
                    paragraph = self.read_txt_file(txt_path, txt_file)
                    for sequence in paragraph:
                        score, dict_item, _, _, _, _ = self.resize_letterboxd_and_tran_to_coco_format(
                            sequence, root_idx, root_dir, txt_idx
                        )
                        if sequence[0] in self.persons:
                            if score >= 0.8:
                                self.person_list.append(dict_item)
                        else:
                            if score >= 0.5:
                                self.object_list.append(dict_item)

        return self.person_list, self.object_list

    def dump(self, is_train=False):
        if is_train:
            ava_df = pd.DataFrame(self.ava_dict)
            ava_df.to_csv(self.output_csv, index=False, header=False)
            print(f"Write person into csv file {self.output_csv} successfully.")

        with open(self.output_person_json, "w") as f:
            json.dump(self.person_list, f)
        print(f"Write person into json file {self.output_person_json} successfully.")

        with open(self.output_object_json, "w") as f:
            json.dump(self.object_list, f)
        print(f"Write object into json file {self.output_object_json} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolov7 format to COCO format")
    parser.add_argument(
        "--train",
        help="Build training dataset",
        action="store_true",
    )
    parser.add_argument(
        "--timestamp",
        help="specify the timestamp",
    )

    args = parser.parse_args()

    t1 = time.time()
    yolo72coco = Yolo72coco(is_train=args.train)
    yolo72coco.transform(is_train=args.train, timestamp=args.timestamp)
    yolo72coco.dump(is_train=args.train)
    t2 = time.time()
    print("Total Time :", t2 - t1)
