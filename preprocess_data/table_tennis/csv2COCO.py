import argparse
import itertools
import json
import os
import time
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


class Csv2COCOJson:
    def __init__(self, is_train=False) -> None:
        self._movie_list = None
        if is_train:
            self._csv_path = "data/table_tennis/annotations/table_tennis_train.csv"
            self._img_root = "data/table_tennis/keyframes/train/"
            self._movie_path = "data/table_tennis/videos/train"
            self._json_path = "data/table_tennis/annotations/table_tennis_train.json"
            self._min_json_path = "data/table_tennis/annotations/table_tennis_train_min.json"
        else:
            self._csv_path = None
            self._img_root = "data/table_tennis/keyframes/test/"
            self._movie_path = "data/table_tennis/videos/test"
            self._json_path = "data/table_tennis/annotations/test.json"
            self._min_json_path = "data/table_tennis/annotations/test_min.json"

    @property
    def csv_path(self):
        return self._csv_path

    @property
    def movie_list(self):
        return self._movie_list

    @property
    def img_root(self):
        return self._img_root

    @property
    def movie_path(self):
        return self._movie_path

    @property
    def json_path(self):
        return self._json_path

    @property
    def min_json_path(self):
        return self._min_json_path

    @csv_path.setter
    def csv_path(self, csv_path):
        self._csv_path = csv_path

    @movie_list.setter
    def movie_list(self, movie_list):
        self._movie_list = movie_list

    @img_root.setter
    def img_root(self, img_root):
        self._img_root = img_root

    @movie_path.setter
    def movie_path(self, movie_path):
        self._movie_path = movie_path

    @json_path.setter
    def json_path(self, json_path):
        self._json_path = json_path

    @min_json_path.setter
    def min_json_path(self, min_json_path):
        self._min_json_path = min_json_path

    def csv2COCOJson(self):
        ann_df = pd.read_csv(self._csv_path, header=None)
        movie_ids = {}
        if self._movie_list:
            with open(self._movie_list) as movief:
                for idx, line in enumerate(movief):
                    # name = line[: line.find('.')]
                    name = line.split("/")[1].split()[0]
                    movie_ids[name] = idx
        else:
            for idx, movie_name in enumerate(sorted(os.listdir(self._movie_path))):
                name = os.path.splitext(movie_name)[0]
                movie_ids[name] = idx
        movie_infos = {}
        iter_num = len(ann_df)

        for rows in tqdm(ann_df.itertuples(), total=iter_num, desc="Calculating  info"):
            _, movie_name, timestamp, x1, y1, x2, y2, action_id, person_id = rows

            timestamp = int(timestamp)
            if movie_name not in movie_infos:
                movie_infos[movie_name] = {}
                movie_infos[movie_name]["img_infos"] = {}
                img_path = os.path.join(movie_name, "{}.jpg".format(str(timestamp)))
                movie_infos[movie_name]["size"] = Image.open(os.path.join(self._img_root, img_path)).size
                movie_info = movie_infos[movie_name]
                img_infos = movie_info["img_infos"]
                width, height = movie_info["size"]
                movie_id = movie_ids[movie_name] * 100000
                # tid_range = len(os.listdir(os.path.join(img_root, movie_name)))

                tid_range = pd.unique(ann_df.loc[ann_df[0] == movie_name][1].values.flatten())
                # for tid in range(tid_range):
                for tid in tid_range:
                    if tid > 100000:
                        raise Exception("tid greater than 100000")
                    img_id = movie_id + tid
                    img_path = os.path.join(movie_name, "{}.jpg".format(tid))
                    video_path = os.path.join(movie_name, "{}.mp4".format(tid))
                    img_infos[tid] = {
                        "id": int(img_id),
                        "img_path": str(img_path),
                        "video_path": str(video_path),
                        "height": height,
                        "width": width,
                        "movie": movie_name,
                        "timestamp": int(tid),
                        "annotations": {},
                    }

            img_info = movie_infos[movie_name]["img_infos"][timestamp]
            if person_id not in img_info["annotations"]:
                box_id = img_info["id"] * 1000 + person_id
                box_w, box_h = x2 - x1, y2 - y1
                width = img_info["width"]
                height = img_info["height"]
                real_x1, real_y1 = x1 * width, y1 * height
                real_box_w, real_box_h = box_w * width, box_h * height
                area = real_box_w * real_box_h
                img_info["annotations"][person_id] = {
                    "id": box_id,
                    "image_id": img_info["id"],
                    "category_id": 1,
                    "action_ids": [],
                    "person_id": person_id,
                    "bbox": list(
                        map(
                            lambda x: round(x, 2),
                            [real_x1, real_y1, real_box_w, real_box_h],
                        )
                    ),
                    "area": round(area, 5),
                    "keypoints": [],
                    "iscrowd": 0,
                }
            box_info = img_info["annotations"][person_id]
            box_info["action_ids"].append(action_id)

        jsondata = {}
        jsondata["categories"] = [{"supercategory": "person", "id": 1, "name": "person"}]

        anns = [
            img_info.pop("annotations").values()
            for movie_info in movie_infos.values()
            for img_info in movie_info["img_infos"].values()
        ]
        anns = list(itertools.chain.from_iterable(anns))
        jsondata["annotations"] = anns
        imgs = [movie_info["img_infos"].values() for movie_info in movie_infos.values()]
        imgs = list(itertools.chain.from_iterable(imgs))
        jsondata["images"] = imgs

        return jsondata

    def genCOCOJson(self, timestamp=None):
        movie_ids = {}
        if self._movie_list:
            with open(self._movie_list) as movief:
                for idx, line in enumerate(movief):
                    # name = line[: line.find('.')]
                    name = line.split("/")[1].split()[0]
                    movie_ids[name] = idx
        else:
            for idx, movie_name in enumerate(sorted(os.listdir(self._movie_path))):
                name = os.path.splitext(movie_name)[0]
                movie_ids[name] = idx
        movie_infos = {}

        for movie_name in tqdm(movie_ids):
            movie_root = os.path.join(self._img_root, movie_name)
            if timestamp:  # 如果有指定timestamp
                # right_span = self.frame_span // 2
                # left_span = self.frame_span - right_span
                # image_list = [
                #    str(x) + ".jpg"
                #    for x in range(int(timestamp) - left_span, int(timestamp) + right_span)
                #    if os.path.exists(os.path.join(self._img_root, movie_name, str(x) + ".jpg"))
                # ]
                image_list = [str(timestamp) + ".jpg"]
            else:
                image_list = sorted(os.listdir(movie_root), key=lambda x: x.split(".")[0])

            movie_infos[movie_name] = {}
            movie_infos[movie_name]["img_infos"] = {}
            img_path = os.path.join(movie_name, image_list[0])
            movie_infos[movie_name]["size"] = Image.open(os.path.join(self._img_root, img_path)).size
            movie_info = movie_infos[movie_name]
            img_infos = movie_info["img_infos"]
            width, height = movie_info["size"]
            movie_id = movie_ids[movie_name] * 100000
            # tid_range = len(os.listdir(os.path.join(img_root, movie_name)))
            tid_range = [int(x.split(".")[0]) for x in image_list]
            # for tid in range(tid_range):
            for tid in tid_range:
                if tid > 100000:
                    raise Exception("tid greater than 100000")
                img_id = movie_id + tid
                img_path = os.path.join(movie_name, "{}.jpg".format(tid))
                video_path = os.path.join(movie_name, "{}.mp4".format(tid))
                img_infos[tid] = {
                    "id": img_id,
                    "img_path": img_path,
                    "video_path": video_path,
                    "height": height,
                    "width": width,
                    "movie": movie_name,
                    "timestamp": tid,
                    "annotations": {},
                }

        jsondata = {}
        jsondata["categories"] = [{"supercategory": "person", "id": 1, "name": "person"}]

        anns = [
            img_info.pop("annotations").values()
            for movie_info in movie_infos.values()
            for img_info in movie_info["img_infos"].values()
        ]
        anns = list(itertools.chain.from_iterable(anns))
        jsondata["annotations"] = anns
        imgs = [movie_info["img_infos"].values() for movie_info in movie_infos.values()]
        imgs = list(itertools.chain.from_iterable(imgs))
        jsondata["images"] = imgs
        return jsondata

    def dump(self, jsondata):
        tic = time.time()
        print("Writing into json file...")
        with open(self._json_path, "w") as jsonf:
            json.dump(jsondata, jsonf, indent=4)
        print("Write json dataset into json file {} successfully.".format(self._json_path))

        with open(self._min_json_path, "w") as jsonminf:
            json.dump(jsondata, jsonminf)
        print("Write json dataset with no indent into json file {} successfully.".format(self._min_json_path))

        print("Done (t={:0.2f}s)".format(time.time() - tic))


def main():
    parser = argparse.ArgumentParser(description="Generate coco format json for AVA.")
    parser.add_argument(
        "--train",
        help="Build training annotation",
        action="store_true",
    )
    parser.add_argument(
        "--csv_path",
        default="",
        help="path to csv file",
        type=str,
    )
    parser.add_argument(
        "--movie_list",
        help="path to movie list",
        type=str,
    )
    parser.add_argument(
        "--img_root",
        help="root directory of extracted key frames",
        type=str,
    )
    parser.add_argument(
        "--json_path",
        default="",
        help="path of output json",
        type=str,
    )
    parser.add_argument(
        "--min_json_path",
        default="",
        help="path of output minimized json",
        type=str,
    )
    args = parser.parse_args()

    csv_2_coco_json = Csv2COCOJson(is_train=args.train)

    if args.movie_list:
        csv_2_coco_json.movie_list = args.movie_list
    if args.img_root:
        csv_2_coco_json.img_root = args.img_root
    if args.json_path:
        csv_2_coco_json.json_path = args.json_path
    if args.min_json_path:
        csv_2_coco_json.min_json_path = args.min_json_path
    if args.csv_path:
        csv_2_coco_json.csv_path = args.csv_path

    if args.train:
        jsondata = csv_2_coco_json.csv2COCOJson()
    else:
        jsondata = csv_2_coco_json.genCOCOJson()
    csv_2_coco_json.dump(jsondata)


if __name__ == "__main__":
    main()
