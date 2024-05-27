"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/chaoen/yoloNhit_calvin/HIT/data"
    DATASETS = {
        "table_tennis_train": {
            "video_root": "table_tennis/clips/train",
            "ann_file": "table_tennis/annotations/table_tennis_train_min.json",
            "box_file": "",
            "eval_file_paths": {
                "csv_gt_file": "table_tennis/annotations/table_tennis_train.csv",
                "labelmap_file": "table_tennis/annotations/table_tennis_action_list.txt",
                "exclusuion_file": "",
            },
            "object_file": "table_tennis/boxes/table_tennis_train_det_object_bbox.json",
            "keypoints_file": "table_tennis/annotations/table_tennis_train_person_bbox_kpts.json",
        },
        "table_tennis_val": {
            "video_root": "table_tennis/clips/train",
            "ann_file": "table_tennis/annotations/table_tennis_train_min.json",
            "box_file": "table_tennis/boxes/table_tennis_train_det_person_bbox.json",
            "eval_file_paths": {
                "csv_gt_file": "table_tennis/annotations/table_tennis_train.csv",
                "labelmap_file": "table_tennis/annotations/table_tennis_action_list.txt",
                "exclusuion_file": "",
            },
            "object_file": "table_tennis/boxes/table_tennis_train_det_object_bbox.json",
            "keypoints_file": "table_tennis/annotations/table_tennis_train_person_bbox_kpts.json",
        },
        "table_tennis_test": {
            "video_root": "table_tennis/clips/test",
            "ann_file": "table_tennis/annotations/test_min.json",
            "box_file": "table_tennis/boxes/table_tennis_test_det_person_bbox.json",
            "eval_file_paths": {
                "csv_gt_file": "",
                "labelmap_file": "",
                "exclusuion_file": "",
            },
            "object_file": "table_tennis/boxes/table_tennis_test_det_object_bbox.json",
            "keypoints_file": "table_tennis/annotations/table_tennis_test_person_bbox_kpts.json",
        },
        "ava_video_train_v2.2": {
            "video_root": "AVA/clips/trainval",
            "ann_file": "AVA/annotations/ava_train_v2.2_min.json",
            "box_file": "",
            "eval_file_paths": {
                "csv_gt_file": "AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
            },
        },
        "ava_video_val_v2.2": {
            "video_root": "AVA/clips/trainval",
            "ann_file": "AVA/annotations/ava_val_v2.2_min.json",
            "box_file": "AVA/boxes/ava_val_det_person_bbox.json",
            "eval_file_paths": {
                "csv_gt_file": "AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            },
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        if attrs["box_file"] == "":
            box_file = ""
        else:
            box_file = os.path.join(data_dir, attrs["box_file"])
        args = dict(
            video_root=os.path.join(data_dir, attrs["video_root"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            box_file=box_file,
            eval_file_paths={
                key: os.path.join(data_dir, attrs["eval_file_paths"][key]) for key in attrs["eval_file_paths"]
            },
            object_file=os.path.join(data_dir, attrs["object_file"]),
            keypoints_file=os.path.join(data_dir, attrs["keypoints_file"]),
        )
        return dict(factory="DatasetEngine", args=args)
        raise RuntimeError("Dataset not available: {}".format(name))
