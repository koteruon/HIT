import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool
from types import SimpleNamespace

import cv2
import pandas as pd
import tqdm


class ProcessVideos:
    def max_width_n_max_height(self, width, height, targ_size=960):
        if min(width, height) <= targ_size:
            new_width, new_height = width, height
        else:
            new_width = targ_size
            new_height = int(round(new_width * height / width / 2) * 2)
        return new_width, new_height

    def slice_movie_yuv(
        self,
        movie_path,
        clip_root,
        stamp="",
        midframe_root="",
        targ_size=960,
    ):
        probe_args = [
            "ffprobe",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            movie_path,
        ]
        p = subprocess.Popen(probe_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            return "Message from {}:\nffprobe error!".format(movie_path)
        video_stream = json.loads(out.decode("utf8"))["streams"][0]
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        num, denom = map(int, video_stream["r_frame_rate"].split("/"))
        targ_fps = round(num / denom)

        new_width, new_height = self.max_width_n_max_height(width, height, targ_size)

        vid_name = os.path.basename(movie_path)
        vid_id, _ = os.path.splitext(vid_name)
        targ_dir = os.path.join(clip_root, vid_id)
        os.makedirs(targ_dir, exist_ok=True)
        if midframe_root != "":
            frame_targ_dir = os.path.join(midframe_root, vid_id)
            os.makedirs(frame_targ_dir, exist_ok=True)

        count = 0
        cap = cv2.VideoCapture(movie_path)

        ## train/valid keyframe只保留有在action_timestamp中
        if stamp != "":
            csv_file = pd.read_csv(stamp)
            time_stamp = csv_file.loc[csv_file["video_id"] == vid_id]["frame_stamp"].copy()
            time_stamp.reset_index(drop=True, inplace=True)

            time_stamp = pd.unique(time_stamp.values.flatten())
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.resize(frame, (new_width, new_height))
                clip_filename = f'{os.path.join(targ_dir, str(count)+".jpg")}'
                cv2.imwrite(clip_filename, frame)

                if time_stamp.size > 0:
                    stamp_item = time_stamp.min()
                else:
                    stamp_item = -1

                if stamp_item - targ_fps // 2 <= count <= stamp_item + targ_fps // 2:
                    keyframe_filename = f'{os.path.join(frame_targ_dir, str(count)+".jpg")}'
                    cv2.imwrite(keyframe_filename, frame)
                count += 1

                if count >= stamp_item + targ_fps // 2:
                    time_stamp = time_stamp[1:]
            cap.release()

        ## test
        else:
            targ_fps = 5
            while True:
                success, frame = cap.read()
                if not success:
                    print(f"Early Stop at frame: {count}, total frame:{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                    break
                frame = cv2.resize(frame, (new_width, new_height))
                clip_filename = f'{os.path.join(targ_dir, str(count)+".jpg")}'
                cv2.imwrite(clip_filename, frame)
                if (count + 1 + targ_fps // 2) % targ_fps == 0:
                    keyframe_filename = f'{os.path.join(frame_targ_dir, str(count)+".jpg")}'
                    cv2.imwrite(keyframe_filename, frame)
                count += 1
            cap.release()
        return ""


class ProcessVideosPool:
    def __init__(self, args=None, is_train=False):
        if args is None:
            args = SimpleNamespace()
            args.process_num = 4
            if is_train:
                args.movie_root = r"./data/table_tennis/videos/train/"
                args.clip_root = r"./data/table_tennis/clips/train/"
                args.kframe_root = r"./data/table_tennis/keyframes/train/"
                args.time_stamp_file = r"./data/table_tennis/annotations/action_timestamp.csv"
            else:
                args.movie_root = r"./data/table_tennis/videos/test/"
                args.clip_root = r"./data/table_tennis/clips/test/"
                args.kframe_root = r"./data/table_tennis/keyframes/test/"
                args.time_stamp_file = r""

        self.movie_path_list = []
        self.clip_root_list = []
        self.kwargs_list = []
        self.stamp_list = []

        movie_names = os.listdir(args.movie_root)
        for movie_name in movie_names:
            movie_path = os.path.join(args.movie_root, movie_name)
            movie_name_base, movie_name_ext = os.path.splitext(movie_name)
            if not self.is_finish(movie_name_base, args.clip_root, args.kframe_root):
                self.movie_path_list.append(movie_path)
                self.clip_root_list.append(args.clip_root)
                self.kwargs_list.append(dict(midframe_root=args.kframe_root))
                if args.time_stamp_file != "":
                    self.stamp_list.append(args.time_stamp_file)
                else:
                    self.stamp_list.append("")

        self.process_num = args.process_num

    def is_finish(self, movie_name, clip_root, kframe_root):
        # clip
        if not os.path.isdir(os.path.join(clip_root, movie_name)):
            return False

        # kframe
        if not os.path.isdir(os.path.join(kframe_root, movie_name)):
            return False

        return True

    def do_process(self):
        pool = Pool(self.process_num)
        for ret_msg in tqdm.tqdm(
            pool.imap_unordered(
                multiprocess_wrapper,
                zip(zip(zip(self.movie_path_list, self.clip_root_list), self.stamp_list), self.kwargs_list),
            ),
            total=len(self.movie_path_list),
        ):
            if ret_msg != "":
                tqdm.tqdm.write(ret_msg)


def multiprocess_wrapper(args):
    args, kwargs = args
    (x, y), z = args[0], args[1]
    args = [x, y, z]
    process_vides = ProcessVideos()
    return process_vides.slice_movie_yuv(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for processing AVA videos.")
    parser.add_argument(
        "--movie_root",
        required=True,
        help="root directory of downloaded movies",
        type=str,
    )
    parser.add_argument(
        "--clip_root",
        required=True,
        help="root directory to store segmented video clips",
        type=str,
    )
    parser.add_argument(
        "--kframe_root",
        default="",
        help="root directory to store extracted key frames",
        type=str,
    )
    parser.add_argument(
        "--time_stamp_file",
        default="",
        help="file of video time stamp",
        type=str,
    )
    parser.add_argument(
        "--process_num",
        default=4,
        help="the number of processes",
        type=int,
    )
    args = parser.parse_args()
    t1 = time.time()
    processVideosPool = ProcessVideosPool(args=args)
    processVideosPool.do_process()
    t2 = time.time()
    print("Total Time :", t2 - t1)
