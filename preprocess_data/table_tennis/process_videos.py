import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool

import cv2
import pandas as pd
import tqdm


class ProcessVideos:
    def max_width_n_max_height(self, width, height, targ_size=360):
        if min(width, height) <= targ_size:
            new_width, new_height = width, height
        else:
            if height > width:
                new_width = targ_size
                new_height = int(round(new_width * height / width / 2) * 2)
            else:
                new_height = targ_size
                new_width = int(round(new_height * width / height / 2) * 2)
        return new_width, new_height

    def slice_movie_yuv(
        self,
        movie_path,
        clip_root,
        stamp="",
        midframe_root="",
        # start_sec=0,
        # end_sec=48,
        targ_fps=60,
        targ_size=360,
    ):
        # if movie_path != '/home/siplab2/chiahe/HIT/data/table_tennis/videos/test/M-4.MOV':
        #     return ''
        # targ_fps should be int
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

        new_width, new_height = self.max_width_n_max_height(width, height, targ_size)

        vid_name = os.path.basename(movie_path)
        vid_id = vid_name[: vid_name.find(".")]
        targ_dir = os.path.join(clip_root, vid_id)
        os.makedirs(targ_dir, exist_ok=True)
        if midframe_root != "":
            frame_targ_dir = os.path.join(midframe_root, vid_id)
            os.makedirs(frame_targ_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        count = 0
        frames_list = []
        cap = cv2.VideoCapture(movie_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        ## train/valid
        if stamp != "":
            csv_file = pd.read_csv(stamp)
            time_stamp = csv_file.loc[csv_file["video_id"] == vid_id]["frame_stamp"].copy()
            time_stamp.reset_index(drop=True, inplace=True)

            time_stamp = pd.unique(time_stamp.values.flatten())
            while True:
                success, frame = cap.read()
                if not success:
                    break
                # if count >= time_stamp.max()+targ_fps/2:
                #     break
                frame = cv2.resize(frame, (new_width, new_height))
                # frames_list.append(frame)
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

                if count >= stamp_item + 15:
                    time_stamp = time_stamp[1:]
            cap.release()

            # for time_idx, time_zone in enumerate(time_stamp):
            #     start_frame = time_zone - targ_fps//2
            #     end_frame = time_zone + targ_fps//2
            #     clip_filename = f'{os.path.join(targ_dir, str(time_idx)+".mp4")}'
            #     clip = cv2.VideoWriter(clip_filename, fourcc, targ_fps, (new_width, new_height))
            #     while start_frame < 0:
            #         start_frame += 1
            #         clip.write(frames_list[time_zone])
            #     for frame in frames_list[start_frame:end_frame+1]:
            #         clip.write(frame)
            #     while end_frame >= total_frames:
            #         end_frame -= 1
            #         clip.write(frames_list[time_zone])
            #     clip.release()

            #     keyframe_filename = f'{os.path.join(frame_targ_dir, str(time_idx)+".jpg")}'
            #     cv2.imwrite(keyframe_filename, frames_list[time_zone])
        ## test
        else:
            # time_idx = 0
            # while True:
            #     success, frame = cap.read()
            #     if not success:
            #         break
            #     frame = cv2.resize(frame, (new_width, new_height))
            #     frames_list.append(frame)
            #     count += 1
            #     # if count % targ_fps == 0:
            #     if len(frames_list) % targ_fps == 0:
            #         clip_filename = f'{os.path.join(targ_dir, str(time_idx)+".mp4")}'
            #         clip = cv2.VideoWriter(clip_filename, fourcc, targ_fps, (new_width, new_height))
            #         # for frame in frames_list[count-targ_fps:count+1]:
            #         #     clip.write(frame)
            #         for frame in frames_list:
            #             clip.write(frame)
            #         clip.release()

            #         keyframe_filename = f'{os.path.join(frame_targ_dir, str(time_idx)+".jpg")}'
            #         # cv2.imwrite(keyframe_filename, frames_list[count-targ_fps//2])
            #         cv2.imwrite(keyframe_filename, frames_list[targ_fps//2])
            #         time_idx += 1

            #         frames_list.pop(0)
            # cap.release()
            targ_fps = 5
            while True:
                success, frame = cap.read()
                if not success:
                    print(f"Early Stop at frame: {count}, total frame:{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                    break
                # if count >= time_stamp.max()+targ_fps/2:
                #     break
                frame = cv2.resize(frame, (new_width, new_height))
                # frames_list.append(frame)
                clip_filename = f'{os.path.join(targ_dir, str(count)+".jpg")}'
                cv2.imwrite(clip_filename, frame)
                if (count + 1 + targ_fps // 2) % targ_fps == 0:
                    keyframe_filename = f'{os.path.join(frame_targ_dir, str(count)+".jpg")}'
                    cv2.imwrite(keyframe_filename, frame)
                count += 1
            cap.release()
        return ""

    def resize_image(
        self,
        frame,
        timestamp: int,
        clip_root,
        midframe_root,
        targ_fps=60,
        targ_size=360,
    ):
        targ_dir = os.path.join(clip_root, "M-4")
        if midframe_root != "":
            frame_targ_dir = os.path.join(midframe_root, "M-4")
            os.makedirs(frame_targ_dir, exist_ok=True)

        width, height, channels = frame.shape
        new_width, new_height = self.max_width_n_max_height(width, height, targ_size)
        frame = cv2.resize(frame, (new_width, new_height))

        clip_filename = f'{os.path.join(targ_dir, str(timestamp)+".jpg")}'
        cv2.imwrite(clip_filename, frame)
        targ_fps = 5
        if (timestamp + 1 + targ_fps // 2) % targ_fps == 0:
            keyframe_filename = f'{os.path.join(frame_targ_dir, str(timestamp)+".jpg")}'
            cv2.imwrite(keyframe_filename, frame)


def multiprocess_wrapper(args):
    args, kwargs = args
    (x, y), z = args[0], args[1]
    args = [x, y, z]
    process_vides = ProcessVideos()
    return process_vides.slice_movie_yuv(*args, **kwargs)


def main():
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

    movie_path_list = []
    clip_root_list = []
    kwargs_list = []
    stamp_list = []

    movie_names = os.listdir(args.movie_root)
    for movie_name in movie_names:
        movie_path = os.path.join(args.movie_root, movie_name)
        movie_path_list.append(movie_path)
        clip_root_list.append(args.clip_root)
        kwargs_list.append(dict(midframe_root=args.kframe_root))
        if args.time_stamp_file:
            stamp_list.append(args.time_stamp_file)
        else:
            stamp_list.append("")

    pool = Pool(args.process_num)
    for ret_msg in tqdm.tqdm(
        pool.imap_unordered(
            multiprocess_wrapper,
            zip(zip(zip(movie_path_list, clip_root_list), stamp_list), kwargs_list),
        ),
        total=len(movie_path_list),
    ):
        if ret_msg != "":
            tqdm.tqdm.write(ret_msg)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total Time :", t2 - t1)
