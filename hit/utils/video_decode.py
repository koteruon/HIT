import av
import cv2
import numpy as np
from PIL import Image


def av_decode_video(video_path):
    with av.open(video_path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
    return frames


def cv2_decode_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        success, image = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

def cv2_decode_one_image(video_path, counter=1,flag='first'):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if flag == 'first':
        success, image = cap.read()
    else:
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frame-1)
        success, image = cap.read()
    while counter > 0:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        counter -= 1
    return frames

def image_decode(video_path):
    frames = []
    try:
        with Image.open(video_path) as img:
            frames.append(np.array(img.convert('RGB')))
    except BaseException as e:
        raise RuntimeError('Caught "{}" when loading {}'.format(str(e), video_path))

    return frames
