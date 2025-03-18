import cv2
import numpy as np

from hit.dataset.datasets import table_tennis_tools


def draw_skeleton(image, keypoints, model):
    keypoints = np.array(keypoints, dtype=np.int32)
    pairs = model.pairs.numpy()

    for idx, (p1, p2) in enumerate(pairs):
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]

        # 確定骨架的顏色
        if idx < model.sections[0]:
            color = model.colors["head"]
        elif idx < model.sections[1]:
            color = model.colors["torso"]
        elif idx < model.sections[2]:
            color = model.colors["left_arm"]
        elif idx < model.sections[3]:
            color = model.colors["right_arm"]
        elif idx < model.sections[4]:
            color = model.colors["left_leg"]
        else:
            color = model.colors["right_leg"]

        cv2.line(image, (x1, y1), (x2, y2), color, 3)

    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image


# 關節點資料
keypoints = [
    [217.90919494628906, 52.27071762084961],
    [222.2791290283203, 48.165855407714844],
    [203.8900146484375, 45.155487060546875],
    [220.07794189453125, 48.1525764465332],
    [203.59024047851562, 48.1525764465332],
    [235.66632080078125, 68.2330551147461],
    [198.7938232421875, 71.23014068603516],
    [261.44708251953125, 82.01964569091797],
    [222.17637634277344, 85.61614990234375],
    [265.0444030761719, 67.33392333984375],
    [243.46051025390625, 70.93042755126953],
    [226.07347106933594, 138.66458129882812],
    [200.29269409179688, 135.66749572753906],
    [245.8587188720703, 191.11358642578125],
    [182.60589599609375, 183.62088012695312],
    [261.44708251953125, 236.36959838867188],
    [173.31283569335938, 226.17950439453125],
]

# 讀取圖片
image_path = "data/jhmdb/videos/Faith_Rewarded_swing_baseball_f_nm_np0_fr_bad_67/00001.png"  # 替換成你的圖片路徑
image = cv2.imread(image_path)

# 確保圖片讀取成功
if image is None:
    raise FileNotFoundError("無法讀取圖片，請檢查路徑是否正確")

# 繪製關節點
for i, (x, y) in enumerate(keypoints):
    x, y = int(x), int(y)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 畫圓點
    cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 標號

model = table_tennis_tools.joint2bone()
image = draw_skeleton(image, keypoints, model)

# 存檔
cv2.imwrite("output.jpg", image)
