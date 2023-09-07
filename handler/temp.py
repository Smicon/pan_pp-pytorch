import cv2
import numpy as np
import torch
import time


def show_pse_line(img_path, data, show=True):
    img = cv_imread(img_path)
    h, w = img.shape[0:2]


    img_line = draw_line2img_pse(img, data)

    if show:
        # cv2.imshow("img", img_line)
        # cv2.waitKey()
        cv2.imwrite("./img.jpg", img_line)


def cv_imread(filePath):
    try:
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    except Exception as e:
        print(str(e) + "    :     " + filePath)
    return cv_img


def draw_line2img_pse(img, data, scale=1):
    ...
    # todo: 文本行中线貌似有点偏上，在文本行收尾两端变化幅度比较大
    if len(data) == 3:

        bboxes, mided, orgin_mid = data
    elif len(data) == 2:
        bboxes, mided = data

    img_copy = img.copy()
    # print(res)
    for index, bbox in enumerate(bboxes):
        color = np.random.randint(0, 256, 3)
        bbox_revise = np.array(bbox) / scale
        cv2.drawContours(img, [bbox_revise.astype('int32')], -1, color.tolist(), 1)
        mid_line = np.array(mided[index]) / scale
        cv2.polylines(img, [mid_line.astype('int32')], False, (128, 0, 0), 1)

        cv2.putText(img, str(index), (int(mid_line[0][0]), int(mid_line[0][1])),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        # cv2.imshow("img", img)
        # cv2.waitKey()

    return img


def temp():
    ...
    x = np.random.randn(1000,1000).astype(np.float16)
    y = np.random.randn(1000,1000).astype(np.float16)
    # x = torch.from_numpy(x).cuda()
    # y = torch.from_numpy(y).cuda()

    count = 0
    # torch.cuda.synchronize()
    t0 = time.time()
    while count < 10:
        count += 1
        # z = x @ y
        z = np.matmul(x,y)
    # torch.cuda.synchronize()
    t1 = time.time()
    print((t1-t0)/count)

if __name__ == "__main__":
    temp()