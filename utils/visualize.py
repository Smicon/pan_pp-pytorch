import os
import cv2
import numpy as np
# from eval import get_pred
import file_util

script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置当前工作目录
os.chdir(script_dir)

def get_pred(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes

def visualize_from_label(img_paths, output_root, label_dir, show=False, save=False):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for txt_name in os.listdir(label_dir):
        if not txt_name.endswith(".txt"):
            continue

        img_name = txt_name.replace(".txt", ".jpg")
        img_path = os.path.join(img_paths, img_name)
        label_path = os.path.join(label_dir, txt_name)
        preds = get_pred(label_path)
        img = cv2.imread(img_path)
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)[:, ::-1]
            cv2.drawContours(img, [pred], -1, (0, 255, 0), 1)
        if show:
            cv2.imshow("img", img)
            cv2.waitKey()
        if save:
            cv2.imwrite(output_root + img_name, img)


def show_ctw():
    output_root = "../outputs/vis_ctw1500/"
    test_image_dir = "/app/data/4.0_subset_8000"
    test_image_pred_dir = "../outputs/submit_ctw"

    visualize_from_label(test_image_dir, output_root, test_image_pred_dir, show=False, save=True)


def show_cukuang():
    output_root = "../outputs/vis_cukuang_v1/"
    test_image_dir = "../data/CTW1500/test/text_image_cukuang"
    test_image_pred_dir = "../outputs/submit_cukuang_v1"
    test_imgs = os.listdir(test_image_dir)
    test_img_paths = [os.path.join(test_image_dir, i) for i in os.listdir(test_image_dir)]

    visualize_from_label(test_img_paths, output_root, test_image_pred_dir, show=False, save=True)



if __name__ == "__main__":
    show_ctw()
    # show_cukuang()
