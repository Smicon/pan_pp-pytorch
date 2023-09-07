import file_util
import Polygon as plg
import numpy as np
import mmcv
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置当前工作目录
os.chdir(script_dir)
project_root = '../../'


pred_root = project_root + 'outputs/submit_ctw'
gt_root = '/app/data/4.0_subset_8000/'


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


def get_gt(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ',')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1, y1] * 14)

        bboxes.append(bbox)
    return bboxes

def get_labelme_result(label_path):
    bboxes = []
    words = []
    with open(label_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        label = json.loads(content)
        shapes = label['shapes']
        for shape in shapes:
            points = shape['points']
            if shape["label"] == "2":
                # print("label : 2 \n polygon will be ignored")
                continue
            if len(points) == 1:
                # print("points number: 1 \n polygon will be ignored")
                continue
            elif shape["shape_type"] == "polygon" and len(points) == 2:
                # print("shape_type: polygon but points number : 2 \n polygon will be ignored")
                continue
            elif shape["shape_type"] == "rectangle" and len(points) == 2:
                # print("shape_type: rectangle and points number : 2 \n polygon will not be ignored")
                points = [points[0], [points[1][0], points[0][1]], points[1], [points[0][0], points[1][1]]]

            points_ = [int(val) for sublist in points for val in sublist]
            bboxes.append(points_)
            words.append(shape["label"])
    return bboxes


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG);


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


if __name__ == '__main__':
    th = 0.8
    pred_list = file_util.read_dir(pred_root)

    tp, fp, npos = 0, 0, 0

    for pred_path in pred_list:
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].replace('.txt', '.json')
        # gts = get_gt(gt_path)
        gts = get_labelme_result(gt_path)
        npos += len(gts)

        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)[:, ::-1]

            pred_p = plg.Polygon(pred)

            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(int(gt.shape[0] / 2), 2)
                gt_p = plg.Polygon(gt)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    # print tp, fp, npos
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
