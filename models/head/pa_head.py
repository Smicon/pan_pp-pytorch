import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import time
from ..loss import build_loss, ohem_batch, iou
from ..post_processing import pa
from scipy import signal


class PA_Head(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel,
                 loss_emb):
        super(PA_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        # pa
        label = pa(kernels, emb)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        with_rec = hasattr(cfg.model, 'recognition_head')

        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:  # 默认值是16, 是不是太小了
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))
        if with_rec:
            outputs.update(dict(
                label=label,
                bboxes_h=bboxes_h,
                instances=instances
            ))

        return outputs

    @staticmethod
    def get_results_v2(out, img_meta, cfg):
        """
         将PseNET 的后处理过程移植到这里
        :param out:
        :param img_meta:
        :param cfg:
        :return:
        """
        outputs = dict()

        if cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        # emb = emb * text_mask.float()
        emb = emb * text_mask

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        # pa
        t0 = time.time()
        label = pa(kernels, emb)
        t1 = time.time()
        print("pa time: {}".format(t1-t0))

        # image size
        org_img_size = img_meta['org_img_size'][0]  # h0, w0
        img_size = img_meta['img_size'][0]  # h1, w1
        img_shape = (int(img_size[1]), int(img_size[0]))  # w,h

        label_num = np.max(label) + 1
        # label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        # score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        if cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        mides = []

        t3 = time.time()

        for area_num in range(1, label_num):
            area_index = label == area_num
            points = np.array(np.where(area_index)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area / (cfg.test_cfg.scale**2):  # 这个值可能需要根据业务调节下
                label[area_index] = 0
                print("area {} < {}, ignored".format(points.shape[0], cfg.test_cfg.min_area / (cfg.test_cfg.scale**2)))
                continue

            score_i = np.mean(score[area_index])
            if score_i < cfg.test_cfg.min_score:  # 这个值可能需要根据业务调节下
                label[area_index] = 0
                print("score {} < {}, ignored".format(score_i, cfg.test_cfg.min_score))
                continue

            elif cfg.test_cfg.bbox_type == 'poly':
                label_new = np.zeros_like(label)  # shape: h * w
                label_new[area_index] = area_num
                row_vector = np.zeros((1, label.shape[1]))
                label_new = np.r_[row_vector, label_new, row_vector]
                label_diff = np.diff(label_new, axis=0).astype(int)  # 3-4ms
                x_ymid_hight = get_h_mid(label_diff, area_num)
                if x_ymid_hight[2].mean() <= 12 / cfg.test_cfg.scale:
                    print("omit line with height < {}".format(12 / cfg.test_cfg.scale))
                    continue
                bbox, mid = get_equal_height_contours_temp(x_ymid_hight, img_shape, cfg)
                # 如果特征图没有
                bboxes.append(np.round(bbox * 4 * scale).astype(np.int32))
                # bboxes.append(np.ceil(bbox * 4 * scale).astype(np.int32))
                mides.append(np.round(mid * 4 * scale).astype(np.int32))

                # bboxes.append(np.round(bbox * scale).astype(np.int32))
                # mides.append(np.round(mid * scale).astype(np.int32))

                scores.append(score_i)
        print(" for loop takes {} ".format(time.time() - t3))
        outputs.update(dict(
            bboxes=bboxes,
            mides=mides,
            scores=scores
        ))

        return outputs

    @staticmethod
    def get_results_v2_1(out, img_meta, cfg):
        """
         将PseNET 的后处理过程移植到这里, 使用cpu，结果显示没有加速效果，耗时基本没有变化
        :param out:
        :param img_meta:
        :param cfg:
        :return:
        """
        outputs = dict()

        if cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        # pa
        t0 = time.time()
        label = pa(kernels, emb)
        t1 = time.time()
        print("pa time: {}".format(t1 - t0))

        # image size
        org_img_size = img_meta['org_img_size'][0]  # h0, w0
        img_size = img_meta['img_size'][0]  # h1, w1
        img_shape = (int(img_size[1]), int(img_size[0]))  # w,h

        label_num = np.max(label)
        # label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        # score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        if cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        mides = []

        t0 = time.time()
        h, w = label.shape
        batch_label_np = np.zeros([label_num, h + 2, w], dtype=np.int8)
        label_np = label.astype(np.int8)
        batch_area_index = label_np[None,:,:] == np.linspace(1, label_num, label_num, dtype=np.int8)[:,None,None]
        batch_label_np[:, 1:-1, :][batch_area_index] = 1

        batch_label_diff = batch_label_np[:, 1:, :] - batch_label_np[:, :-1, :]

        batch_label_diff = batch_label_diff.astype(int)

        batch_area = np.sum(batch_area_index, axis=(1, 2))
        batch_score_avg = np.sum(score[None, :, :] * batch_area_index, axis=(1, 2)) / batch_area
        index_omit1 = np.where(batch_score_avg < cfg.test_cfg.min_score)[0].tolist()

        index_omit2 = np.where(batch_area < cfg.test_cfg.min_area / (cfg.test_cfg.scale ** 2))[0].tolist() # area点数
        index_omit = index_omit1 + index_omit2
        t1 = time.time()
        print(" diff takes {} ".format(t1-t0))
        for area_num in range(0, label_num):
            if area_num in index_omit:
                print("omit")
                continue

            if cfg.test_cfg.bbox_type == 'poly':
                label_diff = batch_label_diff[area_num]
                x_ymid_hight = get_h_mid(label_diff, 1)
                if x_ymid_hight[2].mean() <= 12 / cfg.test_cfg.scale:
                    print("omit line with height < {}".format(12 / cfg.test_cfg.scale))
                    continue
                bbox, mid = get_equal_height_contours_temp(x_ymid_hight, img_shape, cfg)

                bboxes.append(np.round(bbox * 4 * scale).astype(np.int32))
                # bboxes.append(np.ceil(bbox * 4 * scale).astype(np.int32))
                mides.append(np.round(mid * 4 * scale).astype(np.int32))

                # bboxes.append(np.round(bbox * scale).astype(np.int32))
                # mides.append(np.round(mid * scale).astype(np.int32))

                scores.append(batch_score_avg[area_num])
        t2 = time.time()
        print(" for-loop takes {} ".format(t2 - t1))
        outputs.update(dict(
            bboxes=bboxes,
            mides=mides,
            scores=scores
        ))

        return outputs

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        # embedding loss
        loss_emb = self.emb_loss(embs, gt_instances, gt_kernels[:, -1, :, :], training_masks, gt_bboxes, reduce=False)
        losses.update(dict(
            loss_emb=loss_emb
        ))

        return losses


def get_h_mid(label_diff, area_index):
    # temp = (label_diff * 255).astype(np.uint8)
    # cv2.imshow("up", temp)
    # cv2.waitKey()

    t0 = time.time()
    t1__ = time.time()  # 1_-0:
    t1_ = time.time()  # 1_-0:

    up_xy = np.array(np.where(label_diff == area_index)).transpose((1, 0))[:, ::-1]


    up_xy = up_xy[up_xy[:, 0].argsort()]
    x_new, x_new_index = np.unique(up_xy[:, 0], return_index=True)
    up_xy = up_xy[x_new_index, :]

    t1 = time.time()  # 1-0: 5ms

    dw_xy = np.array(np.where(label_diff == -area_index)).transpose((1, 0))[:, ::-1]
    dw_xy = dw_xy[dw_xy[:, 0].argsort()]
    dw_xy = dw_xy[x_new_index, :]

    # temp1 = (label_diff * -255).astype(np.uint8)
    # cv2.imshow("dowm", temp1)
    # cv2.waitKey()

    t2 = time.time()  # 2-1: 5ms

    area_hight = dw_xy[:, 1] - up_xy[:, 1]
    # area_mid = ((dw_xy[:, 1] + up_xy[:, 1]) / 2).astype(int)  # 向下取整
    area_mid = (dw_xy[:, 1] + up_xy[:, 1]) / 2  # 不向下取整

    t3 = time.time()  # 3-2 : < 1ms

    mid_x_y = np.c_[dw_xy[:, 0], area_mid]
    x_ymid_hight = [mid_x_y[:, 0], mid_x_y[0:, 1], area_hight]

    t4 = time.time()  # 4-3: <1ms

    # print("t1 - t0 : {}".format(t1__ - t0))
    # print("t2 - t1 : {}".format(t1_ - t1__))
    # print("t3 - t2 : {}".format(t3 - t2))
    # print("t4 - t3 : {}".format(t4 - t3))
    return x_ymid_hight


def get_equal_height_contours(x_ymid_hight, img_shape):
    '''
    :param bbox:  文本行的轮廓
    :param img_gray: 灰度图
    :return:  文本行的等高轮廓和中线
    '''

    # crop_img = img_gray[y_min: y_max, x_min+5: x_min+8]
    #
    # 中值滤波
    # GrayImage = cv2.medianBlur(crop_img, 5)
    #
    # 二值化处理
    # th1 = cv2.adaptiveThreshold(GrayImage, 1, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 3, 5)
    #
    # 判断是否切到了字，若是，则向左移动5像素点
    # if np.sum(1-th1) > 0 and x_ymid_hight[0][0] >= 5:
    #     x_ymid_hight[0][0] -= 5

    # 中线平滑

    mid = np.c_[x_ymid_hight[0], x_ymid_hight[1]].astype(np.int16)  # 原始中线
    x_ymid_hight = mid_smooth(x_ymid_hight)

    # 获取1.3倍的平均高度
    height = x_ymid_hight[2].mean()
    half_mean_height = round(1.1 * height / 2)
    # half_mean_height = round(1.1 * height / 2)
    # half_max_height = int(np.max(x_ymid_hight[:,2])/2)
    # x_ymid_hight = rdp(x_ymid_hight, 1.0)

    new_bbox_up = np.c_[np.maximum(x_ymid_hight[0], 0), \
                        np.maximum(x_ymid_hight[1] - half_mean_height, 0)]

    new_bbox_down = np.c_[np.minimum(x_ymid_hight[0], img_shape[0] - 1), np.minimum(x_ymid_hight[1] + half_mean_height, img_shape[1] - 1)]

    new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]  # astype(np.int32) 相当于math.floor
    # new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]

    # return new_bbox, mid
    return new_bbox, np.c_[x_ymid_hight[0], x_ymid_hight[1]]  # astype(np.int32) 相当于math.floor


def get_equal_height_contours_temp(x_ymid_hight, img_shape, cfg):

    # 中线平滑
    # mid = np.c_[x_ymid_hight[0], x_ymid_hight[1]]  # 原始中线
    x_ymid_hight = mid_smooth(x_ymid_hight)

    # 中等长度的文本行一般弯曲程度有限，二次函数能满足要求
    # 文本行轮廓提取出中线后，两侧靠近断点处可能波动较大，曲线拟合时剔除掉这部分点，曲线拟合效果更加稳健
    # 两侧靠近断点处波动较大可能由于轮廓质量低或者文本明显倾斜时提取中线导致
    if 320 / cfg.test_cfg.scale > len(x_ymid_hight[0]) > 64 / cfg.test_cfg.scale:
        start = int(20/cfg.test_cfg.scale)
        end = int(-20/cfg.test_cfg.scale)
        interval = int(4/cfg.test_cfg.scale)  # 如果拟合耗时太长，这里可以增大interval, 让拟合点更加稀疏
        z = np.polyfit(
            x_ymid_hight[0][start:end:interval],
            x_ymid_hight[1][start:end:interval], 2
        )
        p = np.poly1d(z)
        x_ymid_hight[1] = p(x_ymid_hight[0])
    # 较长文本行2次、3次多项式 对较长的含有多个极值点的曲线拟合效果不佳，这种情况下使用滤波平滑可能更好
    elif len(x_ymid_hight[0]) >= 320 / cfg.test_cfg.scale:
        pass
    # 对于非常短的文本暂不做处理，效果如何可能需要进一步观察
    elif len(x_ymid_hight[0]) <= 64 / cfg.test_cfg.scale:
        interval = int(4 / cfg.test_cfg.scale)
        z = np.polyfit(x_ymid_hight[0][::interval], x_ymid_hight[1][::interval], 2)
        p = np.poly1d(z)
        x_ymid_hight[1] = p(x_ymid_hight[0])

    x_ymid_hight[0][-1] = x_ymid_hight[0][-1] + 1
    x_ymid_hight[0][0] = x_ymid_hight[0][0] - 1

    # 多项式拟合可能会导致中线插值到图片范围外
    # if np.any(x_ymid_hight[1] < 0):
    #     print("mid line < 0")
    # if np.any(x_ymid_hight[1] > img_shape[1] - 1):
    #     print("mid line > img_shape[1] - 1 : {}".format(img_shape[1] - 1))
    x_ymid_hight[1] = np.maximum(x_ymid_hight[1], 0)
    x_ymid_hight[1] = np.minimum(x_ymid_hight[1], img_shape[1] - 1)

    # 获取1.3倍的平均高度
    height = x_ymid_hight[2].mean()
    # half_mean_height = round(1.1 * height / 2)
    half_mean_height = 1.1 * height / 2
    # half_mean_height = round(1.1 * height / 2)
    # half_max_height = int(np.max(x_ymid_hight[:,2])/2)
    # x_ymid_hight = rdp(x_ymid_hight, 1.0)

    new_bbox_up = np.c_[
        np.maximum(x_ymid_hight[0], 0),
        np.maximum(x_ymid_hight[1] - half_mean_height - 0.75, 0)
        # np.minimum(np.maximum(x_ymid_hight[1] - half_mean_height, 0), img_shape[1] - 1)
    ]

    new_bbox_down = np.c_[
        np.minimum(x_ymid_hight[0], img_shape[0] - 1),
        np.minimum(x_ymid_hight[1] + half_mean_height + 0.5, img_shape[1] - 1)
        # np.maximum(np.minimum(x_ymid_hight[1] + half_mean_height, img_shape[1] - 1), 0)
    ]

    new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]  # astype(np.int32) 相当于math.floor
    # new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]

    # return new_bbox, mid
    return new_bbox, np.c_[x_ymid_hight[0], x_ymid_hight[1]]  # astype(np.int32) 相当于math.floor


def get_equal_height_contours_temp1(x_ymid_hight, img_shape):

    # 中线平滑
    # mid = np.c_[x_ymid_hight[0], x_ymid_hight[1]]  # 原始中线
    x_ymid_hight = mid_smooth(x_ymid_hight)

    # 中等长度的文本行一般弯曲程度有限，二次函数能满足要求
    # 文本行轮廓提取出中线后，两侧靠近断点处可能波动较大，曲线拟合时剔除掉这部分点，曲线拟合效果更加稳健
    # 两侧靠近断点处波动较大可能由于轮廓质量低或者文本明显倾斜时提取中线导致
    if 300 > len(x_ymid_hight[0]) > 60:
        z = np.polyfit(x_ymid_hight[0][20:-20:4], x_ymid_hight[1][20:-20:4], 2)
        p = np.poly1d(z)
        x_ymid_hight[1] = p(x_ymid_hight[0])
    # 较长文本行2次、3次多项式 对较长的含有多个极值点的曲线拟合效果不佳，这种情况下使用滤波平滑可能更好
    elif len(x_ymid_hight[0]) >= 300:
        pass
    # 对于非常短的文本暂不做处理，效果如何可能需要进一步观察
    elif len(x_ymid_hight[0]) <= 60:
        z = np.polyfit(x_ymid_hight[0][::4], x_ymid_hight[1][::4], 2)
        p = np.poly1d(z)
        x_ymid_hight[1] = p(x_ymid_hight[0])

    # 多项式拟合可能会导致中线插值到图片范围外
    # if np.any(x_ymid_hight[1] < 0):
    #     print("mid line < 0")
    # if np.any(x_ymid_hight[1] > img_shape[1] - 1):
    #     print("mid line > img_shape[1] - 1 : {}".format(img_shape[1] - 1))
    x_ymid_hight[1] = np.maximum(x_ymid_hight[1], 0)
    x_ymid_hight[1] = np.minimum(x_ymid_hight[1], img_shape[1] - 1)

    # 获取1.3倍的平均高度
    height = x_ymid_hight[2].mean()
    half_mean_height = round(1.1 * height / 2)
    # half_mean_height = round(1.1 * height / 2)
    # half_max_height = int(np.max(x_ymid_hight[:,2])/2)
    # x_ymid_hight = rdp(x_ymid_hight, 1.0)

    new_bbox_up = np.c_[
        np.maximum(x_ymid_hight[0], 0),
        np.maximum(x_ymid_hight[1] - half_mean_height, 0)
        # np.minimum(np.maximum(x_ymid_hight[1] - half_mean_height, 0), img_shape[1] - 1)
    ]

    new_bbox_down = np.c_[
        np.minimum(x_ymid_hight[0], img_shape[0] - 1),
        np.minimum(x_ymid_hight[1] + half_mean_height, img_shape[1] - 1)
        # np.maximum(np.minimum(x_ymid_hight[1] + half_mean_height, img_shape[1] - 1), 0)
    ]

    new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]  # astype(np.int32) 相当于math.floor
    # new_bbox = np.r_[new_bbox_up, new_bbox_down[::-1]]

    # return new_bbox, mid
    return new_bbox, np.c_[x_ymid_hight[0], x_ymid_hight[1]]  # astype(np.int32) 相当于math.floor

def mid_smooth(x_ymid_hight):
    # ####### 中线平滑 #######
    mid_m = signal.medfilt(x_ymid_hight[1], 5)  # 中值滤波: when kernel_size is longer than array_length * 2, return 0
    mid_g = gauss(mid_m)  # 高斯滤波  不知道在什么情况下起作用
    mid_g = gauss(mid_g)  # 高斯滤波
    x_ymid_hight[1] = mid_g

    return x_ymid_hight


def gauss(mid):
    # mid size < 5, do not use gauss
    if mid.size < 11:
        return mid

    window = signal.gaussian(M=11, std=100)
    window /= window.sum()
    mid_smoothed = np.convolve(mid, window, mode='valid')
    win_h = int(window.shape[0] / 2)

    # conv result size < win_h, use origin mid padding else use conv result padding
    if mid_smoothed.size < win_h:
        mid_smoothed = np.r_[mid[:win_h], mid_smoothed, mid[mid.shape[0] - win_h:]]
    else:
        mid_smoothed = np.r_[mid_smoothed[:win_h], mid_smoothed, mid_smoothed[mid_smoothed.shape[0] - win_h:]]

    return mid_smoothed