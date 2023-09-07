import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PAN(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PAN, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):
        outputs = dict()
        cfg.report_speed = True
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes)
            outputs.update(det_loss)
        else:
            print("before upsampling det_out: {}".format(det_out.size()))
            # todo: 确认下这里scale=1 or 4 会不会对 pa 过程产生影响，paper里面是1，我设置成4是想后面提取等高文本行时减少耗时，
            # todo: 但仔细检查了后面的code，在提取等高文本行前，其实feature map已经resize到 img_size, 所以提取文本行那块并没有减少耗时，至于减少的pa
            # todo: 耗时并没有去计算，预计影响还需要进一步确认(相比pse过程)
            # det_out = self._upsample(det_out, imgs.size(), 4)  # source code is 4: https://github.com/whai362/pan_pp.pytorch/blob/master/models/pan.py
            det_out = self._upsample(det_out, imgs.size(), 1)
            print("after upsampling det_out: {}".format(det_out.size()))

            # original code which produces contours
            det_res = self.det_head.get_results(det_out, img_metas, cfg)

            outputs.update(det_res)

        return outputs

    def forward_inference(self,
                          imgs,
                          gt_texts=None,
                          gt_kernels=None,
                          training_masks=None,
                          gt_instances=None,
                          gt_bboxes=None,
                          img_metas=None,
                          cfg=None):
        outputs = dict()
        cfg.report_speed = True
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        # scale=1 or 4 会对 pa 过程产生影响，pa=1 is very slow, pa=4 is fast
        # pa with scale=1 or 4 almost have no effect in final result which is more
        # robust than pse processes in different scale
        # pa is more time-consuming than pse process

        # det_out = self._upsample(det_out, imgs.size(), 4)
        # det_out = self._upsample(det_out, imgs.size(), 1)

        outputs["det_out"] = det_out

        return outputs

