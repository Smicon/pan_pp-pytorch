import os
import time
import sys
import torch
from models import build_model
from models.utils import fuse_module
from mmcv import Config
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from handler.temp import show_pse_line
from models.head import PA_Head


def preprocess(img, max_length):
    ...
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_length:
        scale = max_length / long_side
        return cv2.resize(img, None, None, fx=scale, fy=scale), scale
    else:
        return img, 1


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)  # fixme: plus 0.5 equals to round 四舍五入
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def scale_aligned(img, scale):
    # set w, h to multiple of 32
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)  # math.floor
    w = int(w * scale + 0.5)  # math.floor
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))  # 这里可能不是等比例变形
    return img


def scale_aligned_long(img, long_size=960):
    h, w = img.shape[0:2]
    long_side = max(h, w)
    if long_side > long_size:
        scale = long_size * 1.0 / min(h, w)
        h = int(h * scale + 0.5)  # fixme: plus 0.5
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        return img


def get_img_v3(img):
    """
    :param img:
    :return:
    """
    try:
        # img = cv2.resize(img, (734, 272))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, [2, 1, 0]]
        # cv2.imshow("up1", img)
        # cv2.waitKey()
        pass
    except Exception as e:
        print(e)
        raise
    # time1 = time.time()

    img = img.swapaxes(1, 2).swapaxes(0, 1)  # C H W
    scaled_img = torch.from_numpy(img)
    scaled_img = scaled_img.cuda().float()
    scaled_img = scaled_img / 255.
    # todo: compare
    # mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda()
    # std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda()
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float32)).cuda()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225], dtype=np.float32)).cuda()
    # fixme: sub_  div_ 节约显存，速度呢？
    scaled_img = (scaled_img - mean[:, None, None]) / std[:, None, None]
    # scaled_img.sub_(mean[:, None, None]).div_(std[:, None, None])

    # time2 = time.time()
    # print(time2 - time1)
    return scaled_img.unsqueeze(0)

def get_img_v3_1(img, half_precision):
    """
    :param img:
    :return:
    """
    try:
        # img = cv2.resize(img, (734, 272))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, [2, 1, 0]]
        # cv2.imshow("up1", img)
        # cv2.waitKey()
        pass
    except Exception as e:
        print(e)
        raise
    # time1 = time.time()

    img = img.swapaxes(1, 2).swapaxes(0, 1)  # C H W
    scaled_img = torch.from_numpy(img)
    if half_precision:
        scaled_img = scaled_img.cuda().half()
    else:
        scaled_img = scaled_img.cuda().float()
    scaled_img = scaled_img / 255.
    # todo: compare
    # mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda()
    # std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda()
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float16 if half_precision else np.float32)).cuda()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225], dtype=np.float16 if half_precision else np.float32)).cuda()
    # fixme: sub_  div_ 节约显存，速度呢？
    scaled_img = (scaled_img - mean[:, None, None]) / std[:, None, None]
    # scaled_img.sub_(mean[:, None, None]).div_(std[:, None, None])

    # time2 = time.time()
    # print(time2 - time1)
    return scaled_img.unsqueeze(0)


class Service():

    def __init__(self,
                 config,
                 checkpoint_path,
                 report_speed=False
                 ):
        self.half_precision = True
        print("using half precison: {}".format(self.half_precision))
        cfg = Config.fromfile(config)
        for d in [cfg, cfg.data.test]:
            d.update(dict(
                report_speed=report_speed
            ))
        self.cfg = cfg
        self.detector = Detector(self.cfg, checkpoint=checkpoint_path, half_precision=self.half_precision)
        self.short_size = cfg.data.test.short_size

    def prepare_data(self, img):
        # todo: 这里是不是少了这一步
        img = img[:, :, [2, 1, 0]]

        img_meta = dict(
            org_img_size=torch.from_numpy(np.array(img.shape[:2])).unsqueeze(0)
        )
        # 原始的pan的图片处理流程可能不适用于我们的图片
        # img = scale_aligned_short(img, self.short_size)
        img = scale_aligned(img, 1)  # w,h resize to multiple of 32
        img_meta.update(dict(
            img_size=torch.from_numpy(np.array(img.shape[:2])).unsqueeze(0)
        ))
        # print(img_meta)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        # img = img.cuda()
        data = dict(
            imgs=img.unsqueeze(0),
            img_metas=img_meta
        )
        data.update(dict(
            cfg=self.cfg
        ))
        return data

    def prepare_data_v1(self, img):
        # todo: 这里是不是少了这一步
        # img = img[:, :, [2, 1, 0]]

        img_meta = dict(
            org_img_size=torch.from_numpy(np.array(img.shape[:2])).unsqueeze(0)
        )
        # 原始的pan的图片处理流程可能不适用于我们的图片
        # img = scale_aligned_short(img, self.short_size)
        img = scale_aligned(img, 1)  # w,h resize to multiple of 32
        img_meta.update(dict(
            img_size=torch.from_numpy(np.array(img.shape[:2])).unsqueeze(0)
        ))
        # print(img_meta)

        # img = Image.fromarray(img)
        # img = img.convert('RGB')
        # img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        img = get_img_v3_1(img, self.half_precision)

        data = dict(
            imgs=img,
            img_metas=img_meta
        )
        data.update(dict(
            cfg=self.cfg
        ))
        return data

    def run(self, image_data):

        with torch.no_grad():
            data = self.prepare_data(image_data)
            result = self.detector.detect(data)
        return [result["bboxes"], result["mides"]]

    def run_v1(self, image_data):

        with torch.no_grad():
            t0 = time.time()
            data = self.prepare_data_v1(image_data)
            t1 = time.time()
            print("prepare_data_v1 time: {}".format(t1-t0))
            result = self.detector.detect_v1(data)
        return [result["bboxes"], result["mides"]]

    def run_debug(self, image_data):

        with torch.no_grad():
            img = self.prepare_data_v1(image_data)
            result = self.detector.detect_debug(img)
        return result["bboxes"]


class Detector():

    def __init__(self, cfg, checkpoint, half_precision):
        self.half_precision = half_precision
        model = build_model(cfg.model)
        # model = model.cuda()

        if checkpoint is not None:
            if os.path.isfile(checkpoint):
                print("Loading model and optimizer from checkpoint '{}'".format(checkpoint))
                sys.stdout.flush()

                checkpoint = torch.load(checkpoint)

                d = dict()
                for key, value in checkpoint['state_dict'].items():
                    tmp = key[7:]
                    d[tmp] = value
                model.load_state_dict(d)
            else:
                print("No checkpoint found at '{}'".format(checkpoint))
                raise FileNotFoundError

        # fuse conv and bn
        self.model = fuse_module(model)  # fixme: 看不懂
        if self.half_precision:
            self.model.half()
        self.model.cuda()
        self.model.eval()

    def detect(self, data):
        ...
        data['imgs'] = data['imgs'].cuda()
        outputs = self.model(**data)
        return outputs

    def detect_v1(self, data):
        # torch.cuda.synchronize()
        t0 = time.time()
        outputs = self.model.forward_inference(**data)
        # torch.cuda.synchronize()
        t1 = time.time()
        print("forward_inference time: {}".format(t1 - t0))
        det_res = PA_Head.get_results_v2(outputs["det_out"], data["img_metas"], data["cfg"])
        t2 = time.time()
        print("get_results_v2 time: {}".format(t2 - t1))
        del outputs["det_out"]
        outputs.update(det_res)
        return outputs


    def detect_debug1(self, data):
        ...
        t0 = time.time()
        # data.update({"scale": 1})
        outputs = self.model.forward_inference(**data)
        t1 = time.time()
        print("debug forward_inference time: {}".format(t1 - t0))

        det_res = self.model.det_head.get_results(outputs["det_out"], data["img_metas"], data["cfg"])
        del outputs["det_out"]
        outputs.update(det_res)
        outputs["bboxes"] = [i.reshape([-1, 2])for i in outputs["bboxes"]]
        return outputs

    def detect_debug(self, data):
        ...
        #
        t0 = time.time()
        # data.update({"scale": 1})
        outputs = self.model(**data)
        t1 = time.time()
        print("debug forward time: {}".format(t1 - t0))

        outputs["bboxes"] = [i.reshape([-1, 2])for i in outputs["bboxes"]]
        return outputs


if __name__ == "__main__":
    ...
    config = r"../config/pan/pan_r18_ctw_custom_v1.py"
    ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/pan_r18_ctw_custom_v1/checkpoint.pth.tar"
    service = Service(config=config, checkpoint_path=ckpt)

    img_path = "../data/CTW1500/test/text_image_cukuang/material_PAAPhoto_70S5C9A0B192M_PAAPhoto20200708203727_1001_1057_341_2448_1338.jpg"
    img = cv2.imread(img_path)

    result = service.run(img)
    # print(result)
    data = result
    # show_pse_line(img_path, data, True)
