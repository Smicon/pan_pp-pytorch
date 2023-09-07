import json
import base64
import torch
import cv2
import numpy as np
import time
import tornado.ioloop
import tornado.web
import os, sys
import ujson
import pybase64
# from turbojpeg import TurboJPEG
# jpeg = TurboJPEG()

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)
path = os.path.dirname(path)
sys.path.insert(0, path)
from handler.service import Service, preprocess

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class MainHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.write("Not implement Get Function")

    def post(self, *args, **kwargs):
        ######################## start time ##########################
        start_time = time.time()
        params = self.request.body.decode('utf-8')
        # params = json.loads(params, strict=False)
        params = ujson.loads(params)

        # img_data = base64.b64decode(params["img"])
        img_data = pybase64.b64decode(params["img"])

        # t0 = time.time()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        # img = jpeg.decode(img_data)

        # print("cv2.imdecode TIME IS {} ".format(time.time() - t0))
        print("img shape: {}".format(img.shape))
        img, scale = preprocess(img, 1280)  # 这个要设置的小一点，960 或者 640 中间的某一个值吧，控制下显存和耗时
        mid_time = time.time()
        print("mid TIME IS {} ".format(mid_time - start_time))

        bboxes, mides = service.run_v1(img)
        print(" text lines number: {}".format(len(bboxes)))
        print("service.run TIME IS {} ".format(time.time() - mid_time))

        if scale == 1:
            pass
            bboxes = [bbox.tolist() for bbox in bboxes]
            mides = [mid.tolist() for mid in mides]
        else:
            # print(" scale : {} ".format(scale))
            bboxes = [(bbox.astype(np.float) / scale).astype(np.int).tolist() for bbox in bboxes]
            mides = [(mid.astype(np.float) / scale).astype(np.int).tolist() for mid in mides]

        end_time = time.time()
        print("TOTAL TIME IS {} \n".format(end_time - start_time))
        test = ujson.dumps([bboxes, mides])
        # test = json.dumps([bboxes, mides], cls=NpEncoder, ensure_ascii=False)
        self.write(test)


class MainHandlerDebug(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.write("Not implement Get Function")

    def post(self, *args, **kwargs):
        ######################## start time ##########################
        start_time = time.time()
        params = self.request.body.decode('utf-8')
        params = json.loads(params, strict=False)

        img_data = base64.b64decode(params["img"])
        img = cv2.imdecode(np.fromstring(img_data, np.uint8), cv2.IMREAD_COLOR)

        bboxes = service.run_debug(img)
        print("/pan/debug: text lines number: {}".format(len(bboxes)))

        end_time = time.time()
        print("TOTAL TIME IS {} \n".format(end_time - start_time))
        test = json.dumps([bboxes], cls=NpEncoder, ensure_ascii=False)
        self.write(test)


def make_app():
    return tornado.web.Application([
        (r"/pan", MainHandler),
        (r"/pan/debug", MainHandlerDebug),
    ])


def main():
    app = make_app()
    app.listen(6007)
    print("Sever start at 6007!")
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    with torch.no_grad():
        # 9.2
        config = r"../config/pan/pan_r18_ctw_custom_v1.py"
        ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/pan_r18_ctw_custom_v1_bk/checkpoint.pth.tar"

        # config = r"../config/pan/pan_r18_ctw_custom_v1.py"
        # 9.21
        # ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/pan_r18_ctw_custom_v1/checkpoint.pth.tar"

        #
        # config = r"../config/pan/pan_r18_ctw_custom_v1.py"
        # 9.22
        # ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/pan_r18_ctw_custom_v1_PAN_CUSTOM_V2_PSE/checkpoint.pth.tar"

        # 9.23
        config = r"../config/pan/pan_r18_ctw_custom_test.py"
        ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/test/checkpoint.pth.tar"

        # 9.2301
        config = r"../config/pan/pan_r18_ctw_custom_test.py"
        ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/test0.1/checkpoint.pth.tar"

        # 9.231
        # config = r"../config/pan_pp/pan_pp_r18_ctw.py"
        # ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/test1/checkpoint.pth.tar"

        # 9.232
        config = r"../config/pan_pp/pan_pp_r18_ctw.py"
        ckpt = r"/data/zhangyl/pan_pp.pytorch-master/checkpoints/test1.1/checkpoint.pth.tar"


        # docker
        config = r"config/pan_pp/pan_pp_r18_ctw.py"
        ckpt = r"../checkpoint/checkpoint.pth.tar"
        service = Service(config=config, checkpoint_path=ckpt)
    main()

