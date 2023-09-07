"""
purpose: 试试非搜题的效果
"""
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
import json


ctw_root_dir1 = './data/CTW1500/'
ctw_test_data_dir = ctw_root_dir1 + 'test/text_image_cukuang/'
ctw_test_gt_dir = ctw_root_dir1 + 'test/text_label_circum/'


ctw_root_dir = '/data1/nas/textline/'
# OCR 4.0 前摄搜题1W 张数据  原图裁剪
ctw_train_data_dir1 = ctw_root_dir + '4.0/'
ctw_train_gt_dir1 = ctw_root_dir + '4.0/'

# OCR 4.0 前摄搜题1W 张数据 减去 挑选出的 2000张后剩余的部分  原图裁剪
ctw_train_data_dir1_1 = ctw_root_dir + '4.0_subset_8000/'
ctw_train_gt_dir1_1 = ctw_root_dir + '4.0_subset_8000/'

# OCR 5.0 前摄搜题0.3W 张数据  矫正图
ctw_train_data_dir2 = ctw_root_dir + '5.0_3000/'
ctw_train_gt_dir2 = ctw_root_dir + '5.0_3000/'

# OCR 5.0 前摄搜题0.3W 张数据  矫正图，经过调整
ctw_train_data_dir2_1 = ctw_root_dir + '5.0_adjusted/'
ctw_train_gt_dir2_1 = ctw_root_dir + '5.0_adjusted/'


# A1007 标注的 0.38W 张
ctw_train_data_dir3 = ctw_root_dir + 'A1007_3780/'
ctw_train_gt_dir3 = ctw_root_dir + 'A1007_3780/'

# A1007 自研 adjusted 0.38W 张
ctw_train_data_dir3_1 = ctw_root_dir + 'A1007adjusted/'
ctw_train_gt_dir3_1 = ctw_root_dir + 'A1007adjusted/'

# 4.0 手写文本行 标注的 0.11W 张
ctw_train_data_dir4 = ctw_root_dir + 'hand_label/'
ctw_train_gt_dir4 = ctw_root_dir + 'hand_label/'

# 4.0 随机挑选的 0.17w 张调整标注的图片
ctw_train_data_dir5 = ctw_root_dir + '4.0_subset/'
ctw_train_gt_dir5 = ctw_root_dir + '4.0_subset/'

# 200 张彩底字词
ctw_train_data_dir6 = ctw_root_dir + 'color_word/'
ctw_train_gt_dir6 = ctw_root_dir + 'color_word/'


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


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


def random_scale(img, short_size=640):
    # fixme: 这里的逻辑可能和训练图片不完全一致, 4.0和A1007的训练图片基本都是水平矩形，高远小于640
    h, w = img.shape[0:2]
    # todo: very different scale valued chosen below comparing with PSENET random sacle in dataloader
    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)  # fixme: plus 0.5
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


#解析labelme接口
def load_labelme_line_result(img, label_path):
    h, w = img.shape[0:2]
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

            bboxes.append((np.asarray(points) / [w, h]).reshape(-1))
            words.append(shape["label"])
    return bboxes, words


class PAN_CUSTOM_V1(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            data_dirs = [
                # ctw_train_data_dir1,
                # ctw_train_data_dir1_1,
                # ctw_train_data_dir2,
                # ctw_train_data_dir3,
                ctw_train_data_dir3_1,
                ctw_train_data_dir4,
                ctw_train_data_dir5,
                ctw_train_data_dir6
            ]
            gt_dirs = [
                # ctw_train_data_dir1,
                # ctw_train_gt_dir1_1,
                # ctw_train_data_dir2,
                # ctw_train_data_dir3,
                ctw_train_data_dir3_1,
                ctw_train_data_dir4,
                ctw_train_data_dir5,
                ctw_train_data_dir6
            ]
        elif split == 'test':
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
            img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.json'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.max_word_num = 200

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path, self.read_type)
        # bboxes, words = get_ann(img, gt_path)
        bboxes, words = load_labelme_line_result(img, gt_path)


        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')  # pixel value 1
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')
                # bboxes[i] = np.reshape(bboxes[i] * [img.shape[1], img.shape[0]],
                #                        (bboxes[i].shape[0], 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '####':  # public datasets have some dense or special text labeled "###" which will ignored during training
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)  # set pexel value of ignored area to 0

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            # imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,  # 训练 CTW 数据时并没有用到，只是顺带传了过去，可能其他非弯曲数据用到了吧
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2])
        ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        data = dict(
            imgs=img,
            img_metas=img_meta
        )
        # print(data)

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            try:
                return self.prepare_train_data(index)
            except Exception as e:
                print(str(e))
                print(self.img_paths[index])
        elif self.split == 'test':
            return self.prepare_test_data(index)


if __name__ == "__main__":
    ...
    t = dict(
        split='test',
        is_transform=False,
        img_size=640,
        short_size=640,
        kernel_scale=0.7,
        read_type='cv2'
    )
    data_loader = PAN_CUSTOM_V1(**t)
    t_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        pin_memory=True
    )
    for idx, data in enumerate(t_loader):
        print(idx)
        print(data)
