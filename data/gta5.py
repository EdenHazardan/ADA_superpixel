import os
import sys
import cv2
import shutil
import math
import json
import random
import numpy as np
import os.path as osp
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttransforms

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19
class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class gta5_dataset(Dataset):
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/ActiveDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.crop_size = (512, 1024)

        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def random_crop(self, img, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        img = img[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
        gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
        return img, gt

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop(img, gt)

        return img, gt
