"""Lazy-loading dataset for NRKNet (data_v3 style).

Input and ground-truth are separate folders. All images in each folder are
loaded, each side sorted by filename, then paired in order.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils import data


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _list_folder_pair(img_path: str, gt_path: str):
    """Return (img_list, gt_list) of sorted image paths, paired by order."""
    img_list = sorted(p for p in Path(img_path).iterdir()
                      if p.suffix.lower() in IMAGE_EXTS)
    gt_list  = sorted(p for p in Path(gt_path).iterdir()
                      if p.suffix.lower() in IMAGE_EXTS)
    assert len(img_list) == len(gt_list), (
        f"img/gt count mismatch: {len(img_list)} vs {len(gt_list)}"
    )
    return img_list, gt_list


class Dataset(data.Dataset):
    def __init__(self, img_path: str, gt_path: str, crop_size=(256, 256)):
        super().__init__()
        self.crop_size = crop_size
        self.img_list, self.gt_list = _list_folder_pair(img_path, gt_path)

    def __len__(self):
        return len(self.img_list)

    def _augment(self, img, crop_left, crop_top, hf, vf, rot):
        img = img[:, crop_top:crop_top + self.crop_size[1], crop_left:crop_left + self.crop_size[0]]
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        return torch.rot90(img, rot, [1, 2])

    def __getitem__(self, idx):
        clear_img  = torch.from_numpy(np.array(cv2.imread(str(self.gt_list[idx]))).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(str(self.img_list[idx]))).transpose((2, 0, 1)))
        _, h, w = clear_img.shape
        crop_left = int(np.floor(np.random.uniform(0, w - self.crop_size[0] + 1)))
        crop_top  = int(np.floor(np.random.uniform(0, h - self.crop_size[1] + 1)))
        hf  = np.random.randint(0, 2)
        vf  = np.random.randint(0, 2)
        rot = np.random.randint(0, 4)
        blurry_img = self._augment(blurry_img, crop_left, crop_top, hf, vf, rot) / 255.0
        clear_img  = self._augment(clear_img,  crop_left, crop_top, hf, vf, rot) / 255.0
        return {'img256': blurry_img, 'label256': clear_img}


class TestDataset(data.Dataset):
    def __init__(self, img_path: str, gt_path: str):
        super().__init__()
        self.img_list, self.gt_list = _list_folder_pair(img_path, gt_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Return full-size images; the test script handles /multiple padding+crop
        # so the saved output matches the original input size.
        clear_img  = torch.from_numpy(np.array(cv2.imread(str(self.gt_list[idx]))).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(str(self.img_list[idx]))).transpose((2, 0, 1)))
        return {'img256': blurry_img / 255.0, 'label256': clear_img / 255.0}
