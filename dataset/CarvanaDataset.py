#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:12:03 2019

@author: ubuntu
"""
from torch.utils.data.dataset import Dataset
import os
import cv2
import numpy as np
from PIL import Image
# Create pytorch dataset


def load_image(path: Path):
    img = cv2.imread(str(path))
    img = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_REFLECT_101)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def load_mask(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if '.gif' in str(path):
                img = (np.asarray(img) > 0)
            else:
                img = (np.asarray(img) > 255 * 0.5)
            img = cv2.copyMakeBorder(img.astype(np.uint8), 0, 0, 1, 1, cv2.BORDER_REFLECT_101)
            return img.astype(np.float32)
        

class CarvanaDataset(Dataset):
    def __init__(self, root: Path, to_augment=False):
        # TODO This potentially may lead to bug.
        self.image_paths = sorted(root.joinpath('images').glob('*.jpg'))
        self.mask_paths = sorted(root.joinpath('masks').glob('*'))
        self.to_augment = to_augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])

        if self.to_augment:
            img, mask = augment(img, mask)

        return utils.img_transform(img), torch.from_numpy(np.expand_dims(mask, 0))

if __name__ == '__main__':
    path = './data/carvana'
    dataset = CarvanaDataset(path, )

    def __len__(self):
        return len(self.data_path)