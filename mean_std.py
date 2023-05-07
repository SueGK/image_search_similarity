#!/usr/bin/env python3
# -*- coding: utf-8 -*
#File :split dataset.py
# Date ：
# Last Modified Date: 
# Last Modified By : 

import os
import glob
import random
import shutil
import numpy as np
from PIL import Image

"""统计数据库中所有图片的每个通道的均值和标准差"""

if __name__ == '__main__':
    root_dir = './dataset/'
    train_dir = root_dir + 'train/'
    train_dataset = glob.glob(os.path.join(train_dir, "*.jpg"))

    print(f"Totally {len(train_dataset)} files for training")

    results = []

    for img in train_dataset:
        img = Image.open(img).convert('RGB')
        img = np.array(img).astype(np.unit8)
        img = img/225.
        results.append(img)
    print(f"result shape: {np.array(results).shape}") #[BS, H, W, C]
    mean = np.mean(results, axis=(0,1,2))
    std = np.std(results, axis=(0,1,2))
    print(f"mean: {mean}, std: {std}")