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
from PIL import Image

# 对所有图片进行RGB转化，并且统一调整到一致大小，但不让圈片发生变形或扭曲，划分了训练集和测试集

if __name__ == '__main__':
    test_split_ratio = 0.05
    img_size = 128
    root_dir = './dataset/'
    data_dir = root_dir + 'data/'
    train_dir = root_dir + 'train/'
    test_dir = root_dir + 'test/'

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    dirs = glob.glob(os.path.join(data_dir, "*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"Totally {len(dirs)} classes: {dirs}")

    for path in dirs:
        # 对每个类别单独处理
        path = path.split('/')[-1]
        os.mkdir(os.path.join(train_dir, path), exist_ok=True)
        os.mkdir(os.path.join(test_dir, path), exist_ok=True)

        files = glob.glob(os.path.join(path, "*.jpg"))
        files += glob.glob(os.path.join(path, "*.png"))
        files += glob.glob(os.path.join(path, "*.jpeg"))
        files += glob.glob(os.path.join(path, "*.JPG"))

        random.shuffle(files)
        n = int(len(files) * test_split_ratio)

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            old_size = img.size # old_size[0] is in (width, height） format
            ratio = float(img_size) /max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            # resizes the image object to the new size using the antialias filter, which reduces the distortion caused by resizing
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (img_size, img_size))
            new_im.paste(im, ((img_size-new_size[0])//2,
                              (img_size-new_size[1])//2))
            
            assert new_im.mode == "RGB"
            
            if i <= n:
                new_im.save(os.path.join(f'test_dir/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))
            else :
                new_im.save(os.path.join(f'train_dir/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))
  
    test_files = glob.glob(os.path.join(test_dir, "*", "*.jpg"))
    train_files = glob.glob(os.path.join(train_dir, "*", "*.jpg"))

    print(f"Totally {len(test_files)} test images, {len(train_files)} train images")

