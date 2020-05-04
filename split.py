# -*- coding: utf-8 -*-
import os
import random
import shutil


prefix = '/mnt/cephfs_wj/vc/lixiangyu.leexy/datasets/'
origin = os.path.join(prefix, 'Animals_with_Attributes2', 'JPEGImages')

# 获取所有的类名称
categories = os.listdir(origin)
for cat in categories:
    # 获取每个类目录下的文件名
    names = os.listdir(os.path.join(origin, cat))
    # 转换为绝对路径
    names = [os.path.join(os.path.join(origin, cat), each) for each in names]
    val = random.sample(names, len(names) // 11)
    train = list(set(names) - set(val))
    train_path = os.path.join(prefix, 'animals', 'train', cat)
    val_path = os.path.join(prefix, 'animals', 'val', cat)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    for t in train:
        shutil.copy(t, train_path)
    for v in val:
        shutil.copy(v, val_path)