#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-5-20 下午5:46
@Author     : lishanlu
@File       : make_samples.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import os
import cv2
import numpy as np


def main():
    label_file = '/disk3/lishanlu/dataset/iQIYI_data/personai_icartoonface_rectest/icartoonface_rectest_det.txt'
    image_dir = '/disk3/lishanlu/dataset/iQIYI_data/personai_icartoonface_rectest/icartoonface_rectest'
    save_path = '/disk3/lishanlu/dataset/iQIYI_data/icartoonface_rec/test_112x112_new'
    margin = 0
    image_size = 112
    with open(label_file, 'r') as f:
        s = f.readline()
        while s:
            s_list = s.strip('\n').split('\t')
            image_path = os.path.join(image_dir, s_list[0])
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                xmin = np.maximum(int(s_list[1]) - margin//2, 0)
                ymin = np.maximum(int(s_list[2]) - margin//2, 0)
                xmax = np.minimum(int(s_list[3]) + margin//2, w)
                ymax = np.minimum(int(s_list[4]) + margin//2, h)
                face = img[ymin:ymax, xmin:xmax, :]
                scale = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                save_dir = os.path.join(save_path, s_list[0][:s_list[0].rfind('/')])
                save_name = os.path.join(save_path, s_list[0])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_name, scale)
            else:
                pass
            s = f.readline()


if __name__ == '__main__':
    #main()
    import pandas
    data = pandas.read_csv("/home/lishanlu/data/iQIYI_data/personai_icartoonface_dettrain/icartoonface_dettrain.csv")
    print(data.shape)
    srcs = data.get_values()
    d = []
    for i in srcs:
        print(i)
        d.append(i[0]+' '+'%d,%d,%d,%d,0'%(i[1],i[2],i[3],i[4]))

