#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-4-15 下午5:32
@Author     : lishanlu
@File       : config.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
from easydict import EasyDict as edict

__C = edict()
cfg = __C


# Train
__C.TRAIN = edict()
__C.TRAIN.GPU_IDX = '0'
#__C.TRAIN.IMAGE_DIR = '/home/lishanlu/data/iQIYI_data/personai_icartoonface_rectrain/det_face/'
__C.TRAIN.IMAGE_DIR = '/data1/lishanlu/dataset/train/merged_train_data_182/,/data1/lishanlu/dataset/train/high_light_mt_182,/data1/lishanlu/dataset/train/glasses_data_mt_182_train,/data1/lishanlu/dataset/train/old_data_mt_182_train,/data1/lishanlu/dataset/train/face_error_type1_182_train,/disk1/lishanlu/dataset/train/longhu_train_data_182,/disk3/lishanlu/dataset/train/campus_data_child_182,/disk3/lishanlu/dataset/train/child_train_data_182,/disk3/lishanlu/dataset/train/glasses_train_data_182'
__C.TRAIN.ANNOTATION ='/data1/wanjinchang/attirbutes_dataset/Person_attributes_bak.txt'
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.TARGET_SIZE = (112, 112)
__C.TRAIN.SHUFFLE = True
__C.TRAIN.DATA_AUG = True
__C.TRAIN.EPOCH_SIZE = 2000
__C.TRAIN.MAX_EPOCHS = 30
__C.TRAIN.MOVING_AVG_DECAY = 0.9995
__C.TRAIN.WEIGHT_DECAY = 0.00005
__C.TRAIN.SAVE_DIR = './train_results/'
__C.TRAIN.PRETRAINED_MODEL = ''

# Test
__C.TEST = edict()
__C.TEST.IMAGE_DIR = '/home/lishanlu/data/attributes_dataset/Images/'
__C.TEST.ANNOTATION ='/home/lishanlu/data/attributes_dataset/Person_attributes_bak.txt'
__C.TEST.BATCH_SIZE = 4
__C.TEST.TARGET_SIZE = (160, 160)
__C.TEST.SHUFFLE = False
__C.TEST.DATA_AUG = False
__C.TEST.MAX_EPOCHS = 1

__C.MODEL = edict()
__C.MODEL.EBEDDING_SIZE = 256