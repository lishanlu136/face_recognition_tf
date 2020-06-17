#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-4-30 上午10:34
@Author     : lishanlu
@File       : dataset.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import random
import cv2
import os
from scipy import misc
from config import cfg


class ImageClass(object):
    """
    Stores the paths to images for a given class.
    """
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + 'images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(datadir):
    dataset = []
    path_exp = os.path.expanduser(datadir)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]

    return image_paths


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)

    return image_paths_flat, labels_flat


def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)

    return image_paths_shuff, labels_shuff


class Dataset(object):

    def __init__(self, datatype='train'):
        self.data_dir = cfg.TRAIN.IMAGE_DIR if datatype == 'train' else cfg.TEST.IMAGE_DIR
        self.batch_size = cfg.TRAIN.BATCH_SIZE if datatype == 'train' else cfg.TEST.BATCH_SIZE
        self.target_size = cfg.TRAIN.TARGET_SIZE if datatype == 'train' else cfg.TEST.TARGET_SIZE
        self.shuffle = cfg.TRAIN.SHUFFLE if datatype == 'train' else cfg.TEST.SHUFFLE
        self.data_aug = cfg.TRAIN.DATA_AUG if datatype == 'train' else cfg.TEST.DATA_AUG
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS if datatype == 'train' else cfg.TEST.MAX_EPOCHS
        self.image_list, self.label_list = self.read_data_to_list()
        self.data_type =datatype

    def __str__(self):
        return 'Class: ' + str(len(self.data_set)) + ', Images: ' + str(len(self.image_list))

    def __len__(self):
        return len(self.data_set)

    def read_data_to_list(self):
        data_ary = self.data_dir.split(",")
        self.data_set = []
        for data_dir in data_ary:
            increased_set = get_dataset(data_dir)
            self.data_set += increased_set
        image_paths_list, labels_list = get_image_paths_and_labels(self.data_set)
        if self.shuffle:
            image_paths_list, labels_list = shuffle_examples(image_paths_list, labels_list)
        return image_paths_list, labels_list

    @staticmethod
    def corrupt_brightness(image, label):
        """
        Radnomly applies a random brightness change.
        """
        if random.uniform(0,1) > 0.5:
            image = tf.image.random_hue(image, 0.3)
        return image, label

    @staticmethod
    def corrupt_contrast(image, label):
        """
        Randomly applies a random contrast change.
        """
        if random.uniform(0,1) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    @staticmethod
    def corrupt_saturation(image, label):
        """
        Randomly applies a random saturation change.
        """
        if random.uniform(0,1) > 0.5:
            image = tf.image.random_saturation(image, 0.3, 0.5)
        return image, label

    @staticmethod
    def random_flip_left_right(image, label):
        """
        Randomly flips image left or right in accord.
        """
        if random.uniform(0,1) > 0.5:
            image = tf.image.random_flip_left_right(image)
        return image, label

    @staticmethod
    def random_rotate(image, label):
        if random.uniform(0,1) > 0.5:
            angle = random.uniform(-20.0, 20.0)
            image = tf.py_func(misc.imrotate, [image, angle], tf.uint8)
        return image, label

    def random_crop(self, image, label):
        if random.uniform(0,1) > 0.5:
            image = tf.random_crop(image, [self.target_size[0], self.target_size[1], 3])
        return image, label

    @staticmethod
    def image_standardization(image, label):
        image = tf.image.per_image_standardization(image)
        return image, label

    def resize_image(self, image, label):
        #image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_images(image, [self.target_size[0], self.target_size[1]])
        #image = tf.squeeze(image, axis=0)
        return image, label

    # for 112x112
    @staticmethod
    def read_image_from_cv2(filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def cv2_resize_img(self, image):
        if self.data_type == 'train':
            image = cv2.resize(image, (128, 128))
        elif self.data_type == 'val':
            image = cv2.resize(image, (self.target_size[0], self.target_size[1]))
        else:
            raise print('Data type error.')
        return image

    def parse_data(self, image_path, label):
        """
        Reads image and label depending on
        specified exxtension.
        """
        image_content = tf.read_file(image_path)
        """
        image = tf.cond(tf.image.is_jpeg(image_content),
                        lambda: tf.image.decode_jpeg(image_content, channels=self.channels),
                        lambda: tf.image.decode_image(image_content, channels=self.channels))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        """
        image = tf.image.decode_image(image_content, channels=3)
        image.set_shape([None,None,3])
        if self.target_size == [112, 112]:
            if self.data_type == 'train':
                image = tf.image.resize_images(image, (128, 128))
            elif self.data_type == 'val':
                image = tf.image.resize_images(image, (self.target_size[0], self.target_size[1]))
            else:
                raise print("Data type error.")
        return image, label

    def __call__(self, num_threads=4, buffer=32):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            num_threads: Number of parallel subprocesses to load data.
            buffer: Number of images to prefetch in buffer.
        Returns:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding lable batch.
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches.
        """

        # Convert lists of paths to tensors for tensorflow
        images_name_tensor = tf.constant(self.image_list)
        label_tensor = tf.constant(self.label_list)

        # one-hot code if use tf.nn.softmax_cross_entropy_with_logits()
        #label_tensor = tf.one_hot(self.label_list, len(self.data_set), axis=-1)

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices((images_name_tensor, label_tensor))
        # Refill data indefinitely.
        data = data.repeat()
        if self.shuffle:
            data = data.shuffle(buffer)
        # Parse images and labels
        data = data.map(self.parse_data, num_parallel_calls=num_threads)

        # If augmentation is to be applied
        if self.data_aug:
            data = data.map(self.corrupt_brightness, num_parallel_calls=num_threads)

            data = data.map(self.corrupt_contrast, num_parallel_calls=num_threads)

            data = data.map(self.corrupt_saturation, num_parallel_calls=num_threads)

            data = data.map(self.random_rotate, num_parallel_calls=num_threads)

            data = data.map(self.random_crop, num_parallel_calls=num_threads)

            data = data.map(self.random_flip_left_right, num_parallel_calls=num_threads)

        data = data.map(self.resize_image, num_parallel_calls=num_threads)
        data = data.map(self.image_standardization, num_parallel_calls=num_threads)
        # Batch the data
        data = data.batch(self.batch_size)
        # Prefetch batch
        data = data.prefetch(buffer_size=1)
        # Create iterator
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        # Next element Op
        next_element = iterator.get_next()
        # Data set init. op
        init_op = iterator.make_initializer(data)

        return next_element, init_op


if __name__ == '__main__':
    """"
    d = Dataset('train')
    train_data = d(num_threads=4, buffer=32)
    for i, data in enumerate(train_data):
        print(data)
        images = tf.cast(data[0], dtype=tf.uint8)
        image = images[0].numpy()
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, str(data[1][0].numpy()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(1000)
    """
    d = tf.data.Dataset.from_tensor_slices(tf.range(10))
    d = d.repeat(2)
    #d = d.shuffle(5)
    d = d.batch(2).prefetch(1)

    iterator = tf.data.Iterator.from_structure(d.output_types, d.output_shapes)
    next_d = iterator.get_next()
    init_op = iterator.make_initializer(d)
    sess = tf.Session()
    for _ in range(2):
        sess.run(init_op)
        while True:
            try:
                print(sess.run(next_d))
            except tf.errors.OutOfRangeError:
                break
