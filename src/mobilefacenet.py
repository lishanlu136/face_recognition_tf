#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-5-13 下午3:26
@Author     : lishanlu
@File       : mobilefacenet.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.contrib import slim
import os
print(tf.__version__)


def conv_layer(x, output_filters, kernal_size, downsample=False, activate=True, bn=True, name='conv'):
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)
    with tf.name_scope(name=name):
        conv = slim.conv2d(inputs=x, num_outputs=output_filters, kernel_size=kernal_size,
                           stride=strides, padding='same', activation_fn=None, normalizer_fn=None)
        if bn:
            conv = slim.batch_norm(inputs=conv)
        if activate:
            conv = tf.nn.relu(conv)
        return conv


def inverted_block(x, input_filters, output_filters, expand_ratio, stride, name='inverted_block'):
    with tf.name_scope(name=name):
        res_block = conv_layer(x, input_filters*expand_ratio, 1, downsample=False,
                               activate=True, bn=False, name=name+'_conv1x1_expand')
        # depthwise conv2d
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[3, 3],
                                          stride=stride, activation_fn=tf.nn.relu,
                                          depth_multiplier=1.0, normalizer_fn=slim.batch_norm,
                                          scope=name+'_separable_conv')
        if stride == 2:
            return res_block
        else:
            if input_filters != output_filters:
                x = conv_layer(x, output_filters, 1, downsample=False, activate=False, bn=False, name=name+'_conv1x1_res')
            return tf.add(res_block, x)


def mobilefacenet_inverted_block(net, stride, depth, expand_ratio, repeate, name='InvertedBlock'):
    input_filters = net.get_shape().as_list()[3]
    with tf.name_scope(name=name):
        net = inverted_block(net, input_filters, depth, expand_ratio=expand_ratio, stride=stride, name=name+'_1')
        for i in range(1, repeate):
            net = inverted_block(net, input_filters, depth, expand_ratio=expand_ratio, stride=1, name=name+'_%d'%(i+1))
        return net


def mobilefacenet(x, bottleneck_layer_size=128, weight_decay=0.0005, name='MobileFaceNet'):
    with tf.name_scope(name=name):
        net = conv_layer(x, 64, 3, downsample=True, activate=True, bn=True, name='conv1')
        net = slim.separable_conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=1, padding='same', depth_multiplier=1.0,
                                    normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu, scope='separable_conv2')
        # inverted block
        net = mobilefacenet_inverted_block(net, stride=2, depth=64, expand_ratio=2, repeate=5, name='InvertedBlock1')
        net = mobilefacenet_inverted_block(net, 2, 128, 4, 1, name='InvertedBlock2')
        net = mobilefacenet_inverted_block(net, 1, 128, 2, 6, name='InvertedBlock3')
        net = mobilefacenet_inverted_block(net, 2, 128, 4, 1, name='InvertedBlock4')
        net = mobilefacenet_inverted_block(net, 1, 128, 2, 2, name='InvertedBlock5')
        net = conv_layer(net, 1, 512, downsample=False, activate=True, bn=True, name='conv18')
        # Global depthwise conv2d
        kernel_size = [net.get_shape().as_list()[1], net.get_shape().as_list()[2]]
        net = slim.separable_conv2d(inputs=net, num_outputs=512, kernel_size=kernel_size, stride=1,
                                    depth_multiplier=1.0, activation_fn=None, padding='VALID', scope='global_separable_conv')
        logits = slim.conv2d(net, bottleneck_layer_size, kernel_size=[1, 1], stride=1, activation_fn=None,
                             weights_regularizer=tf.contrib.layers.l2_regularizer(10 * weight_decay), scope='LinearConv1x1')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        return logits


def prelu(input_x, name=''):
    alphas = tf.get_variable(name=name + 'prelu_alphas',
                             initializer=tf.constant(0.25, dtype=tf.float32, shape=[input_x.get_shape()[-1]]))
    pos = tf.nn.relu(input_x)
    neg = alphas * (input_x - abs(input_x)) * 0.5
    return pos + neg


def mobilefacenet_arg_scope(is_training=True, weight_decay=0.0, regularize_separable=False):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'fused': True,
        'decay': 0.995,
        'epsilon': 2e-5,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    # Set weight_decay for weights in Conv and InvResBlock layers.
    weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_separable:
        separable_regularizer = regularizer
    else:
        separable_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=prelu, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=separable_regularizer) as sc:
                    return sc


def inference(images, bottleneck_layer_size=128, phase_train=True, weight_decay=0.0):
    arg_scope = mobilefacenet_arg_scope(is_training=phase_train, weight_decay=weight_decay, regularize_separable=False)
    with slim.arg_scope(arg_scope):
        return mobilefacenet(x=images, bottleneck_layer_size=bottleneck_layer_size, weight_decay=weight_decay)


if __name__ == '__main__':
    import numpy as np
    data_x = np.random.rand(4,112,112,3)
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, 112,112,3], name='input')
    logits = inference(input_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _logits = sess.run(logits, feed_dict={input_data: data_x})
        print(_logits)






