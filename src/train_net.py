#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-4-30 上午11:42
@Author     : lishanlu
@File       : train_net.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
from config import cfg
from dataset import Dataset
import mobilefacenet
from face_losses import arcface_loss
import time
import os
import numpy as np


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU_IDX
    # Dataset
    dataset = Dataset('train')
    print(dataset)
    nrof_classes = len(dataset)
    sub_dir = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    train_logdir = os.path.join(cfg.TRAIN.SAVE_DIR, sub_dir, 'log')
    train_modeldir = os.path.join(cfg.TRAIN.SAVE_DIR, sub_dir, 'model')
    if not os.path.exists(train_logdir):
        os.makedirs(train_logdir)
    if not os.path.exists(train_modeldir):
        os.makedirs(train_modeldir)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        next_element, init_op = dataset(num_threads=4, buffer=1000)
        # Network
        input_data = tf.placeholder(dtype=tf.float32, shape=[None,cfg.TRAIN.TARGET_SIZE[0],cfg.TRAIN.TARGET_SIZE[1],3], name='input')
        label = tf.placeholder(dtype=tf.int32, shape=[None,], name='label')
        phase_train = tf.placeholder(dtype=tf.bool)
        prelogits = mobilefacenet.inference(input_data, bottleneck_layer_size=cfg.MODEL.EBEDDING_SIZE,
                                            phase_train=phase_train, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        _, cross_entropy_loss, regular_loss, total_loss = compute_loss(prelogits, label, nrof_classes)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 50*cfg.TRAIN.EPOCH_SIZE, 0.1, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cross_entropy', cross_entropy_loss)
        tf.summary.scalar('total_loss', total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=total_loss, global_step=global_step)
        if cfg.TRAIN.PRETRAINED_MODEL:
            loader_saver = tf.train.Saver(tf.trainable_variables())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        summary_op = tf.summary.merge_all()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        with sess.as_default():
            if cfg.TRAIN.PRETRAINED_MODEL:
                all_vars = tf.trainable_variables()
                for var in all_vars:
                    print(var.name)
                print("Restoring pretrained model.")
                loader_saver.restore(sess, cfg.TRAIN.PRETRAINED_MODEL)
            print("Start training")
            epoch = 0
            while epoch < cfg.TRAIN.MAX_EPOCHS:
                step = sess.run(global_step)
                epoch = step // cfg.TRAIN.EPOCH_SIZE
                data = sess.run(next_element)
                feed_dict = { input_data: data[0],
                              label: data[1],
                              phase_train: True}
                if step%cfg.TRAIN.EPOCH_SIZE==100:
                    _, c_loss, reg_loss, loss, summary_str = sess.run([train_op, cross_entropy_loss,
                                                                       regular_loss, total_loss, summary_op],
                                                                      feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step=step)
                else:
                    _, c_loss, reg_loss, loss = sess.run([train_op, cross_entropy_loss,
                                                          regular_loss, total_loss],
                                                         feed_dict=feed_dict)
                print("[%d/%d]   cross_entropy_loss: %4.2f  regular_loss: %4.2f    total_loss: %4.2f"
                      %(epoch, step%cfg.TRAIN.EPOCH_SIZE, c_loss, np.sum(reg_loss), loss))

                if step > 0 and step%cfg.TRAIN.EPOCH_SIZE==0:
                    print("Saving model.")
                    saver.save(sess, os.path.join(train_modeldir, 'model.ckpt'), global_step=step, write_meta_graph=False)
                    metagraph_filename = os.path.join(train_modeldir, 'model.meta')
                    if not os.path.exists(metagraph_filename):
                        saver.export_meta_graph(metagraph_filename)


def compute_loss(prelogits, label_batch, nrof_classes):
    eps = 1e-5
    prelogits_norm = tf.reduce_mean(tf.norm(prelogits + eps, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 2e-4)

    # Add arcface loss
    logit = arcface_loss(embedding=prelogits, labels=label_batch,
                         w_init=tf.random_normal_initializer(stddev=0.02), out_num=nrof_classes)
    # Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logit)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # get the regular loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Calculate the total losses
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    return logit, cross_entropy_mean, regularization_losses, total_loss


def _add_loss_summaries(total_loss, summaries):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        summaries.append(tf.summary.scalar(l.op.name + ' (raw)', l))
        summaries.append(tf.summary.scalar(l.op.name, loss_averages.average(l)))

    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, summaries,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss, summaries)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op] + update_ops):
        train_op = tf.no_op(name='train')

    return train_op


def main_v2():
    from tools.dataset_processing import parse_function
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sub_dir = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        train_logdir = os.path.join(cfg.TRAIN.SAVE_DIR, sub_dir, 'log')
        train_modeldir = os.path.join(cfg.TRAIN.SAVE_DIR, sub_dir, 'model')
        if not os.path.exists(train_logdir):
            os.makedirs(train_logdir)
        if not os.path.exists(train_modeldir):
            os.makedirs(train_modeldir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, cfg.TRAIN.TARGET_SIZE[0],cfg.TRAIN.TARGET_SIZE[1], 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        # random flip left right
        tfrecords_f = os.path.join("/disk2/lishanlu/dataset/tfrecord_ms1m_112x112", 'tran.tfrecords')
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        """
        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)
        """
        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits = mobilefacenet.inference(inputs, bottleneck_layer_size=cfg.MODEL.EBEDDING_SIZE,
                                            phase_train=phase_train_placeholder, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        logits, cross_entropy_loss, regular_loss, total_loss = compute_loss(prelogits, labels, 85164)  #MS1M-V1: 85164, MS1M-V2: 85742'

        # define the learning rate schedule
        #learning_rate = tf.train.piecewise_constant(global_step, boundaries=[2000, 10000, 20000, 40000],
        #                                            values=[0.1, 0.01, 0.001, 0.0001, 0.00001], name='lr_schedule')
        learning_rate = tf.train.exponential_decay(0.1, global_step,
                                                   5000, 0.5, staircase=True, name='lr_schedule')

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(train_logdir, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', cross_entropy_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, 'ADAM', learning_rate, 0.999,
                         tf.global_variables(), summaries, False)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if cfg.TRAIN.PRETRAINED_MODEL:
            print('Restoring pretrained model: %s' % cfg.TRAIN.PRETRAINED_MODEL)
            ckpt = tf.train.get_checkpoint_state(cfg.TRAIN.PRETRAINED_MODEL)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        count = 0
        total_accuracy = {}
        for i in range(cfg.TRAIN.MAX_EPOCHS):
            sess.run(iterator.initializer)
            _ = sess.run(inc_epoch_op)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)

                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    start = time.time()
                    _, total_loss_val, inference_loss_val, reg_loss_val, step, acc_val = \
                        sess.run([train_op, total_loss, cross_entropy_loss, regular_loss, global_step, Accuracy_Op],
                                 feed_dict=feed_dict)
                    end = time.time()
                    pre_sec = cfg.TRAIN.BATCH_SIZE / (end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % 10 == 0:
                        print(
                            'epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                            (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if count > 0 and count % 100 == 0:
                        feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % 1000 == 0:
                        filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(train_modeldir, filename)
                        saver.save(sess, filename)
                    """
                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')
                        for db_index in range(len(ver_list)):
                            start_time = time.time()
                            data_sets, issame_list = ver_list[db_index]
                            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            nrof_batches = data_sets.shape[0] // args.test_batch_size
                            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                                start_index = index * args.test_batch_size
                                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                feed_dict = {inputs: data_sets[start_index:end_index, ...],
                                             phase_train_placeholder: False}
                                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list,
                                                                             nrof_folds=args.eval_nrof_folds)
                            duration = time.time() - start_time

                            print("total time %.3fs to evaluate %d images of %s" % (
                            duration, data_sets.shape[0], ver_name_list[db_index]))
                            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                            auc = metrics.auc(fpr, tpr)
                            print('Area Under Curve (AUC): %1.3f' % auc)
                            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                            print('Equal Error Rate (EER): %1.3f\n' % eer)

                            with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])),
                                      'at') as f:
                                f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                            if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                                print('best accuracy is %.5f' % np.mean(accuracy))
                                filename = 'MobileFaceNet_iter_best_{:d}'.format(count) + '.ckpt'
                                filename = os.path.join(args.ckpt_best_path, filename)
                                saver.save(sess, filename)
                    """

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break


if __name__ == '__main__':
    main_v2()