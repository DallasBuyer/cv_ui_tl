# -*- coding:utf-8 -*-
import tensorlayer as tl
import tensorflow as tf
import time
import numpy as np
from classes_name import *


def get_cost_acc(network, labels):
    y = network.outputs
    cost = tl.cost.cross_entropy(y, labels, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cost, acc

def train_params(n_epoch, batch_size, learning_rate):
    n_epoch_ = n_epoch
    batch_size_ = batch_size
    learning_rate_ = learning_rate
    return n_epoch_, batch_size_, learning_rate_

def select_optimization(opt_index, learning_rate):
    opt_list = [tf.train.GradientDescentOptimizer(learning_rate),
                tf.train.MomentumOptimizer(learning_rate, 0.9),
                tf.train.AdagradOptimizer(learning_rate),
                tf.train.AdadeltaOptimizer(learning_rate),
                tf.train.RMSPropOptimizer(learning_rate),
                tf.train.AdamOptimizer(learning_rate),
                tf.train.ProximalAdagradOptimizer(learning_rate),
                tf.train.ProximalGradientDescentOptimizer(learning_rate)]
    train_op = opt_list[opt_index]
    return train_op

def load_name(model_name):
    size_dict = {"MNIST": mnist_names,
                 "ImageNet": imagenet_names}
    names = size_dict[str(model_name)]
    return names

def model_size(model_name):
    size_dict = {"CNN_28_28_1": (28, 28, 1),
                 "CNN_32_32_3": (32, 32, 1),
                 "CNN_32_32_1": (32, 32, 1),
                 "AlexNet_28_28_1": (224, 224, 1),
                 "AlexNet_32_32_3": (32, 32, 3),
                 "VGG16_224_224_3": (224, 224, 3),
                 "Inception_v1_224_224_1": (224, 224, 1),
                 "Inception_v1_224_224_3": (224, 224, 3),
                 "ResNet": (224, 224, 3)}
    size = size_dict[str(model_name)]
    return size

def train_model(network, X_train, y_train, X_test, y_test, X_holder, y_holder,
                batch_size, n_epoch, print_freq, optimizer, sess):
    # get cost and acc
    print y_holder.dtype, y_holder.shape
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_holder, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_holder)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_op = optimizer.minimize(cost)
    # get batch tensors
    x_train_batch, y_train_batch = tf.train.shuffle_batch([X_train, y_train],
                                                          batch_size=batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=1000,
                                                          num_threads=32)
    x_test_batch, y_test_batch = tf.train.batch([X_test, y_test],
                                                batch_size=batch_size,
                                                capacity=50000,
                                                num_threads=32)
    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()
    # test cod*********************
    # print type(y_test)
    # l = sess.run(X_train)
    # print l, type(l), l.shape, l.size
    #**************************

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    n_step_epoch = int(50000 / batch_size)
    n_step = n_epoch * n_step_epoch
    step = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):
            val, l = sess.run([x_train_batch, y_train_batch])
            #print "@@"*10, s, "@@"*10, l
            #print type(val), val.dtype, type(l), l.dtype
            #l = np.array(l, dtype=np.float32)
            #print l.dtype
            err, ac, _ = sess.run([cost, acc, train_op], feed_dict={X_holder: val, y_holder: l})
            # err, ac, _ = sess.run([cost, acc, train_op])  # another way which don't use placeholder
            step += 1
            train_loss += err
            train_acc += ac
            n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print ("Epoch %d: Step %d-%d of %d took %fs" %
                   (epoch, step, step + n_epoch, n_step, time.time() - start_time))
            print ("train loss: %f" % (train_loss / n_batch))
            print ("train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(10000 / batch_size)):
                val_test, l_test = sess.run([x_test_batch, y_test_batch])
                err, ac = sess.run([cost, acc], feed_dict={X_holder: val_test, y_holder: l_test})
                # err, ac = sess.run([cost_test, acc_test])   # another way which don't use placeholder
                test_loss += err
                test_acc += ac
                n_batch += 1
            print ("test loss: %f" % (test_loss / n_batch))
            print ("test acc: %f" % (test_acc / n_batch))

        # save the network
        # if (epoch + 1) % (print_freq * 10) == 0:
        #     print("Save model" + "!" * 10)
        #     saver = tf.train.Saver()
        #     save_path = saver.save(sess, model_file_name)

    coord.request_stop()
    coord.join(threads)
    sess.close()