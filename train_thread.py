# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore
import time
import tensorflow as tf
import tensorlayer as tl
import preprocess_imagess
import inference
import os

class TrainThread(QtCore.QThread):
    finish_signal = QtCore.pyqtSignal(list)
    def __init__(self, current_data, current_model, pre_load, is_save,
                batch_size, n_epoch, print_freq, save_n, optimizer, graph, parent=None):
        super(TrainThread, self).__init__(parent)
        self.graph = graph
        self.current_model = current_model
        self.is_save = is_save
        self.pre_load = pre_load
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.print_freq = print_freq
        self.save_n = save_n

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        with self.graph.as_default():
            self.X_train, self.y_train, self.X_test, self.y_test, size = preprocess_imagess.load_dataset(current_data)
            self.X_holder = tf.placeholder(tf.float32, shape=[None, size[0], size[1], size[2]])
            self.y_holder = tf.placeholder(tf.int64, shape=[None, ])
            self.network = inference.load_model(inputs=self.X_holder, model_name=self.current_model)
            self.optimizer = optimizer

    def run(self):
        with self.graph.as_default():
            print "this is in the train_thread"
            # get cost and acc
            print self.y_holder.dtype, self.y_holder.shape
            y = self.network.outputs
            cost = tl.cost.cross_entropy(y, self.y_holder, 'cost')
            correct_prediction = tf.equal(tf.argmax(y, 1), self.y_holder)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            train_op = self.optimizer.minimize(cost)
            # get batch tensors
            x_train_batch, y_train_batch = tf.train.shuffle_batch([self.X_train, self.y_train],
                                                                  batch_size=self.batch_size,
                                                                  capacity=2000,
                                                                  min_after_dequeue=1000,
                                                                  num_threads=32)
            x_test_batch, y_test_batch = tf.train.batch([self.X_test, self.y_test],
                                                        batch_size=self.batch_size,
                                                        capacity=50000,
                                                        num_threads=32)
            print "x_train_batch: ", x_train_batch, "\ny_train_batch: ", y_train_batch

        with tf.Session(graph=self.graph) as sess:
            tl.layers.initialize_global_variables(sess)
            # restore the pre_trained network
            load_path = str('../zoo/' + self.current_model + '.npz')
            if os.path.isfile(load_path):
                tl.files.load_and_assign_npz(sess, name=load_path, network=self.network)
                print "loaded pre_trained model!!!"
            else:
                print "There does not exist pre_trained model!!!"
            self.network.print_params()
            self.network.print_layers()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            n_step_epoch = int(50000 / self.batch_size)
            n_step = self.n_epoch * n_step_epoch
            step = 0
            for epoch in range(self.n_epoch):
                start_time = time.time()
                train_loss, train_acc, n_batch = 0, 0, 0
                for s in range(n_step_epoch):
                    val, l = sess.run([x_train_batch, y_train_batch])
                    # print "@@"*10, s, "@@"*10, l
                    # print type(val), val.dtype, type(l), l.dtype
                    # l = np.array(l, dtype=np.float32)
                    # print l.dtype
                    err, ac, _ = sess.run([cost, acc, train_op], feed_dict={self.X_holder: val, self.y_holder: l})
                    # err, ac, _ = sess.run([cost, acc, train_op])  # another way which don't use placeholder
                    step += 1
                    train_loss += err
                    train_acc += ac
                    n_batch += 1

                if epoch + 1 == 1 or epoch == self.n_epoch-1 or (epoch + 1) % self.print_freq == 0:
                    print ("Epoch %d: Step %d-%d of %d took %fs" %
                           (epoch, step, step + self.n_epoch, n_step, time.time() - start_time))
                    print ("train loss: %f" % (train_loss / n_batch))
                    print ("train acc: %f" % (train_acc / n_batch))
                    self.train_loss.append(train_loss/n_batch)
                    self.train_acc.append(train_acc/n_batch)

                    test_loss, test_acc, n_batch = 0, 0, 0
                    for _ in range(int(10000 / self.batch_size)):
                        val_test, l_test = sess.run([x_test_batch, y_test_batch])
                        err, ac = sess.run([cost, acc], feed_dict={self.X_holder: val_test, self.y_holder: l_test})
                        # err, ac = sess.run([cost_test, acc_test])   # another way which don't use placeholder
                        test_loss += err
                        test_acc += ac
                        n_batch += 1
                    print ("test loss: %f" % (test_loss / n_batch))
                    print ("test acc: %f" % (test_acc / n_batch))
                    self.test_loss.append(test_loss/n_batch)
                    self.test_acc.append(test_acc/n_batch)
                    # self.finish_signal.emit([self.train_loss, self.train_acc,
                    #                          self.test_loss, self.test_acc])
                    #save the network
                    if self.is_save:
                        if (epoch + 1) % self.save_n == 0:
                            print("Save model" + "!" * 10)
                            name = str('../zoo/' + self.current_model + '.npz')
                            tl.files.save_npz(self.network.all_params, name=name)
                            # saver = tf.train.Saver()
                            # save_path = saver.save(sess, model_file_name)

            coord.request_stop()
            coord.join(threads)
            sess.close()

            # self.finish_signal.emit(['hello', 'world', '!'])
            self.finish_signal.emit([self.train_loss, self.train_acc,
                                     self.test_loss, self.test_acc])