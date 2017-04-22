# -*- coding: utf-8 -*-
from PyQt4 import QtCore
import tensorflow as tf
import tensorlayer as tl
import model_utils
import inference
import os
import numpy as np

class TestThread(QtCore.QThread):
    finish_signal = QtCore.pyqtSignal(list)
    def __init__(self, current_model, graph, parent=None):
        super(TestThread, self).__init__(parent)
        self.current_model = current_model
        self.graph = graph

        with self.graph.as_default():
            self.size = model_utils.model_size(self.current_model)
            self.X_holder = tf.placeholder(tf.float32, shape=[None, self.size[0], self.size[1], self.size[2]])
            self.y_holder = tf.placeholder(tf.int32, shape=[None, ])
            self.network = inference.load_model(inputs=self.X_holder, model_name=self.current_model)

    def run(self):
        with self.graph.as_default():
            print self.y_holder.dtype, self.y_holder.shape
            self.y = self.network.outputs
            self.props = tf.nn.softmax(self.y)

    def predict(self, img):
        with tf.Session(graph=self.graph) as sess:
            tl.layers.initialize_global_variables(sess)
            load_path = str('../zoo/' + self.current_model + '.npz')


            if os.path.isfile(load_path):
                if self.current_model == "VGG16_224_224_3":
                    npz = np.load(load_path)
                    params = []
                    for val in sorted(npz.items()):
                        print "Loading %s" % str(val[1].shape)
                        params.append(val[1])
                    tl.files.assign_params(sess, params, self.network)
                else:
                    tl.files.load_and_assign_npz(sess, name=load_path, network=self.network)
                print "loaded pre_trained model!!!"
            else:
                print "There does not exist pre_trained model!!!"



            props = sess.run(self.props, feed_dict={self.X_holder: [img]})[0]
            print "\nprops: ", props
            return props
