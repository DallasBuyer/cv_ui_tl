# -*- coding: utf-8 -*-
# various cnn models for image classification

import tensorflow as tf
import tensorlayer as tl

def load_model(inputs, model_name):
    model_dict = {"MLP":mlp, "CNN_28_28_1":cnn_28_28_1, "CNN_32_32_3":cnn_32_32_3, "CNN_32_32_1":cnn_32_32_1,
                  "AlexNet_28_28_1":alexnet_28_28_1, "AlexNet_32_32_3":alexnet_32_32_3, "VGG16_224_224_3":vgg16_224_224_3,
                  "Inception_v1_224_224_1":inception_v1_224_224_1, "Inception_v1_224_224_3":inception_v1_224_224_3}
    network = model_dict[str(model_name)](inputs)
    return network

def mlp(inputs):
    network = tl.layers.InputLayer(inputs, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output_layer')
    return network

def cnn_28_28_1(inputs):
    network = tl.layers.InputLayer(inputs, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 1, 32],  # 32 features for each 5x5 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='cnn_layer1')  # output: 28*28*32
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1', )  # output: 14*14*32
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 32, 64],  # 64 features for each 5x5 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='cnn_layer2')  # output: 14*14*64
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2', )  # output: 7*7*64

    network = tl.layers.FlattenLayer(network, name='flatten')
    #network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=256,
                                   act=tf.nn.relu, name='relu1')
    #network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output')
    return network

def cnn_32_32_3(inputs):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    b_init = tf.constant_initializer(value=0)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    network = tl.layers.InputLayer(inputs, name='input_layer')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 3, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    W_init=W_init,
                                    b_init=b_init,
                                    name='cnn_layer1')  # 32*32*64
    network = tl.layers.PoolLayer(network, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1')  # 16*16*64
    network.outputs = tf.nn.lrn(network.outputs, 4,
                                bias=1.0,
                                alpha=0.001 / 9.0,
                                beta=0.75,
                                name='norm1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    W_init=W_init,
                                    b_init=b_init,
                                    name='cnn_layer2')  # 16*16*64
    network.outputs = tf.nn.lrn(network.outputs, 4,
                                bias=1.0,
                                alpha=0.001 / 9.0,
                                beta=0.75,
                                name='norm2')
    network = tl.layers.PoolLayer(network, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2')  # 8*8*64
    network = tl.layers.FlattenLayer(network, name='flatten_layer')
    network = tl.layers.DenseLayer(network, n_units=384,
                                   act=tf.nn.relu,
                                   W_init=W_init2,
                                   b_init=b_init2,
                                   name='relu1')
    network = tl.layers.DenseLayer(network, n_units=192,
                                   act=tf.nn.relu,
                                   W_init=W_init2,
                                   b_init=b_init2,
                                   name='relu2')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                   b_init=tf.constant_initializer(value=0.0),
                                   name='output_layer')
    return network

def cnn_32_32_1(inputs):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    b_init = tf.constant_initializer(value=0)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    network = tl.layers.InputLayer(inputs, name='input_layer')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 1, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    W_init=W_init,
                                    b_init=b_init,
                                    name='cnn_layer1')  # 32*32*64
    network = tl.layers.PoolLayer(network, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1')  # 16*16*64
    network.outputs = tf.nn.lrn(network.outputs, 4,
                                bias=1.0,
                                alpha=0.001 / 9.0,
                                beta=0.75,
                                name='norm1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    W_init=W_init,
                                    b_init=b_init,
                                    name='cnn_layer2')  # 16*16*64
    network.outputs = tf.nn.lrn(network.outputs, 4,
                                bias=1.0,
                                alpha=0.001 / 9.0,
                                beta=0.75,
                                name='norm2')
    network = tl.layers.PoolLayer(network, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2')  # 8*8*64
    network = tl.layers.FlattenLayer(network, name='flatten_layer')
    network = tl.layers.DenseLayer(network, n_units=384,
                                   act=tf.nn.relu,
                                   W_init=W_init2,
                                   b_init=b_init2,
                                   name='relu1')
    network = tl.layers.DenseLayer(network, n_units=192,
                                   act=tf.nn.relu,
                                   W_init=W_init2,
                                   b_init=b_init2,
                                   name='relu2')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                   b_init=tf.constant_initializer(value=0.0),
                                   name='output_layer')
    return network

def alexnet_28_28_1(inputs):
    network = tl.layers.InputLayer(inputs, name='input_layer')  # 28*28*1
    # conv layer 1
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[7, 7, 1, 96],
                                    strides=[1, 3, 3, 1],
                                    padding='SAME',  # padding 'SAME'/out_height=ceil(in_height/strides[1])
                                    name='conv_1')  # 10*10*96
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=2,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_1')  # 10*10*96
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  # padding 'VALID'/out_height=ceil((in_height-filter_height+1)/strides[1])
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1')  # 9*9*96
    # conv layer 2
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 96, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_2')  # 9*9*256
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=2,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_2')  # 9*9*256
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2')  # 8*8*256
    # conv layer 3
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 256, 384],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_3')  # 8*8*384
    # conv layer 4
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 384, 384],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_4')  # 8*8*384
    # conv layer 5
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 384, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_5')  # 8*8*256
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer5')  # 3*3*256
    # fc layer 1
    network = tl.layers.FlattenLayer(network, name='flatten_layer')  # output 3*3*256 = 2304*1
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   b_init=tf.constant_initializer(value=1.0),
                                   W_init=tf.truncated_normal_initializer(stddev=0.01),
                                   name='fc_layer1')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop_1')
    # fc layer 2
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   b_init=tf.constant_initializer(value=1.0),
                                   name='fc_layer2')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop_2')
    # fc layer 3
    network = tl.layers.DenseLayer(network,
                                   n_units=10,
                                   act=tf.identity,
                                   name='outputs')
    return network

def alexnet_32_32_3(inputs):
    network = tl.layers.InputLayer(inputs, name='input_layer')  # 32*32*3
    # conv layer 1
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[7, 7, 3, 96],
                                    strides=[1, 3, 3, 1],
                                    padding='SAME',  # padding 'SAME'/out_height=ceil(in_height/strides[1])
                                    name='conv_1')  # 11*11*96
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=2,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_1')  # 11*11*96
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  # padding 'VALID'/out_height=ceil((in_height-filter_height+1)/strides[1])
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1')  # 11*11*96
    # conv layer 2
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[5, 5, 96, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_2')  # 11*11*256
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=2,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_2')  # 11*11*256
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2')  # 10*10*256
    # conv layer 3
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 256, 384],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_3')  # 10*10*384
    # conv layer 4
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 384, 384],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_4')  # 10*10*384
    # conv layer 5
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 384, 256],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='conv_5')  # 5*5*256
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer5')  # 3*3*256
    # fc layer 1
    network = tl.layers.FlattenLayer(network, name='flatten_layer')  # output 3*3*256 = 2304*1
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   b_init=tf.constant_initializer(value=1.0),
                                   W_init=tf.truncated_normal_initializer(stddev=0.01),
                                   name='fc_layer1')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop_1')
    # fc layer 2
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   b_init=tf.constant_initializer(value=1.0),
                                   name='fc_layer2')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop_2')
    # fc layer 3
    network = tl.layers.DenseLayer(network,
                                   n_units=10,
                                   act=tf.identity,
                                   name='outputs')
    return network

def inception_v1_224_224_3(inputs):
    network = tl.layers.InputLayer(inputs, 'input_layer') # 224*224*3
    # layer-1 conv_1
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[7, 7, 3, 64],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME', # ceil(in_height/strides[1])
                                    name='conv_1')  # 112*112*64
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_1') # 56*56*64
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=5,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_1') # 56*56*64
    # layer-2 conv_2
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[1, 1, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_2') # 56*56*64
    # layer-3 conv_3
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 64, 192],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_3')  # 56*56*192
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=5,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_2')  # 56*56*192
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_2')  # 28*28*192
    # layer-4 inception_3a--first depth
    inception_3a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 192, 96],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3a_3_3_reduce')# 28*28*96
    inception_3a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 192, 16],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3a_5_5_reduce')# 28*28*16
    inception_3a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_3a_pool_reduce')# 28*28*192
    # layer-5 inception_3a--second depth
    inception_3a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 192, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_1_1') # 28*28*64
    inception_3a_3_3 = tl.layers.Conv2dLayer(inception_3a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 96, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_3_3') # 28*28*128
    inception_3a_5_5 = tl.layers.Conv2dLayer(inception_3a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 16, 32],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_5_5') # 28*28*32
    inception_3a_pool = tl.layers.Conv2dLayer(inception_3a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 192, 32],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_3a_pool') # 28*28*32
    network = tf.concat(values=[inception_3a_1_1, inception_3a_3_3,
                                inception_3a_5_5, inception_3a_pool],
                        axis=3,
                        name='concat_3a') # 28*28*256
    # layer-6 inception_3b--first depth
    inception_3b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 256, 128],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3b_3_3_reduce')  # 28*28*128
    inception_3b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 256, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3b_5_5_reduce')  # 28*28*32
    inception_3b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_3b_pool_reduce')  # 28*28*256
    # layer-7 inception_3b--second depth
    inception_3b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 256, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_1_1')  # 28*28*128
    inception_3b_3_3 = tl.layers.Conv2dLayer(inception_3b_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 128, 192],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_3_3')  # 28*28*192
    inception_3b_5_5 = tl.layers.Conv2dLayer(inception_3b_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 96],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_5_5')  # 28*28*96
    inception_3b_pool = tl.layers.Conv2dLayer(inception_3b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 256, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_3b_pool')  # 28*28*64
    network = tf.concat(values=[inception_3b_1_1, inception_3b_3_3,
                                inception_3b_5_5, inception_3b_pool],
                        axis=3,
                        name='concat_3b')  # 28*28*480
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_3') # 14*14*480
    # layer-8 inception_4a--first depth
    inception_4a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                   act=tf.nn.relu,
                                                   shape=[1, 1, 480, 96],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   name='inception_4a_3_3_reduce')  # 14*14*96
    inception_4a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 480, 16],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4a_5_5_reduce')  # 14*14*16
    inception_4a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4a_pool_reduce')  # 14*14*480
    # layer-9 inception_4a--second depth
    inception_4a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 480, 192],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_1_1')  # 14*14*192
    inception_4a_3_3 = tl.layers.Conv2dLayer(inception_4a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 96, 208],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_3_3')  # 14*14*208
    inception_4a_5_5 = tl.layers.Conv2dLayer(inception_4a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 16, 48],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_5_5')  # 14*14*48
    inception_4a_pool = tl.layers.Conv2dLayer(inception_4a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 480, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4a_pool')  # 14*14*64
    network = tf.concat(values=[inception_4a_1_1, inception_4a_3_3,
                                inception_4a_5_5, inception_4a_pool],
                        axis=3,
                        name='concat_4a')  # 14*14*512
    # layer-10 inception_4b--first depth
    inception_4b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 112],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4b_3_3_reduce')  # 14*14*112
    inception_4b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 24],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4b_5_5_reduce')  # 14*14*24
    inception_4b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4b_pool_reduce')  # 14*14*512
    # layer-11 inception_4b--second depth
    inception_4b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 160],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_1_1')  # 14*14*160
    inception_4b_3_3 = tl.layers.Conv2dLayer(inception_4b_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 112, 224],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_3_3')  # 14*14*224
    inception_4b_5_5 = tl.layers.Conv2dLayer(inception_4b_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 24, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_5_5')  # 14*14*64
    inception_4b_pool = tl.layers.Conv2dLayer(inception_4b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4b_pool')  # 14*14*64
    network = tf.concat(values=[inception_4b_1_1, inception_4b_3_3,
                                inception_4b_5_5, inception_4b_pool],
                        axis=3,
                        name='concat_4b')  # 14*14*512
    # layer-12 inception_4c--first depth
    inception_4c_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 128],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4c_3_3_reduce')  # 14*14*128
    inception_4c_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 24],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4c_5_5_reduce')  # 14*14*24
    inception_4c_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4c_pool_reduce')  # 14*14*512
    # layer-13 inception_4c--second depth
    inception_4c_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_1_1')  # 14*14*128
    inception_4c_3_3 = tl.layers.Conv2dLayer(inception_4c_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 128, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_3_3')  # 14*14*256
    inception_4c_5_5 = tl.layers.Conv2dLayer(inception_4c_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 24, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_5_5')  # 14*14*64
    inception_4c_pool = tl.layers.Conv2dLayer(inception_4c_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4c_pool')  # 14*14*64
    network = tf.concat(values=[inception_4c_1_1, inception_4c_3_3,
                                inception_4c_5_5, inception_4c_pool],
                        axis=3,
                        name='concat_4c')  # 14*14*512
    # layer-14 inception_4d--first depth
    inception_4d_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 144],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4d_3_3_reduce')  # 14*14*144
    inception_4d_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4d_5_5_reduce')  # 14*14*32
    inception_4d_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4d_pool_reduce')  # 14*14*512
    # layer-15 inception_4d--second depth
    inception_4d_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 112],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_1_1')  # 14*14*112
    inception_4d_3_3 = tl.layers.Conv2dLayer(inception_4d_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 144, 288],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_3_3')  # 14*14*288
    inception_4d_5_5 = tl.layers.Conv2dLayer(inception_4d_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_5_5')  # 14*14*64
    inception_4d_pool = tl.layers.Conv2dLayer(inception_4d_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4d_pool')  # 14*14*64
    network = tf.concat(values=[inception_4d_1_1, inception_4d_3_3,
                                inception_4d_5_5, inception_4d_pool],
                        axis=3,
                        name='concat_4d')  # 14*14*528
    # layer-16 inception_4e--first depth
    inception_4e_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 528, 160],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4e_3_3_reduce')  # 14*14*160
    inception_4e_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 528, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4e_5_5_reduce')  # 14*14*32
    inception_4e_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4e_pool_reduce')  # 14*14*528
    # layer-17 inception_4e--second depth
    inception_4e_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 528, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_1_1')  # 14*14*256
    inception_4e_3_3 = tl.layers.Conv2dLayer(inception_4e_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 160, 320],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_3_3')  # 14*14*320
    inception_4e_5_5 = tl.layers.Conv2dLayer(inception_4e_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_5_5')  # 14*14*128
    inception_4e_pool = tl.layers.Conv2dLayer(inception_4e_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 528, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4e_pool')  # 14*14*128
    network = tf.concat(values=[inception_4e_1_1, inception_4e_3_3,
                                inception_4e_5_5, inception_4e_pool],
                        axis=3,
                        name='concat_4e')  # 14*14*832
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool_4')  # 7*7*832
    # layer-18 inception_5a--first layer
    inception_5a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 160],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5a_3_3_reduce')  # 7*7*160
    inception_5a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5a_5_5_reduce')  # 7*7*32
    inception_5a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_5a_pool_reduce')  # 7*7*832
    # layer-19 inception_5a--second layer
    inception_5a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 832, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_1_1')  # 7*7*256
    inception_5a_3_3 = tl.layers.Conv2dLayer(inception_5a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 160, 320],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_3_3')  # 7*7*320
    inception_5a_5_5 = tl.layers.Conv2dLayer(inception_5a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_5_5')  # 7*7*128
    inception_5a_pool = tl.layers.Conv2dLayer(inception_5a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 832, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_5a_pool')  # 7*7*128
    network = tf.concat(values=[inception_5a_1_1, inception_5a_3_3,
                                inception_5a_5_5, inception_5a_pool],
                        axis=3,
                        name='concat_5a')  # 7*7*832
    # layer-20 inception_5b--first layer
    inception_5b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 192],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5b_3_3_reduce')  # 7*7*192
    inception_5b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 48],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5b_5_5_reduce')  # 7*7*48
    inception_5b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_5b_pool_reduce')  # 7*7*832
    # layer-21 inception_5b--second layer
    inception_5b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 832, 384],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_1_1')  # 7*7*384
    inception_5b_3_3 = tl.layers.Conv2dLayer(inception_5a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 192, 384],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_3_3')  # 7*7*384
    inception_5b_5_5 = tl.layers.Conv2dLayer(inception_5a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 48, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_5_5')  # 7*7*128
    inception_5b_pool = tl.layers.Conv2dLayer(inception_5b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 832, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_5b_pool')  # 7*7*128
    network = tf.concat(values=[inception_5b_1_1, inception_5b_3_3,
                                inception_5b_5_5, inception_5b_pool],
                        axis=3,
                        name='concat_5b')  # 7*7*1024
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 7, 7, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  pool=tf.nn.avg_pool,
                                  name='avg_pool')  # 1*1*1024
    # network = tl.layers.DropoutLayer(network, keep=0.4, name='dropout_layer') #1*1*1024
    network = tl.layers.FlattenLayer(network, name='flatten_layer') # 1024
    # layer-22
    network = tl.layers.DenseLayer(network,
                                   act=tf.nn.relu,
                                   n_units=1000,
                                   name='fc_layer1')
    return network

def inception_v1_224_224_1(inputs):
    network = tl.layers.InputLayer(inputs, 'input_layer') # 224*224*1
    # layer-1 conv_1
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[7, 7, 1, 64],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME', # ceil(in_height/strides[1])
                                    name='conv_1')  # 112*112*64
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_1') # 56*56*64
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=5,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_1') # 56*56*64
    # layer-2 conv_2
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[1, 1, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_2') # 56*56*64
    # layer-3 conv_3
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 64, 192],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv_3')  # 56*56*192
    network.outputs = tf.nn.lrn(network.outputs,
                                depth_radius=5,
                                bias=1.0,
                                alpha=0.0001,
                                beta=0.75,
                                name='lrn_2')  # 56*56*192
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_2')  # 28*28*192
    # layer-4 inception_3a--first depth
    inception_3a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 192, 96],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3a_3_3_reduce')# 28*28*96
    inception_3a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 192, 16],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3a_5_5_reduce')# 28*28*16
    inception_3a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_3a_pool_reduce')# 28*28*192
    # layer-5 inception_3a--second depth
    inception_3a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 192, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_1_1') # 28*28*64
    inception_3a_3_3 = tl.layers.Conv2dLayer(inception_3a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 96, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_3_3') # 28*28*128
    inception_3a_5_5 = tl.layers.Conv2dLayer(inception_3a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 16, 32],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3a_5_5') # 28*28*32
    inception_3a_pool = tl.layers.Conv2dLayer(inception_3a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 192, 32],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_3a_pool') # 28*28*32
    # network = tf.concat(values=[inception_3a_1_1.outputs, inception_3a_3_3.outputs,
    #                             inception_3a_5_5.outputs, inception_3a_pool.outputs],
    #                     axis=3,
    #                     name='concat_3a') # 28*28*256
    network = tl.layers.ConcatLayer([inception_3a_1_1, inception_3a_3_3,
                                     inception_3a_5_5, inception_3a_pool],
                                    concat_dim=3,
                                    name='concat_3a')
    # layer-6 inception_3b--first depth
    inception_3b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 256, 128],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3b_3_3_reduce')  # 28*28*128
    inception_3b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 256, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_3b_5_5_reduce')  # 28*28*32
    inception_3b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_3b_pool_reduce')  # 28*28*256
    # layer-7 inception_3b--second depth
    inception_3b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 256, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_1_1')  # 28*28*128
    inception_3b_3_3 = tl.layers.Conv2dLayer(inception_3b_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 128, 192],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_3_3')  # 28*28*192
    inception_3b_5_5 = tl.layers.Conv2dLayer(inception_3b_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 96],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_3b_5_5')  # 28*28*96
    inception_3b_pool = tl.layers.Conv2dLayer(inception_3b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 256, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_3b_pool')  # 28*28*64
    # network = tf.concat(values=[inception_3b_1_1, inception_3b_3_3,
    #                             inception_3b_5_5, inception_3b_pool],
    #                     axis=3,
    #                     name='concat_3b')  # 28*28*480
    network = tl.layers.ConcatLayer([inception_3b_1_1, inception_3b_3_3,
                                     inception_3b_5_5, inception_3b_pool],
                                    concat_dim=3,
                                    name='concat_3b')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_3') # 14*14*480
    # layer-8 inception_4a--first depth
    inception_4a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                   act=tf.nn.relu,
                                                   shape=[1, 1, 480, 96],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   name='inception_4a_3_3_reduce')  # 14*14*96
    inception_4a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 480, 16],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4a_5_5_reduce')  # 14*14*16
    inception_4a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4a_pool_reduce')  # 14*14*480
    # layer-9 inception_4a--second depth
    inception_4a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 480, 192],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_1_1')  # 14*14*192
    inception_4a_3_3 = tl.layers.Conv2dLayer(inception_4a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 96, 208],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_3_3')  # 14*14*208
    inception_4a_5_5 = tl.layers.Conv2dLayer(inception_4a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 16, 48],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4a_5_5')  # 14*14*48
    inception_4a_pool = tl.layers.Conv2dLayer(inception_4a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 480, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4a_pool')  # 14*14*64
    # network = tf.concat(values=[inception_4a_1_1, inception_4a_3_3,
    #                             inception_4a_5_5, inception_4a_pool],
    #                     axis=3,
    #                     name='concat_4a')  # 14*14*512
    network = tl.layers.ConcatLayer([inception_4a_1_1, inception_4a_3_3,
                                     inception_4a_5_5, inception_4a_pool],
                                    concat_dim=3,
                                    name='concat_4a')
    # layer-10 inception_4b--first depth
    inception_4b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 112],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4b_3_3_reduce')  # 14*14*112
    inception_4b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 24],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4b_5_5_reduce')  # 14*14*24
    inception_4b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4b_pool_reduce')  # 14*14*512
    # layer-11 inception_4b--second depth
    inception_4b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 160],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_1_1')  # 14*14*160
    inception_4b_3_3 = tl.layers.Conv2dLayer(inception_4b_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 112, 224],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_3_3')  # 14*14*224
    inception_4b_5_5 = tl.layers.Conv2dLayer(inception_4b_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 24, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4b_5_5')  # 14*14*64
    inception_4b_pool = tl.layers.Conv2dLayer(inception_4b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4b_pool')  # 14*14*64
    # network = tf.concat(values=[inception_4b_1_1, inception_4b_3_3,
    #                             inception_4b_5_5, inception_4b_pool],
    #                     axis=3,
    #                     name='concat_4b')  # 14*14*512
    network = tl.layers.ConcatLayer([inception_4b_1_1, inception_4b_3_3,
                                     inception_4b_5_5, inception_4b_pool],
                                    concat_dim=3,
                                    name='concat_4b')
    # layer-12 inception_4c--first depth
    inception_4c_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 128],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4c_3_3_reduce')  # 14*14*128
    inception_4c_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 24],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4c_5_5_reduce')  # 14*14*24
    inception_4c_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4c_pool_reduce')  # 14*14*512
    # layer-13 inception_4c--second depth
    inception_4c_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_1_1')  # 14*14*128
    inception_4c_3_3 = tl.layers.Conv2dLayer(inception_4c_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 128, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_3_3')  # 14*14*256
    inception_4c_5_5 = tl.layers.Conv2dLayer(inception_4c_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 24, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4c_5_5')  # 14*14*64
    inception_4c_pool = tl.layers.Conv2dLayer(inception_4c_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4c_pool')  # 14*14*64
    # network = tf.concat(values=[inception_4c_1_1, inception_4c_3_3,
    #                             inception_4c_5_5, inception_4c_pool],
    #                     axis=3,
    #                     name='concat_4c')  # 14*14*512
    network = tl.layers.ConcatLayer([inception_4c_1_1, inception_4c_3_3,
                                     inception_4c_5_5, inception_4c_pool],
                                    concat_dim=3,
                                    name='concat_4c')
    # layer-14 inception_4d--first depth
    inception_4d_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 144],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4d_3_3_reduce')  # 14*14*144
    inception_4d_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 512, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4d_5_5_reduce')  # 14*14*32
    inception_4d_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4d_pool_reduce')  # 14*14*512
    # layer-15 inception_4d--second depth
    inception_4d_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 512, 112],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_1_1')  # 14*14*112
    inception_4d_3_3 = tl.layers.Conv2dLayer(inception_4d_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 144, 288],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_3_3')  # 14*14*288
    inception_4d_5_5 = tl.layers.Conv2dLayer(inception_4d_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 64],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4d_5_5')  # 14*14*64
    inception_4d_pool = tl.layers.Conv2dLayer(inception_4d_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 512, 64],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4d_pool')  # 14*14*64
    # network = tf.concat(values=[inception_4d_1_1, inception_4d_3_3,
    #                             inception_4d_5_5, inception_4d_pool],
    #                     axis=3,
    #                     name='concat_4d')  # 14*14*528
    network = tl.layers.ConcatLayer([inception_4d_1_1, inception_4d_3_3,
                                     inception_4d_5_5, inception_4d_pool],
                                    concat_dim=3,
                                    name='concat_4d')
    # layer-16 inception_4e--first depth
    inception_4e_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 528, 160],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4e_3_3_reduce')  # 14*14*160
    inception_4e_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 528, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_4e_5_5_reduce')  # 14*14*32
    inception_4e_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_4e_pool_reduce')  # 14*14*528
    # layer-17 inception_4e--second depth
    inception_4e_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 528, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_1_1')  # 14*14*256
    inception_4e_3_3 = tl.layers.Conv2dLayer(inception_4e_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 160, 320],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_3_3')  # 14*14*320
    inception_4e_5_5 = tl.layers.Conv2dLayer(inception_4e_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_4e_5_5')  # 14*14*128
    inception_4e_pool = tl.layers.Conv2dLayer(inception_4e_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 528, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_4e_pool')  # 14*14*128
    # network = tf.concat(values=[inception_4e_1_1, inception_4e_3_3,
    #                             inception_4e_5_5, inception_4e_pool],
    #                     axis=3,
    #                     name='concat_4e')  # 14*14*832
    network = tl.layers.ConcatLayer([inception_4e_1_1, inception_4e_3_3,
                                     inception_4e_5_5, inception_4e_pool],
                                    concat_dim=3,
                                    name='concat_4e')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool_4')  # 7*7*832
    # layer-18 inception_5a--first layer
    inception_5a_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 160],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5a_3_3_reduce')  # 7*7*160
    inception_5a_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 32],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5a_5_5_reduce')  # 7*7*32
    inception_5a_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_5a_pool_reduce')  # 7*7*832
    # layer-19 inception_5a--second layer
    inception_5a_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 832, 256],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_1_1')  # 7*7*256
    inception_5a_3_3 = tl.layers.Conv2dLayer(inception_5a_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 160, 320],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_3_3')  # 7*7*320
    inception_5a_5_5 = tl.layers.Conv2dLayer(inception_5a_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 32, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5a_5_5')  # 7*7*128
    inception_5a_pool = tl.layers.Conv2dLayer(inception_5a_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 832, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_5a_pool')  # 7*7*128
    # network = tf.concat(values=[inception_5a_1_1, inception_5a_3_3,
    #                             inception_5a_5_5, inception_5a_pool],
    #                     axis=3,
    #                     name='concat_5a')  # 7*7*832
    network = tl.layers.ConcatLayer([inception_5a_1_1, inception_5a_3_3,
                                     inception_5a_5_5, inception_5a_pool],
                                    concat_dim=3,
                                    name='concat_5a')
    # layer-20 inception_5b--first layer
    inception_5b_3_3_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 192],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5b_3_3_reduce')  # 7*7*192
    inception_5b_5_5_reduce = tl.layers.Conv2dLayer(network,
                                                    act=tf.nn.relu,
                                                    shape=[1, 1, 832, 48],
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME',
                                                    name='inception_5b_5_5_reduce')  # 7*7*48
    inception_5b_pool_reduce = tl.layers.PoolLayer(network,
                                                   ksize=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME',
                                                   pool=tf.nn.max_pool,
                                                   name='inception_5b_pool_reduce')  # 7*7*832
    # layer-21 inception_5b--second layer
    inception_5b_1_1 = tl.layers.Conv2dLayer(network,
                                             act=tf.nn.relu,
                                             shape=[1, 1, 832, 384],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_1_1')  # 7*7*384
    inception_5b_3_3 = tl.layers.Conv2dLayer(inception_5b_3_3_reduce,
                                             act=tf.nn.relu,
                                             shape=[3, 3, 192, 384],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_3_3')  # 7*7*384
    inception_5b_5_5 = tl.layers.Conv2dLayer(inception_5b_5_5_reduce,
                                             act=tf.nn.relu,
                                             shape=[5, 5, 48, 128],
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='inception_5b_5_5')  # 7*7*128
    inception_5b_pool = tl.layers.Conv2dLayer(inception_5b_pool_reduce,
                                              act=tf.nn.relu,
                                              shape=[1, 1, 832, 128],
                                              strides=[1, 1, 1, 1],
                                              padding='SAME',
                                              name='inception_5b_pool')  # 7*7*128
    # network = tf.concat(values=[inception_5b_1_1, inception_5b_3_3,
    #                             inception_5b_5_5, inception_5b_pool],
    #                     axis=3,
    #                     name='concat_5b')  # 7*7*1024
    network = tl.layers.ConcatLayer([inception_5b_1_1, inception_5b_3_3,
                                     inception_5b_5_5, inception_5b_pool],
                                    concat_dim=3,
                                    name='concat_5b')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 7, 7, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  pool=tf.nn.avg_pool,
                                  name='avg_pool')  # 1*1*1024
    # network = tl.layers.DropoutLayer(network, keep=0.4, name='dropout_layer') #1*1*1024
    network = tl.layers.FlattenLayer(network, name='flatten_layer') # 1024
    # layer-22
    network = tl.layers.DenseLayer(network,
                                   act=tf.nn.relu,
                                   n_units=1000,
                                   name='fc_layer1')
    return network

def vgg16_224_224_3(inputs):
    network = tl.layers.InputLayer(inputs, name='input_layer')

    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                       shape=[1, 1, 1, 3], name='img_mean')

    network.outputs = network.outputs - mean
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 3, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool1')

    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 64, 128],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 128, 128],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool2')

    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 128, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 256, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_2')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 256, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool3')

    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 256, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                                    act = tf.nn.relu,
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_2')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool4')

    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_1')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_2')
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool5')

    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc1_relu')
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc2_relu')
    network = tl.layers.DenseLayer(network,
                                   n_units=1000,
                                   act=tf.identity,
                                   name='fc3_relu')
    return network
