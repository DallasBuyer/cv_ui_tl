# -*- coding:utf-8 -*-

import numpy as np
import tensorlayer as tl
import tensorflow as tf
from PIL import Image
import os
from scipy.misc import imread, imresize


def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    print("Converting data into %s..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)

    for index, img in enumerate(images):
        img_raw = img.tobytes()
        label = int(labels[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())    # Serialize to String
    writer.close()

def read_and_decode(size, filename, is_train=None):
    """ Return tensor to read from TFRecord """
    """
    :param size: three dimension
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                features={'label': tf.FixedLenFeature([], tf.int64),
                          'img_raw': tf.FixedLenFeature([], tf.string)})

    # do image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [size[0], size[1], size[2]])

    # if is_train:
    #     img = tf.random_crop(img, [24, 24, 3])
    #     img = tf.image.random_flip_up_down(img)
    #     img = tf.image.random_brightness(img, max_delta=63)
    #     img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    #     img = tf.image.per_image_standardization(img)
    # else:
    #     img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
    #     img = tf.image.per_image_standardization(img)

    label = tf.cast(features['label'], tf.int32)
    return img, label


def resize_mnist(size):
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1), path="../data/mnist/")
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    new_X_train = np.zeros((num_train, size[0], size[1], 1), dtype=np.float32)
    new_X_test = np.zeros((num_test, size[0], size[1], 1), dtype=np.float32)

    for i in np.arange(num_train):
        temp_img = X_train[i]
        temp_img = temp_img[:, :, 0] * 256
        img_in = Image.fromarray(temp_img)
        img_out = img_in.resize(size, resample=Image.BICUBIC)
        img_array = np.asarray(img_out) / np.float32(256)
        new_X_train[i, :, :, 0] = img_array

    for i in np.arange(num_test):
        temp_img = X_test[i]
        temp_img = temp_img[:, :, 0] * 256
        img_in = Image.fromarray(temp_img)
        img_out = img_in.resize(size, resample=Image.BICUBIC)
        img_array = np.asarray(img_out) / np.float32(256)
        new_X_test[i, :, :, 0] = img_array

    data_to_tfrecord(images=new_X_train, labels=y_train,
                     filename="../data/mnist/train%s_%s.mnist" % (str(size[0]),str(size[1])))
    data_to_tfrecord(images=new_X_test, labels=y_test,
                     filename="../data/mnist/test%s_%s.mnist" % (str(size[0]),str(size[1])))
    return new_X_train, y_train, new_X_test, y_test

def load_dataset(data_name):
    data_dict = {"mnist_28_28_1":["mnist/train28_28.mnist","mnist/test28_28.mnist"],
                 "mnist_32_32_1":["mnist/train32_32.mnist","mnist/test32_32.mnist"],
                 "mnist_224_224_1":["mnist/train224_224.mnist","mnist/test224_224.mnist"],
                 "cifar10_32_32_3":["cifar10/train.cifar10", "cifar10/test.cifar10"]}
    size_dict = {"mnist_28_28_1": (28, 28, 1),
                 "mnist_32_32_1": (32, 32, 1),
                 "mnist_224_224_1": (224, 224, 1),
                 "cifar10_32_32_3": (32, 32, 3)}
    filename = data_dict[str(data_name)]
    size = size_dict[str(data_name)]
    x_train_, y_train_ = read_and_decode(size, filename="../data/%s" % filename[0])
    x_test_, y_test_ = read_and_decode(size, filename="../data/%s" % filename[1])
    print "x_train:", x_train_.shape, "y_train:", y_train_, "x_test: ", x_test_, "y_test: ", y_test_
    return x_train_, y_train_, x_test_, y_test_, size

def image_preprocess(filename, image_size):
    img = Image.open(filename)
    img = img.resize([image_size[0],image_size[1]])
    if image_size[2]==1:
        img = img.convert('L')
        print "read image of 'L' format"
    else:
        img = img.convert('RGB')
        print "read image of 'RGB' format"
    img.load()
    img = np.asarray(img, dtype="float32")
    #img /= 225.
    if image_size[2]==1:
        img /=255.
        img_new = np.ndarray((img.shape[0], img.shape[1], image_size[2]), dtype="float32")
        img_new[:, :, 0] = img
        img = img_new
    return img

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = resize_mnist(size=[224, 224])
    #load_mnist("mnist_28_28_1")
    # a = X_train[5000]
    # b = a[:, :, 0] * 256
    # im = Image.fromarray(b)
    # im.show()