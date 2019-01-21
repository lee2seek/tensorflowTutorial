#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : save_and_restore_model.py
@software : PyCharm
@time : 2019/1/21 14:53

展示如何采用tensorflow 中的 keras API来保存和恢复模型
https://www.tensorflow.org/tutorials/keras/save_and_restore_models?hl=zh-cn

"""
from __future__ import print_function, absolute_import, division
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.__version__


def get_mnist_data(path, size=None):
    """
    获取mnist数据
    :param path:
    :param size: 提取前多少条
    :return:
    """
    # tf.keras.datasets.mnist.load_data()
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print("x_train type:", type(x_train))
    print("x_train shape:", x_train.shape)
    if size:
        y_train = y_train[:size]
        y_test = y_test[:size]
        # reshape将训练集转为二维的，其中第二维的大小固定为784,
        x_train = x_train[:size].reshape(-1, 28 * 28) / 255.0
        x_test = x_test[:size].reshape(-1, 28 * 28) / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_model():
    """
    建立一个较为简单的神经网络
    :return:
    """
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    print("model summary:")
    print(model.summary())
    return model


def train_model(model, model_dir, train_x, train_y, test_x, test_y):
    """
    训练模型,演示了如何保存模型
    :param model:
    :param model_dir:
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    checkpoint_path = model_dir
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("model save path:", checkpoint_dir)
    # 检查点的回调，每次epoch后，都会保存模型到checkpoint_path，save_weights_only表示仅仅保存权重
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y),
              callbacks=[cp_callback])

    return model


def compare_untrain_model_vs_train_model(test_x, test_y, model_path):
    """
    比较未训练模型和加载已经训练好的模型
    :param test_x:
    :param test_y:
    :param model_path:
    :return:
    """
    model = create_model()
    loss, acc = model.evaluate(test_x, test_y)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    print("Untrained model, accuracy: {:.5f}".format(100 * loss))

    model.load_weights(model_path)
    loss, acc = model.evaluate(test_x, test_y)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, accuracy: {:.5f}".format(100 * loss))


def train_model_and_set_callback(model, model_dir, train_x, train_y, test_x, test_y):
    """
    训练模型并且定制保存模型的回调参数
    :param model:
    :param model_dir:
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    checkpoint_path = model_dir
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # period指定间隔多长epoch保存一次模型
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                     save_weights_only=True, period=5)
    model.fit(train_x, train_y, epochs=50,
              validation_data=(test_x, test_y), callbacks=[cp_callback],
              verbose=0)


def save_whole_model(train_x, train_y):
    """
    保存整个模型
    :param train_x:
    :param train_y:
    :return:
    """
    model = create_model()
    model.fit(train_x, train_y, epochs=5)
    # 手动保存模型参数
    model.save_weights("./model/my_weights/my_checkpoint")
    # 保存整个模型,包括 权重值,模型配置（架构）,优化器配置
    model.save('./model/mnist/my_model.h5')


def load_whole_model(model_path, test_x, test_y):
    """
    加载整个模型
    :param model_path:
    :param test_x:
    :param test_y:
    :return:
    """
    new_model = tf.keras.models.load_model(model_path)
    print("new model sunmary:")
    print(new_model.summary())
    loss, acc = new_model.evaluate(test_x, test_y)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    mnist_path = "./data/mnist/mnist.npz"
    (train_x, train_y), (test_x, test_y) = get_mnist_data(path=mnist_path, size=1000)
    model = create_model()
    # "./model/mnist"中model表示模型存储的文件夹名称，mnist是模型文件的前缀
    model_dir = "./model/mnist"
    train_model(model, model_dir, train_x, train_y, test_x, test_y)
    compare_untrain_model_vs_train_model(test_x, test_y, model_dir)

    # 定制模型保存策略
    # cp-{epoch:04d}.ckpt 指定了模型文件前缀格式，注意{}的使用
    model_dir = "./model/mnist/5epoch/cp-{epoch:04d}.ckpt"
    train_model_and_set_callback(model, model_dir, train_x, train_y, test_x, test_y)
    save_whole_model(train_x, train_y)
    model_path = './model/mnist/my_model.h5'
    load_whole_model(model_path, test_x, test_y)