#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io
采用tensorflow实现回归，https://www.tensorflow.org/tutorials/keras/basic_regression?hl=zh-cn
预测汽车的燃油效率
@version : 1.0
@file : regression.py
@software : PyCharm
@time : 2019/1/17 23:12

"""
from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print(tf.__version__)


def get_data(path):
    """
    从path中获取数据
    :param path:
    :return:
    """
    # mpg数据总共八列，第一列就是我们需要回归的目标
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    # na_values一组用于替换NA/NaN的值
    raw_dataset = pd.read_csv(path, names=column_names, na_values='?', comment='\t',
                              sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    print("数据最后几行:")
    print(raw_dataset.tail())
    return dataset


def clean_data(data):
    """
    清理数据，将所有带有nan的行去掉，同时将origin进行one-hot编码
    :param data:
    :return:
    """
    print("data info:")
    print(data.isna().sum())
    data = data.dropna()
    origin = data.pop('Origin')
    print(data.head())
    # 这里会有警告信息
    data['USA'] = (origin == 1) * 1.0
    data['Europe'] = (origin == 2) * 1.0
    data['Japan'] = (origin == 3) * 1.0
    print("处理后的数据:")
    print(data.tail())
    return data


def split_train_test(data):
    """
    将数据集分成训练集和测试集
    从数据中随机抽取80%的比例作为训练集
    :param data:
    :return:
    """
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset = data.drop(train_dataset.index)
    return train_dataset, test_dataset


def inspect_data(train_dataset):
    """
    探索数据
    :param train_dataset:
    :return:去除MPG后的数据
    """
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    train_stats = train_dataset.describe()
    train_stats.pop('MPG')
    print("训练集统计信息:")
    print(train_stats)
    train_stats = train_stats.transpose()
    return train_stats


def norm(x, train_stats):
    """
    对特征进行标准化
    :param x:
    :param train_stats:
    :return:
    """
    return (x - train_stats['mean']) / train_stats['std']


# ----------------下面是模型相关内容 ------------
def build_model(feature_num):
    """
    构建模型
    :param feature_num: 特征的数目
    :return:
    """
    # model第一层 input_shape指定接收的维度为(*, feture_num)，输出为(*, 64)
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[feature_num]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # 损失函数：均方误差，评价指标：mse，mae(平均绝对误差)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    print('model summary:')
    print(model.summary())
    return model


def train_model(model, train_data, train_label, early_stop=True):
    """
    训练模型
    :param model:
    :param train_data:
    :param train_label:
    :param early_stop:表示是否根据验证集结果来提前终止训练
    :return: 返回训练过程中的信息，已经将数据处理为pandas中的Dataframe类型
    """
    if early_stop:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        callbacks_value = [early_stop, PrintDot()]
    else:
        callbacks_value = [PrintDot()]

    EPOCHS = 1000
    history = model.fit(train_data, train_label, epochs=EPOCHS, validation_split=0.2,
                        verbose=0, callbacks=callbacks_value)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("hist内容:")
    print(hist.tail())
    return model, hist


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show()


def predict(model, test_data, test_label):
    test_predictions = model.predict(test_data).flatten()
    plt.scatter(test_label, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    error = test_predictions - test_label
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")


if __name__ == '__main__':
    """
    整个程序的流程分为两个部分：
    1. 数据准备阶段：
        - 获取数据
        - 清理数据
        - 数据分隔为训练集和测试集
        - 探索数据
        - 归一化
    2. 模型训练阶段：
        - 构建模型
        - 训练模型(可以提前退出训练)
        - 预测(采用测试集)
    """
    # ------- 数据准备阶段
    mpg_path = './data/auto-mpg/auto-mpg.data'
    dataset = get_data(mpg_path)
    dataset = clean_data(dataset)
    train_dataset, test_dataset = split_train_test(dataset)
    train_stats = inspect_data(train_dataset)
    train_label = train_dataset.pop('MPG')
    print("训练集：")
    print(train_dataset.head())
    test_label = test_dataset.pop('MPG')
    print("训练集维度:", train_dataset.shape)
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)
    print("处理后的训练集:")
    print(normed_train_data.head())
    # -------- 建立模型阶段
    feature_num = len(normed_train_data.keys())
    model = build_model(feature_num)
    model, hist = train_model(model, normed_train_data, train_label, early_stop=True)
    plot_history(hist)
    predict(model, normed_test_data, test_label)
