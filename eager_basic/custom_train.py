#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : custom_train.py
@software : PyCharm
@time : 2019/1/23 11:16

采用tensorflow中的一些基础概念,如Tensor, GradientTape, Variable 来构建一个线性回归模型
https://www.tensorflow.org/tutorials/eager/custom_training?hl=zh-cn
"""
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()


class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        # 因为开启了eager模式，所以这里不能使用tf.Variable，而使用tf.contrib.eager.Variable
        self.W = tf.contrib.eager.Variable(5.0)
        self.b = tf.contrib.eager.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()

assert model(3.0).numpy() == 15.0


def loss(predicted_y, desired_y):
    """
    损失函数
    :param predicted_y:
    :param desired_y:
    :return:
    """
    # reduce_mean计算平均值
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


def obtain_data():
    """
    产生一千个输入和加入噪声的一千个输出，等会用这个数据进行训练
    :return:
    """
    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000

    inputs = tf.random_normal(shape=[NUM_EXAMPLES])
    noise = tf.random_normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise
    return inputs, outputs


def plot_input_output(model, inputs, outputs):
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()

    print("损失:")
    print(loss(model(inputs), outputs).numpy())


def train_gradient(model, inputs, outputs, learning_rate):
    """
    求取梯度，更新W，b
    :param model:
    :param inputs:
    :param outputs:
    :param learning_rate:
    :return:
    """
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


def train_data(model, inputs, outputs, TRUE_W, TRUE_b):
    """
    训练
    :param model:
    :param inputs:
    :param outputs:
    :param TRUE_W:
    :param TRUE_b:
    :return:
    """
    Ws, bs = [], []
    epochs = range(30)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(model(inputs), outputs)
        train_gradient(model, inputs, outputs, learning_rate=0.1)
        print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
              (epoch, Ws[-1], bs[-1], current_loss))

    # Let's plot it all
    plt.plot(epochs, Ws, 'r',
             epochs, bs, 'b')
    plt.plot([TRUE_W] * len(epochs), 'r--',
             [TRUE_b] * len(epochs), 'b--')
    plt.legend(['W', 'b', 'true W', 'true_b'])
    plt.show()


if __name__ == '__main__':
    """
    1. 定义模型
    2. 定义损失函数
    3. 获取训练数据
    4. 使用优化器来求取权重和偏置
    """
    model = Model()
    inputs, outputs = obtain_data()
    plot_input_output(model, inputs, outputs)
    # TRUE_W和TRUE_b表示真实的W和b
    train_data(model, inputs, outputs, TRUE_W=3, TRUE_b=2)
