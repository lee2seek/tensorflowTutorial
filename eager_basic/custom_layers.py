#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : custom_layers.py
@software : PyCharm
@time : 2019/1/24 10:53

演示如何使用tf.keras中的layer，同时演示如何自定义自己的layer
如何继承Model建立自己的model
"""

import tensorflow as tf

tf.enable_eager_execution()


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        """
        init
        :param num_outputs: 输出单元
        """
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


class ResnetIdentityBlock(tf.keras.Model):
    """
    定义Model，继承tf.keras.Model
    """
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


if __name__ == '__main__':
    # 实例化自己定义的layer
    layer = MyDenseLayer(10)
    print("layer output:")
    print(layer(tf.zeros([10, 5])))
    print("trainable_variables:")
    print(layer.trainable_variables)

    # 实例化自己定义的model
    block = ResnetIdentityBlock(1, [1, 2, 3])
    print(block(tf.zeros([1, 2, 3, 3])))
    print([x.name for x in block.trainable_variables])