#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : eager_basic.py
@software : PyCharm
@time : 2019/1/21 19:26

展示eager execution的一些基本用法
https://www.tensorflow.org/tutorials/eager/eager_basics?hl=zh-cn
"""
import os

# 取消警告信息的显示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import timeit
import tempfile
import numpy as np
import tensorflow as tf

# 启用eager模式
tf.enable_eager_execution()


def tensors_ops():
    print(tf.add(1, 2))
    print(tf.add([1, 2], [3, 4]))
    print(tf.square(5))
    print(tf.reduce_sum([1, 2, 3]))
    print(tf.encode_base64("hello world"))

    # Operator overloading is also supported
    print(tf.square(2) + tf.square(3))

    x = tf.matmul([[1]], [[2, 3]])
    print(x.shape)
    print(x.dtype)


def numpy_vs_tensor():
    ndarray = np.ones([3, 3])
    print("tensorflow 操作自动将numpy类型的数组转为tensor")
    tensor = tf.multiply(ndarray, 42)
    print(tensor)

    print("numpy操作自动将tensor转为numpy类型")
    print(np.add(tensor, 1))

    print("numpy()方法明确将tensor转为numpy array")
    print(tensor.numpy())


def gpu_acceleration():
    x = tf.random_uniform([3, 3])
    print("GPU是否可用:")
    print(tf.test.is_gpu_available())

    print("tensor 的device属性:", x.device)
    print("tensor 是否在 GPU#0上:")
    print(x.device.endswith("GPU:0"))


def time_matmul(x):
    t = timeit.timeit(lambda: tf.matmul(x, x), number=1)
    print("时间消耗:", t)
    # %timeit tf.matmul(x, x)


def device_placement():
    print("CPU上执行:")
    with tf.device("CPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("CPU:0")
        time_matmul(x)

    # 如果GPU可以用，则在GPU上执行
    if tf.test.is_gpu_available():
        print("在GPU上执行")
        with tf.device("GPU:0"):
            x = tf.random_uniform([1000, 1000])
            assert x.device.endswith("GPU:0")
            time_matmul(x)


def dataset_ops():
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

    # 创建临时文件
    _, filename = tempfile.mkstemp()
    print(_)
    print("文件名:", filename)

    with open(filename, 'w') as f:
        f.write("""Line 1
    Line 2
    Line 3
      """)

    ds_file = tf.data.TextLineDataset(filename)
    print("ds_file type：", type(ds_file))
    print("ds_file:", ds_file)
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
    ds_file = ds_file.batch(2)
    print("ds_file:", ds_file)

    print('ds_tensors中的元素:')
    for x in ds_tensors:
        print(x)

    print('ds_file中的元素:')
    for x in ds_file:
        print(x)


if __name__ == '__main__':
    tensors_ops()
    numpy_vs_tensor()
    gpu_acceleration()
    device_placement()
    dataset_ops()
