#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : auto_differentiation.py
@software : PyCharm
@time : 2019/1/22 20:27

tensorflow 中自动求取微分
https://www.tensorflow.org/tutorials/eager/automatic_differentiation?hl=zh-cn
"""

import tensorflow as tf

tf.enable_eager_execution()


def run_example1():
    x = tf.ones((2, 2))

    # 这里记录了z的计算过程，然后在下方的代码计算梯度，从而计算出值为8
    with tf.GradientTape() as t:
        t.watch(x)
        # x 里面元素全部加起来
        y = tf.reduce_sum(x)
        print('y:', y)
        z = tf.multiply(y, y)
        print('z:', z)

    # Derivative of z with respect to the original input tensor x
    dz_dx = t.gradient(z, x)
    print('dz_dx:', dz_dx)
    for i in [0, 1]:
        for j in [0, 1]:
            # numpy：Returns a numpy array or a scalar with the same contents as the Tensor
            assert dz_dx[i][j].numpy() == 8.0


def run_example2():
    x = tf.ones((2, 2))

    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x)
        z = tf.multiply(y, y)

    # Use the tape to compute the derivative of z with respect to the
    # intermediate value y.
    dz_dy = t.gradient(z, y)
    assert dz_dy.numpy() == 8.0


def run_example3():
    """
    演示如何多次计算梯度。因为如果不设置persistent参数为True，调用gradient后，tape资源会释放
    :return:
    """
    x = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        y = x * x
        z = y * y
    dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
    dy_dx = t.gradient(y, x)  # 6.0
    print('dz_dx:', dz_dx)
    print('dy_dx:', dy_dx)
    del t  # Drop the reference to the tape


def f(x, y):
    output = 1.0
    for i in range(y):
        if 1 < i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
        return t.gradient(out, x)


def record_control_flow():
    # convert_to_tensor 将2.0转为tensor
    x = tf.convert_to_tensor(2.0)
    print('x:', x)
    assert grad(x, 6).numpy() == 12.0
    assert grad(x, 5).numpy() == 12.0
    assert grad(x, 4).numpy() == 4.0


def hight_order_gradient():
    """
    演示如何计算高阶梯度
    :return:
    """
    x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

    with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            y = x * x * x
        # Compute the gradient inside the 't' context manager
        # which means the gradient computation is differentiable as well.
        dy_dx = t2.gradient(y, x)
    d2y_dx2 = t.gradient(dy_dx, x)

    assert dy_dx.numpy() == 3.0
    assert d2y_dx2.numpy() == 6.0


if __name__ == '__main__':
    run_example1()
    run_example2()
    run_example3()
    record_control_flow()
    hight_order_gradient()