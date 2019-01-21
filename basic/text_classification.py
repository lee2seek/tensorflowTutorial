#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io
采用深度学习进行文本分类，参考：https://www.tensorflow.org/tutorials/keras/basic_text_classification?hl=zh-cn
使用keras包
@version : 1.0
@file : text_classification.py
@software : PyCharm
@time : 2019/1/11 15:28

"""
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.preprocessing.sequence import _remove_long_seq

import numpy as np
import json

print(tf.__version__)

imdb = keras.datasets.imdb


def read_imdb_data(path='imdb.npz',
                   num_words=None,
                   skip_top=0,
                   maxlen=None,
                   seed=113,
                   start_char=1,
                   oov_char=2,
                   index_from=3,
                   **kwargs):
    """
    仿照keras.datasets.imdb.load_data()函数加载已经存在的数据
    :return:
    """
    # imdb = keras.datasets.imdb
    # num_words 参数保留训练数据中出现频次在前 10000 位的字词
    # (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']
    # 分别对训练集和测试集进行混淆
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def explore_data(data_x, data_y):
    """
    探索数据
    :param data_x:
    :param data_y:
    :return:
    """
    print("训练数据量：{},标签：{}".format(len(data_x), len(data_y)))
    print("第一条影评:" + str(data_x[0]))
    print("前两条影评长度分别为{},{}".format(len(data_x[0]), len(data_x[1])))

    # 单词到数字的映射
    word_index_path = "./data/imdb/imdb_word_index.json"
    word_index = get_word_index(path=word_index_path)
    print("word_index 数量:" + str(len(word_index)))
    word_index = {k: (v + 3) for k, v in word_index.items()}
    # word_index中字词的最小编号是1
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    text_0 = decode_review(reverse_word_index, data_x[0])
    print("第一条文本：{}".format(text_0))
    return word_index, reverse_word_index


def get_word_index(path):
    """
    仿照keras.datasets.imdb.get_word_index()函数，获取单词和数字之间的映射
    :param path:
    :return:
    """
    with open(path) as f:
        return json.load(f)


def decode_review(reverse_word_index, text):
    """
    根据单词数字的映射关系将数字解码为对应的文本
    :param reverse_word_index:
    :param text:
    :return:
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def prepare_data(train_data, test_data, word_index):
    """
    准备数据
    将文本补充为同样长度的
    :param train_data:
    :param test_data:
    :param word_index:
    :return:
    """
    # 填充序列，如果超过256，则在序列后面添加word_index['<PAD>']直到达到256
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
    print("after pad process, length of train_data:" + str(len(train_data[0])))
    print("after pad process, length of train_data:" + str(len(train_data[1])))
    print("the first data:{}".format(train_data[0]))
    return train_data, test_data


def build_model():
    """
    构建模型
    :return:
    """
    # 单词总数大小
    vocab_size = 10000
    # 经过embedding之后的维度，官网例子是16
    embed_size = 16
    model = keras.Sequential()
    # Embedding 将单词进行降维处理，生成词向量。16是输出的词向量的维度，vocab_size是输入序列的维度
    # Embedding输出的维度是(batch, 10000, 16)
    model.add(keras.layers.Embedding(vocab_size, embed_size))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(embed_size, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, train_x, train_y, val_x, val_y, show_process=True):
    """
    训练模型
    :param model:
    :param train_x:
    :param train_y:
    :param val_x:
    :param val_y:
    :param show_process:是否显示训练过程的图像
    :return:
    """
    history = model.fit(train_x, train_y, epochs=40, batch_size=512,
                        validation_data=(val_x, val_y), verbose=1)

    # 查看训练过程中指标变化情况
    history_dict = history.history
    print("训练过程中的指标:" + str(history_dict.keys()))

    if show_process:
        show_train_process(history)

    return model


def show_train_process(history):
    """
    显示训练过程中准确率和损失的变化
    :param history:
    :return:
    """
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 生成第二张图
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("training and validation acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


def evaluate_model(model, test_x, test_y):
    results = model.evaluate(test_x, test_y)
    print("评估结果:")
    print(results)


def split_train_to_valid(train_x, train_y):
    """
    从训练集中获取验证集
    :param train_x:
    :param train_y:
    :return: 返回训练集和验证集
    """
    x_val = train_x[:10000]
    partial_x_train = train_x[10000:]

    y_val = train_y[:10000]
    partial_y_train = train_y[10000:]
    return (partial_x_train, partial_y_train), (x_val, y_val)


if __name__ == '__main__':
    """
    整个程序的流程如下：
    1. 下载imdb数据集
    2. 探索数据
        - 将整数转换为字词
    3. 准备数据
    4. 构建模型
        - 隐藏单元
        - 损失函数和优化器
    5. 创建验证集
    6. 训练模型
    7. 评估模型
    8. 创建准确率和损失随时间变化的图
    """
    # 注意这里数据处理的过程
    imdb_path = "./data/imdb/imdb.npz"
    # num_words指定了训练集中出现次数排在前10000次的单词才会入选，其他字词会被舍弃
    (train_x, train_y), (test_x, test_y) = read_imdb_data(path=imdb_path, num_words=10000)
    word_index, reverse_word_index = explore_data(train_x, train_y)
    train_x, test_x = prepare_data(train_x, test_x, word_index)
    (train_x, train_y), (val_x, val_y) = split_train_to_valid(train_x, train_y)
    model = build_model()
    model = train_model(model, train_x, train_y, val_x, val_y, show_process=True)
    evaluate_model(model, test_x, test_y)
