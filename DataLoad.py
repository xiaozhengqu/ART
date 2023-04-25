import pandas as pd
import numpy as np


def load_data_wine():
    """
    @return: 返回打乱后的wine数据集的data部分（维度：178*13）和label部分（维度：178*1，且label从0开始)
    Wine Data Set原始数据集共有178个样本数、3种数据类别、每个样本的有13个属性。
    """

    # 读取wine数据集并转为numpy,并且shuffle
    wine = pd.read_csv(r'data/wine.data', header=None)
    wine = np.array(wine)
    np.random.shuffle(wine)

    # 分割数据集，提取出label（178）和data（178*13）
    wine_label = wine[:, 0] - 1
    wine_data = wine[:, 1:]
    data_num, feature_num = wine_data.shape

    # 数据归一化
    min_num = np.min(wine_data, axis=0)
    max_num = np.max(wine_data, axis=0)
    data_range = max_num - min_num
    # 按列进行归一化操作
    for col in range(0, feature_num):
        wine_data[:, col] = (wine_data[:, col] - min_num[col]) / data_range[col]

    return wine_data, wine_label


def load_data_wine_nums(repeat_nums, shuffle=False):
    """
    @return: 返回打乱后的wine数据集的data部分（维度：178*13）和label部分（维度：178*1，且label从0开始),同时进行多次数据拼接
    Wine Data Set原始数据集共有178个样本数、3种数据类别、每个样本的有13个属性。
    """
    # 读取wine数据集并转为numpy,并且shuffle
    wine = pd.read_csv(r'data/wine.data', header=None)
    wine = np.array(wine)

    np.random.shuffle(wine)  # 仅打乱最初的一次
    result_wine = wine

    if shuffle:
        # 多次拼接数据
        for i in range(repeat_nums - 1):
            np.random.shuffle(wine)
            result_wine = np.concatenate((result_wine, wine), 0)

    else:
        for i in range(repeat_nums - 1):
            result_wine = np.concatenate((result_wine, wine), 0)

    # 分割数据集，提取出label（178）和data（178*13）
    wine_label = result_wine[:, 0] - 1
    wine_data = result_wine[:, 1:]

    data_num, feature_num = wine_data.shape

    # 数据归一化
    min_num = np.min(wine_data, axis=0)
    max_num = np.max(wine_data, axis=0)
    data_range = max_num - min_num
    # 按列进行归一化操作
    for col in range(0, feature_num):
        wine_data[:, col] = (wine_data[:, col] - min_num[col]) / data_range[col]

    return wine_data, wine_label


if __name__ == '__main__':
    load_data_wine()
