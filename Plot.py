import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding_2D_tsne(M, assign, label, J):
    # t-SNE可视化聚类结果
    print('Starting compute t-SNE Embedding...')
    print('分簇的数目: %d' % J)

    # 降维到2D用于绘图
    ts_2D = TSNE(n_components=2, perplexity=15, init='pca', random_state=0)
    res_2D = ts_2D.fit_transform(M)

    # 调用函数，绘制图像
    fig = plt.figure(1)
    plt.subplot(121)
    plt.scatter(res_2D[:, 0], res_2D[:, 1], c=label)
    plt.colorbar()

    plt.subplot(122)
    plt.scatter(res_2D[:, 0], res_2D[:, 1], c=assign)
    #
    # fig1 = plot_embedding_2D(res_2D,assign,J,'faces:t-SNE')
    plt.colorbar()

    return fig


def plot_embedding_2D(M, assign, label, J):

    print('Starting compute t-SNE Embedding...')
    print('分簇的数目: %d' % J)

    # 调用函数，绘制图像
    fig = plt.figure(1)
    plt.subplot(121)
    plt.scatter(M[:, 0], M[:, 1], c=label)
    plt.colorbar()

    plt.subplot(122)
    plt.scatter(M[:, 0], M[:, 1], c=assign)
    #
    # fig1 = plot_embedding_2D(res_2D,assign,J,'faces:t-SNE')
    plt.colorbar()

    return fig
