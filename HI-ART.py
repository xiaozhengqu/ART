import numpy as np
import DataLoad
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import Plot


def hi_art(M, label, rho, sigma=0.1, delta=0.01, beta=0.6, save_path_root=''):
    """
    @param M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    @param label: m*1，代表所属类别（从0开始）
    @param rho: 警戒参数(0-1)
    @param sigma: the percentage to enlarge or shrink vigilance region
    @param delta: 很小的数值，用于CM-ART策略中微微收缩边界
    @param beta:  has no significant impact on performance with a moderate value of [0.4,0.7]
    @param save_path_root: 保存结果的路径
    @return:
    """

    NAME = 'hi_art'
    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function;
    # to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    alpha = 0.01

    # rho needs carefully tune; used to shape the inter-cluster similarity;
    # rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    # rho = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    print(NAME + "算法开始")

    # 用第一个样本初始化第一个cluster

    # Wv存放cluster权重参数，row行col列
    # 如果后面cluster小于row个，则把多余的行去掉
    Wv = np.zeros((row, col))

    # J为cluster的个数
    J = 0

    # 记录每个cluster的内部点的数量，1行row列（最多row个cluster），如果cluster个数少于row，后面再进行删除多余的
    # 每当有一个cluster内多了一个数据点，就 +1
    L = np.zeros((1, row), dtype=np.int64)

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int64)

    # 警戒参数矩阵，1行row列，用于判断样本点是否满足cluster
    rho_0 = rho * np.ones((1, row))

    # 存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
    T_values = np.zeros((1, row)) - 2

    # 存储某样本对于各个cluster的匹配函数 M
    M_values = np.zeros((1, row)) - 1

    # first cluster
    print('Processing data sample 0')

    Wv[0, :] = M[0, :]  # 直接用第一个样本作为cluster参数
    J = 1
    L[0, J - 1] = 1
    Assign[0, 0] = J - 1

    # 计算其他样本
    for n in range(1, row):

        # print('Processing data sample %d' % n)

        T_winner = -1  # winner的T值
        winner = -1  # index of the winner cluster

        # compute the similarity with all clusters; find the best-matching cluster
        for j in range(0, J):

            # 对于每个input I（n），计算它和每个cluster（j）的匹配函数，算出 取小（min） 之后的一范数，用于后面计算匹配函数M和选择函数T
            Mj_numerator_V = np.sum(np.minimum(M[n, :], Wv[j, :]))

            # ---------------------------------------------------------
            # Template Matching（本部分计算匹配函数M，后面需要大于警戒参数）
            # 计算similarity（除以输入样本的特征值的和），用于跟警戒参数比较
            M_values[0, j] = Mj_numerator_V / np.sum(M[n, :])

            # ----------------------------------------------------------
            # Category Choice（本部分计算选择函数T，后面需要找T_values最大的）
            T_values[0, j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

            # ----------------------------------------------------------
            # 此部分目的是找到winner
            # 根据计算的1.2步，选取匹配函数M大于警戒参数，且如果选择函数T比当前记录的T更大，则更新最大值，同时更新winner
            # (如果多个相同大小的最大值，此代码选取最后面（新）的cluster)
            if M_values[0, j] >= rho_0[0, j] and T_values[0, j] >= T_winner:
                T_winner = T_values[0, j]
                winner = j

        # ---------------------------------------警戒参数自适应----------------------------------------------------------
        # 此处是HI-ART的核心，结合AM-ART和CM-ART，思想是：针对当前要进行聚类的样本I
        # 1. 对于所有类簇，选出 匹配函数M大于警戒参数rho的类簇，称为winner候选者集合
        # （如果没有winner候选者集合,等价于不存在winner，则依照AMR思想，对所有类簇对应的警戒参数rho进行减少 rho_new=（1-sigma）*rho，同时HI-ART结束）
        # 2. 计算这些候选者类簇中，对应T最大的类簇（即winner类簇），根据AMR思想，增加winner类簇的警戒参数 rho_new=（1+sigma）*rho
        # 3. 在候选者集合中，去除winner类簇，得到剩下的类簇（可称为失败者簇），根据CMR思想，对其rho进行调整：rho_new = M + delta
        # 4. 根据CMR思想，需要对那些T够大但因为M<rho而被限制共振的类簇，进行rho的调整，有利于下次共振。
        #    因此要在所有类簇中选出比当前winner的T还大于等于（即T>=T_winner）,且不属于winner候选者（即M<rho）的类簇。
        #    对这些类簇的rho进行减小 rho_new=（1-sigma）*rho

        # 1.选出 匹配函数M大于警戒参数rho的类簇，称为winner候选者集合
        M_compare = np.where(M_values[0, :] >= rho_0[0, :])  # 返回一个tuple，a[0]是个数组，里面每个元素是索引
        b = M_compare[0]
        # 选取选择函数T 大于等于 T（winner） 的cluster(如果winner不存在,则T_winner为-1，存在的类簇的T均大于0，不存在类簇的T为初始值-2，因此相当于选取全部已存在类簇)
        T_compare = np.where(T_values[0, :] >= T_winner)
        c = T_compare[0]

        # 如果winner候选者集合存在，则继续执行HI-ART(等价于winner存在，即winner>-1)
        if b.size > 0:
            # 2. 计算这些候选者类簇中，对应T最大的类簇（即winner类簇），根据AMR思想，增加winner类簇的警戒参数 rho_new=（1+sigma）*rho
            rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]

            # 3.在候选者集合中，去除winner类簇，得到剩下的类簇（可称为失败者簇），根据CMR思想，对其rho进行调整：rho_new = M + delta
            loser = np.delete(b, np.where(b == winner))

            for index in loser:
                rho_0[0, index] = M_values[0, index] + delta

            # 4.根据CMR思想，需要对那些T够大但因为M < rho而被限制共振的类簇，进行rho的调整，有利于下次共振。
            # 因此要在所有类簇中选出比当前winner的T还大于等于（即T>=T_winner）,且不属于winner候选者（即M<rho）的类簇
            # 对这些类簇的rho进行减小 rho_new=（1-sigma）*rho

            c = np.delete(c, np.where(c == winner))  # 去除获胜者

        # 去除获胜者之后，对T_compare中其余类簇的rho进行减小
        # （或者如果没有winner候选者,即不存在winner，则依照AMR思想，对所有类簇对应的警戒参数rho进行减少 同时HI-ART结束）
        rho_0[0, c] = (1 - sigma) * rho_0[0, c]

        # ---------------------------------------------------------------------------------------------------
        # Cluster assignment process
        if winner == -1:  # 没有cluster超过警戒参数
            # 创建新cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            L[0, J - 1] = 1
            Assign[0, n] = J - 1
        else:  # 如果有winner,进行cluster分配并且更新cluster权重参数
            # 更新cluster权重
            Wv[winner, :] = beta * np.minimum(Wv[winner, :], M[n, :]) + (1 - beta) * Wv[winner, :]
            # cluster分配
            L[0, winner] += 1
            Assign[0, n] = winner
            # rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]

    print("algorithm ends")
    # Clean indexing data(去除空的位置)
    Wv = Wv[0: J, :]
    L = L[:, 0: J]

    # -----------------------------------------------------------------------------------------------------------------------
    # 评估

    # 混淆矩阵
    number_of_class = int(max(label)) + 1
    confu_matrix = np.zeros((J, number_of_class))

    for i in range(0, row):
        confu_matrix[Assign[0, i], int(label[i])] += 1

    # 计算每个cluster中的支配者类及其大小
    # 每个cluster中个数最多的类的个数（因为个数最多，所以该cluster都被认为是属于该类）
    max_value = np.amax(confu_matrix, axis=1)
    # 每个cluster中最大类的索引
    max_index = np.argmax(confu_matrix, axis=1)
    # 每个真实类 含有的实例数
    size_of_classes = np.sum(confu_matrix, axis=0)

    # 计算准确率（即 分配成功的类 占 该cluster内实例个数 的比重）
    precision = max_value / L[0, :]

    # 计算召回率
    recall = np.zeros((J))
    for i in range(0, J):
        recall[i] = max_value[i] / size_of_classes[max_index[i]]

    # 簇内距离（欧式距离）
    intra_cluster_distance = np.zeros((J))
    for i in range(0, row):
        temp1 = np.sqrt(
            np.sum(np.square(Wv[Assign[0, i], 0:(col // 2)] - M[i, 0:(col // 2)])))
        temp2 = np.sqrt(
            np.sum(np.square(1 - Wv[Assign[0, i], (col // 2):] - M[i, 0:(col // 2)])))

        # compute average distance between bottom-left and upper-right points of the cluster and the input pattern
        intra_cluster_distance[Assign[0, i]] += (temp1 + temp2) / 2

    intra_cluster_distance = intra_cluster_distance[:] / L[0, :]

    # 簇间距离（欧式距离）
    inter_cluster_distance = np.zeros(((J * (J - 1)) // 2))
    len = 0
    for i in range(0, J):
        for j in range(i + 1, J):
            temp = np.square(Wv[i, :] - Wv[j, :])
            # compute the average distance between bottom-left and upper-right points of two clusters
            inter_cluster_distance[len] = (np.sqrt(np.sum(temp[0:(col // 2)])) + np.sqrt(np.sum(temp[(col // 2):]))) / 2
            len += 1

    # -----------------------------------------------------------------------------------------------
    # 画图以及输出
    M_ = M[:, 0:2]
    assign = Assign.flatten()
    assign = np.array(assign)

    # # t-SNE可视化聚类结果
    # print('Starting compute t-SNE Embedding...')
    # print('分簇的数目: %d' % J)
    # # 降维到2D用于绘图
    # ts_2D = TSNE(n_components=2, perplexity=15, init='pca', random_state=0)
    # res_2D = ts_2D.fit_transform(M)
    #
    # # 调用函数，绘制图像
    # plt.figure(1)
    # plt.subplot(121)
    # plt.scatter(res_2D[:, 0], res_2D[:, 1], c=label)
    # plt.colorbar()
    #
    # plt.subplot(122)
    # plt.scatter(res_2D[:, 0], res_2D[:, 1], c=assign)
    # #
    # # fig1 = plot_embedding_2D(res_2D,assign,J,'faces:t-SNE')
    # plt.colorbar()
    Plot.plot_embedding_2D(M, assign, label, J)
    plt.show(block=True)

    print('rho: %0.3f' % rho)
    print('beta: %0.3f' % beta)
    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('avg_precision: %0.3f' % np.mean(precision))
    print('avg_recall: %0.3f' % np.mean(recall))

    '''
    #3D
    ts_3D = TSNE(n_components=3,perplexity=20, random_state=0)
    res_3D = ts_3D.fit_transform(cluster_distribution)
    fig = plot_embedding_3D(res_3D, assign,'t-SNE')
    plt.show()
    '''

    # -----------------------------------------------------------------------------------------------------------------------
    # 保存结果

    # sio.savemat(
    #     save_path_root + str(number_of_class) + '_class_' + NAME + '_rho_' + str(rho) + '_beta_' + str(beta) + '.mat',
    #     {'precision': precision, 'recall': recall, 'cluster_size': L,
    #      'intra_cluster_distance': intra_cluster_distance, 'inter_cluster_distance': inter_cluster_distance,
    #      'Wv': Wv, 'confu_matrix': confu_matrix})

    return 0


if __name__ == '__main__':
    wine_data, wine_label = DataLoad.load_data_wine()
    hi_art(wine_data, wine_label, 0.2)
