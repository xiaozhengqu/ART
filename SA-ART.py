import scipy.io as sio
import numpy as np
import operator
import DataLoad


def sa_art(M, label, rho, save_path_root=''):
    '''
    % M: numpy arrary; m*n feature matrix; m is number of objects and n is number of visual features
    %rho: the vigilance parameter
    %save_path_root: path to save clustering results for further analysis
    '''

    NAME = 'sa_am_art'

    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters

    # no need to tune; used in choice function; to avoid too small cluster weights
    # (resulted by the learning method of ART; should be addressed sometime); give priority to choosing denser clusters
    alpha = 0.01

    sigma = 0.1  # the percentage to enlarge or shrink vigilance region; hcc makes sigma less sensitive

    lam = 0.9
    max_iteration = 50

    # rho needs carefully tune; used to shape the inter-cluster similarity; rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    # rho = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    print(NAME + "algorithm starts")

    # create initial cluster with the first data sample
    # initialize cluster parameters

    # 类簇的权重矩阵，row:cluster,col:features
    Wv = np.zeros((row, col))

    # 记录各个类簇的每个特征的 频数（不是频率），row: clusters, col:features
    # 认为一个特征>0则该特征频数+1
    feature_salience_count = np.zeros((row, col))

    # 记录各个类簇的每个特征的 均值，row: clusters, col:features
    feature_mean = np.zeros((row, col))

    # 计算 方差 过程中需要的中间变量（初始化为0）
    feature_M = np.zeros((row, col))  # intermediate variable for computing feature_std2
    # 记录 各个类簇每个特征的方差，row: clusters, col:features
    feature_std2 = np.zeros((row, col))  # record variances, being used togather with feature_mean

    # 用于feature_mean和feature_std2（均值和方差）的在线更新
    old_mean = np.zeros((1, col))  # facilitate the online update of feature_mean and feature_std2

    # 显著性权重s
    salience_weight_prob = np.zeros((1, col))

    J = 0  # number of clusters

    # 记录每个cluster的内部点的数量，1行row列（最多row个cluster），如果cluster个数少于row，后面再进行删除多余的
    # 每当有一个cluster内多了一个数据点，就 +1
    L = np.zeros((1, row), dtype=np.int)

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int)

    # 用于AMR策略调整警戒参数rho，一个元素代表一个类簇的rho
    rho_0 = rho * np.ones((1, row))
    # AMR策略中会用：存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
    T_values = np.zeros((1, row)) - 2

    # ----------first cluster,first sample-------------
    print('Iteration 1: Processing data sample 1')
    Wv[0, :] = M[0, :]
    feature_salience_count[0, np.where(M[0, :] > 0)] += 1  # 凡是特征大于0，则频数 0+1 = 1
    feature_mean[0, :] = M[0, :]  # 由于是第一个样本，则特征均值即为权重
    J = J + 1
    L[0, J - 1] = 1
    Assign[0, 0] = J - 1  # 存放索引，注意类簇索引从0开始，所以J-1

    # intermediate variables used in clustering - defined early here
    # temp_a 用于暂存当前在处理的样本向量：In
    # temp_b 用于暂存当前在处理的cluster的特征向量：Wj
    temp_a = np.zeros((1, col))
    temp_b = np.zeros((1, col))
    intersec = np.zeros((1, col))  # get salient features

    # -----------从第二个样本开始，处理之后的样本---------
    for n in range(1, row):

        if n % 100 == 0:
            print('Processing data sample %d' % n)

        T_max = -1  # the maximun choice value
        winner = -1  # index of the winner cluster

        temp_a[0, :] = M[n, :]  # In

        # compute the similarity with all clusters; find the best-matching cluster
        # 对所有现有的cluster循环，寻找最匹配的类簇
        for j in range(0, J):

            temp_b[0, :] = Wv[j, :]

            # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 流程是：
            # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数

            # 先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘显著性得分）
            intersec[0, :] = np.minimum(temp_a, temp_b)
            intersec_index = np.where(intersec[0, :] > 0)
            # 得到此时 特征频数 大于0 的索引（用于后面算显著性得分）
            salience_index = np.where(feature_salience_count[j, :] > 0)

            # 对于intersec_index代表的这些特征，计算显著性权重s。
            # 由公式可知，需要先算出 频率 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
            salience_weight_presence = feature_salience_count[j, :] / L[0, j]
            salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
            # 对他们归一化
            normalized_salience_weight_presence = salience_weight_presence / np.sum(salience_weight_presence)
            normalized_salience_weight_std = salience_weight_std / np.sum(salience_weight_std)
            # 取出其中的intersec_index代表的那些特征
            normalized_salience_weight_presence_intersec = normalized_salience_weight_presence[intersec_index]
            normalized_salience_weight_std_intersec = normalized_salience_weight_std[intersec_index]
            # 计算显著性权重s
            normalized_salience_weight = lam * normalized_salience_weight_presence_intersec + \
                                         (1 - lam) * normalized_salience_weight_std_intersec

            # 计算分子
            temp = np.sum(intersec[0, intersec_index] * normalized_salience_weight)
            # 计算匹配函数M
            Mj_V = temp / np.sum(temp_a[0, intersec_index] * normalized_salience_weight)
            # 计算选择函数T
            T_values[0, j] = temp / (alpha + np.sum(temp_b[0, intersec_index] * normalized_salience_weight))

            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        a = np.where(T_values[0, :] >= T_max)  # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
        b = a[0]
        # 如果有获胜者
        if winner > -1:
            # 对获胜者的rho进行增加
            rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]

            b = np.delete(b, np.where(b == winner)[0])  # 去除获胜者
        # 去除获胜者之后（或者没有获胜者），对剩下的这些cluster的rho进行减少
        rho_0[0, b] = (1 - sigma) * rho_0[0, b]

        # Cluster assignment process
        if winner == -1:
            # indicates no cluster passes the vigilance parameter - the rho
            # create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            feature_salience_count[J - 1, np.where(M[n, :] > 0)] += 1
            feature_mean[J - 1, :] = M[n, :]  # 新类簇标准差为0，无需更新；只更新均值
            L[0, J - 1] = 1
            Assign[0, n] = J - 1
        else:
            # if winner is found, do cluster assignment and update cluster weights
            # cluster assignment; we do assignments first to enable the learning of weights from updated statistics
            # 暂存旧的均值mean，后面用
            old_mean[0, :] = feature_mean[winner, :]
            # 在线更新均值mean
            feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[n, :]) / (L[0, winner] + 1)
            # 在线更新方差std2
            feature_M[winner, :] = feature_M[winner, :] + (M[n, :] - old_mean[0, :]) * (
                    M[n, :] - feature_mean[winner, :])
            feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

            L[0, winner] += 1
            Assign[0, n] = winner

            # update salient features 更新显著性权重
            feature_min_values = np.minimum(M[n, :], Wv[winner, :])
            salient_index = np.where(np.maximum(M[n, :], Wv[winner, :]) > 0)
            feature_salience_count[winner, salient_index] += 1

            # 更新cluster权重向量
            salience_weight_presence = feature_salience_count[winner, :] / L[0, winner]
            salience_weight_std = np.exp(-np.sqrt(feature_std2[winner, :]))

            zero_std_index = np.where(feature_std2[winner, :] == 0)
            non_zero_std_index = np.where(feature_std2[winner, :] > 0)
            salience_weight_prob = np.zeros((1, col))
            salience_weight_prob[0, zero_std_index] = np.exp(
                -(np.square(M[n, :][zero_std_index] - feature_mean[winner, :][zero_std_index])))
            salience_weight_prob[0, non_zero_std_index] = np.exp(-(np.square(
                M[n, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index]) / (
                                                                           2 * feature_std2[winner, :][
                                                                       non_zero_std_index])))

            fused_weight = (salience_weight_presence + salience_weight_std + salience_weight_prob[0, :]) / 3

            Wv[winner, :] = feature_mean[winner, :] * fused_weight + feature_min_values[:] * (1 - fused_weight)

    print("algorithm ends")
    # Clean indexing data
    Wv = Wv[0: J, :]
    L = L[:, 0: J]

    # -----------------------------------------------------------------------------------------------------------------------
    # performance calculation

    # confusion-like matrix
    number_of_class = int(max(label)) + 1
    confu_matrix = np.zeros((J, number_of_class))

    for i in range(0, row):
        confu_matrix[Assign[0, i], int(label[i])] += 1

    # compute dominator class and its size in each cluster
    max_value = np.amax(confu_matrix, axis=1)
    max_index = np.argmax(confu_matrix, axis=1)
    size_of_classes = np.sum(confu_matrix, axis=0)

    # compute precision, recall
    precision = max_value / L[0, :]

    recall = np.zeros((J))
    for i in range(0, J):
        recall[i] = max_value[i] / size_of_classes[max_index[i]]

    # intra_cluster distance - Euclidean
    intra_cluster_distance = np.zeros((J))
    for i in range(0, row):
        intra_cluster_distance[Assign[0, i]] += np.sqrt(np.sum(np.square(Wv[Assign[0, i], :] - M[i, :])))

    intra_cluster_distance = intra_cluster_distance[:] / L[0, :]

    # inter_cluster distance - Euclidean
    inter_cluster_distance = np.zeros(((J * (J - 1)) // 2))
    len = 0
    for i in range(0, J):
        for j in range(i + 1, J):
            inter_cluster_distance[len] = np.sqrt(np.sum(np.square(Wv[i, :] - Wv[j, :])))
            len += 1

    # -----------------------------------------------------------------------------------------------------------------------
    # save results
    #
    # sio.savemat(save_path_root + str(number_of_class) + '_class_' + NAME + '_rho_' + str(rho) + '.mat',
    #             {'precision': precision, 'recall': recall, 'cluster_size': L,
    #              'intra_cluster_distance': intra_cluster_distance, 'inter_cluster_distance': inter_cluster_distance,
    #              'Wv': Wv, 'confu_matrix': confu_matrix})

    return 0


if __name__ == '__main__':
    wine_data, wine_label = DataLoad.load_data_wine()
    sa_art(wine_data, wine_label, 0.2)
