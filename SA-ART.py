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

    # 补码
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

    # 自适应学习率theta
    learning_rate_theta = np.zeros((1, col))

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
    feature_mean[0, :] = M[0, :]  # 由于是第一个样本，则特征均值即为权重 / 且由于是第一个样本，则特征方差为0
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

        T_max = -1  # 所有匹配函数M大于对应的rho的类簇中，记录其中最大的选择函数值
        winner = -1  # index of the winner cluster

        temp_a[0, :] = M[n, :]  # In

        # compute the similarity with all clusters; find the best-matching cluster
        # 对所有现有的cluster循环，寻找最匹配的类簇
        for j in range(0, J):

            temp_b[0, :] = Wv[j, :]  # Wj

            # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 流程是：
            # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数
            # 从而得到分子，用于后面计算T，M

            # 先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘显著性得分）
            intersec[0, :] = np.minimum(temp_a, temp_b)
            intersec_index = np.where(intersec[0, :] > 0)

            # # 得到此时 特征频数 大于0 的索引（用于后面算显著性得分）
            # salience_index = np.where(feature_salience_count[j, :] > 0)

            # 对于intersec_index代表的这些特征，计算显著性权重s。
            # 由公式可知，需要先算出 频率(频数除以类簇样本数) 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
            salience_weight_presence = feature_salience_count[j, :] / L[0, j]  # 频率(频数除以类簇样本数)
            salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))  # e的负标准差(方差开根号)次方
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

            # 判断是否满足rho，同时更新T_max
            # 对于T_max还未更新（即为-1时），and右边一直满足，只要左边满足则进入if，之后便可自然更新T_max；
            # 若左边一直不满足，则T_max无法更新（为-1），则在下面一段a的计算中，所有现存类簇的T_value都会大于-1（暂时不存在的类簇，该值为-2）
            # 从而所有现存类簇都会被a选中,从而在后面通过AMR减小rho。也符合我们的算法思想。
            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        # AMR思想
        # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
        a = np.where(T_values[0, :] >= T_max)
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
            feature_salience_count[J - 1, np.where(M[n, :] > 0)] += 1  # 更新频数
            feature_mean[J - 1, :] = M[n, :]  # 新类簇标准差为0，无需更新；只更新均值
            L[0, J - 1] = 1
            Assign[0, n] = J - 1
        else:
            # if winner is found, do cluster assignment and update

            # 1.先更新cluster权重向量w
            # 需要先计算自适应学习率theta，根据方差>0 和 =0 进行区分，用不同的公式计算theta
            zero_std_index = np.where(feature_std2[winner, :] == 0)
            non_zero_std_index = np.where(feature_std2[winner, :] > 0)
            # 计算方差为0的theta
            up1 = np.square(M[n, :][zero_std_index] - feature_mean[winner, :][zero_std_index]) * 9
            down1 = np.square(np.minimum(feature_mean[winner, :][zero_std_index] + 0.01,
                                         1 - feature_mean[winner, :][zero_std_index])) * 2
            learning_rate_theta[0, zero_std_index] = np.exp(-(up1 / down1))
            # 计算方差不为0的theta(std2已经是方差，不用平方)
            up2 = np.square(M[n, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index])
            down2 = 2 * feature_std2[winner, :][non_zero_std_index]
            learning_rate_theta[0, non_zero_std_index] = np.exp(-(up2 / down2))
            # 更新cluster权重向量w
            Wv[winner, :] = np.minimum(M[n, :], feature_mean[winner, :]) * learning_rate_theta[0, :] + \
                            Wv[winner, :] * (1 - learning_rate_theta[0, :])

            # 2.在线更新频数（频率不用更新，因为在遍历类簇时会根据频数重新计算）
            salient_index = np.where(M[n, :] > 0)
            feature_salience_count[winner, salient_index] += 1

            # 3.在线更新均值mean和方差std2
            # 暂存旧的均值mean，后面用
            old_mean[0, :] = feature_mean[winner, :]
            # 在线更新均值mean
            feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[n, :]) / (L[0, winner] + 1)
            # 在线更新方差std2
            feature_M[winner, :] = feature_M[winner, :] + (M[n, :] - old_mean[0, :]) * (
                    M[n, :] - feature_mean[winner, :])
            feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

            # 更新数目和样本分配（显著性参数不用更新，因为在遍历类簇时会根据 输入样本I 和 类簇权重w 的非零情况，通过 频率和方差 重新计算）
            L[0, winner] += 1
            Assign[0, n] = winner

    print("algorithm ends")
    # Clean indexing data
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
    # save results
    #
    # sio.savemat(save_path_root + str(number_of_class) + '_class_' + NAME + '_rho_' + str(rho) + '.mat',
    #             {'precision': precision, 'recall': recall, 'cluster_size': L,
    #              'intra_cluster_distance': intra_cluster_distance, 'inter_cluster_distance': inter_cluster_distance,
    #              'Wv': Wv, 'confu_matrix': confu_matrix})

    return 0


if __name__ == '__main__':
    wine_data, wine_label = DataLoad.load_data_wine_nums(10, True)
    sa_art(wine_data, wine_label, 0.2)
