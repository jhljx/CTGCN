import numpy as np
import pandas as pd
import os, datetime
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from RWTGCN.utils import check_and_make_path


class AnomalyDetection(object):
    def __init__(self):
        return

    def anomaly_generation(self, ini_graph_percent, anomaly_percent, data, n, m):
        """ generate anomaly
        split the whole graph into training network which includes parts of the
        whole graph edges(with ini_graph_percent) and testing edges that includes
        a ratio of manually injected anomaly edges, here anomaly edges mean that
        they are not shown in previous graph;
         input: ini_graph_percent: percentage of edges in the whole graph will be
                                    sampled in the intitial graph for embedding
                                    learning
                anomaly_percent: percentage of edges in testing edges pool to be
                                  manually injected anomaly edges(previous not
                                  shown in the whole graph)
                data: whole graph matrix in sparse form, each row (nodeID,
                      nodeID) is one edge of the graph
                n:  number of total nodes of the whole graph
                m:  number of edges in the whole graph
         output: synthetic_test: the testing edges with injected abnormal edges,
                                 each row is one edge (nodeID, nodeID, label),
                                 label==0 means the edge is normal one, label ==1
                                 means the edge is abnormal;
                 train_mat: the training network with square matrix format, the training
                            network edges for initial model training;
                 train:  the sparse format of the training network, each row
                            (nodeID, nodeID)
        """
        np.random.seed(1)
        print('[#s] generating anomalous dataset...\n', datetime.datetime.now())
        print('[#s] initial network edge percent: #.1f##, anomaly percent: #.1f##.\n', datetime.datetime.now(),
              ini_graph_percent * 100, anomaly_percent * 100)

        train_num = int(np.floor(ini_graph_percent * m))

        # select part of edges as in the training set
        train = data[0:train_num, :]

        # select the other edges as the testing set
        test = data[train_num:, :]

        # data to adjacency_matrix
        adjacency_matrix = self.edgeList2Adj(data)

        # clustering nodes to clusters using spectral clustering
        kk = 3  # 10#42#42

        # mark
        # sc = SpectralClustering(kk, affinity='precomputed', n_init=100, assign_labels='discretize')
        sc = MiniBatchKMeans(n_clusters=kk)
        labels = sc.fit_predict(adjacency_matrix)

        # generate fake edges that are not exist in the whole graph, treat them as
        # anamalies
        idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)), axis=1)
        idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)), axis=1)
        generate_edges = np.concatenate((idx_1, idx_2), axis=1)

        ####### genertate abnormal edges ####
        fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

        fake_edges = self.processEdges(fake_edges, data)

        # anomaly_num = 12  # int(np.floor(anomaly_percent * np.size(test, 0)))
        anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
        anomalies = fake_edges[0:anomaly_num, :]

        idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
        # randsample: sample without replacement
        # it's different from datasample!

        anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

        # anomaly_pos = np.random.choice(100, anomaly_num, replace=False) + 200
        idx_test[anomaly_pos] = 1
        synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)

        idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
        idx_normal = np.nonzero(idx_test.squeeze() == 0)

        synthetic_test[idx_anomalies, 0:2] = anomalies
        synthetic_test[idx_normal, 0:2] = test

        train_mat = sp.csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])),
                                  shape=(n, n))
        # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
        train_mat = train_mat + train_mat.transpose()

        return synthetic_test, train_mat, train


    def processEdges(self, fake_edges, data):
        """
        remove self-loops and duplicates and order edge
        :param fake_edges: generated edge list
        :param data: orginal edge list
        :return: list of edges
        """
        idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

        tmp = fake_edges[idx_fake]
        tmp[:, [0, 1]] = tmp[:, [1, 0]]

        fake_edges[idx_fake] = tmp

        idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)

        fake_edges = fake_edges[idx_remove_dups]
        a = fake_edges.tolist()
        b = data.tolist()
        c = []

        for i in a:
            if i not in b:
                c.append(i)
        fake_edges = np.array(c)
        return fake_edges


    def edgeList2Adj(self, data):
        """
        converting edge list to graph adjacency matrix
        :param data: edge list
        :return: adjacency matrix which is symmetric
        """

        data = tuple(map(tuple, data))

        n = max(max(user, item) for user, item in data)  # Get size of matrix
        # print(n)
        matrix = np.zeros((n, n))
        for user, item in data:
            matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
            matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
        return matrix


    def dynamic_anomaly_generation(self, data_path, sample_path, init_percent=0.8, anomaly_rate=0.2):
        check_and_make_path(sample_path)
        # 字母前缀个数(要转成数字才能进行array操作，把前缀删掉)
        delete_prefix_num = 1
        f_list = os.listdir(data_path)
        length = len(f_list)
        for i, f_name in enumerate(f_list):
            print(str(length - i) + " file(s) left")
            dataframe = pd.read_csv(os.path.join(data_path, f_name), sep='\t')
            nodes_set = pd.concat([dataframe['from_id'], dataframe['to_id']], axis=0).drop_duplicates()
            if len(nodes_set) < 10:
                continue
            full_node_list = nodes_set.values.tolist()
            nodes_set_only_num = np.array(nodes_set.map(lambda x: x[delete_prefix_num:])).astype(int)
            sort = np.argsort(nodes_set_only_num)
            dataframe['from_id'] = dataframe['from_id'].map(lambda x: x[delete_prefix_num:]).astype(int)
            dataframe['to_id'] = dataframe['to_id'].map(lambda x: x[delete_prefix_num:]).astype(int)
            dataframe['from_id'] = sort[np.searchsorted(nodes_set_only_num, dataframe['from_id'], sorter=sort)]
            dataframe['to_id'] = sort[np.searchsorted(nodes_set_only_num, dataframe['to_id'], sorter=sort)]

            edges = np.array(dataframe[["from_id", "to_id"]])
            # edges = np.loadtxt(data_path, dtype=int, comments='%')
            vertices = np.unique(edges)
            m = len(edges)
            n = len(vertices)
            synthetic_test, train_mat, train = self.anomaly_generation(init_percent, anomaly_rate, edges, n, m)

            df_train = pd.DataFrame(data=train, columns=['from_id', 'to_id', "label"])
            df_train['from_id'] = df_train["from_id"].map(lambda x: full_node_list[x])
            df_train['to_id'] = df_train["to_id"].map(lambda x: full_node_list[x])
            df_train.to_csv(os.path.join(sample_path, os.path.splitext(f_name)[0] + '_train.csv'))

            df_test = pd.DataFrame(data=synthetic_test, columns=['from_id', 'to_id', "label"])
            df_test['from_id'] = df_test["from_id"].map(lambda x: full_node_list[x])
            df_test['to_id'] = df_test["to_id"].map(lambda x: full_node_list[x])
            df_test.to_csv(os.path.join(sample_path, os.path.splitext(f_name)[0] + '_test.csv'))


    def anomaly_detection(self, embedding, train, synthetic_test, k, encoding_method='Hadamard'):
        """
        function anomaly_detection_stream(embedding, train, synthetic_test, k, alfa, n0, c0)
        #  the function generate codes of edges by combining embeddings of two
        #  nodes, and then using the testing codes of edges for anomaly detection
        #  Input: embedding: embeddings of each node; train: training edges; synthetic_test: testing edges with anomlies;
                    k: number of clusters
        #  return scores: The anomaly severity ranking, the top-ranked are the most likely anomlies
        #   auc: AUC score
        #   n:   number of nodes in each cluster
        #   c:   cluster centroids,
        #   res: id of nodes if their distance to nearest centroid is larger than that in the training set
        #   ab_score: anomaly score for the whole snapshot, just the sum of distances to their nearest centroids
        """
        print('[#s] edge encoding...\n', datetime.datetime.now())
        src = embedding[train[:, 0] - 1, :]
        dst = embedding[train[:, 1] - 1, :]
        test_src = embedding[synthetic_test[:, 0] - 1, :]
        test_dst = embedding[synthetic_test[:, 1] - 1, :]

        # the edge encoding
        # refer node2vec paper for details
        codes = np.multiply(src, dst)
        test_codes = np.multiply(test_src, test_dst)
        if encoding_method == 'Average':
            codes = (src + dst) / 2
            test_codes = (test_src + test_dst) / 2
        elif encoding_method == 'Hadamard':
            codes = np.multiply(src, dst)
            test_codes = np.multiply(test_src, test_dst)
        elif encoding_method == 'WeightedL1':
            codes = abs(src - dst)
            test_codes = abs(test_src - test_dst)
        elif encoding_method == 'WeightedL2':
            codes = (src - dst) ** 2
            test_codes = (test_src - test_dst) ** 2

        print('[#s] anomaly detection...\n', datetime.datetime.now())

        # conducting k-means clustering and recording centroids of different
        # clusters
        kmeans = KMeans(n_clusters=k)
        # Fitting the input data
        kmeans = kmeans.fit(codes)
        # Getting the cluster labels
        indices = kmeans.predict(codes)
        # Centroid values
        centroids = kmeans.cluster_centers_
        # [indices, centroids] = kmeans(codes, k)
        # tbl = tabulate(indices)
        # c = dict.fromkeys(indices, 0)
        #
        # for x in indices:
        #     c[x] += 1
        tbl = Counter(indices)
        n = list(tbl.values())
        c = centroids
        assert (len(n) == k)
        labels = synthetic_test[:, 2]
        # calculating distances for testing edge codes to centroids of clusters
        dist_center = cdist(test_codes, c)
        # assinging each testing edge code to nearest centroid
        min_dist = np.min(dist_center, 1)
        # sorting distances of testing edges to their nearst centroids
        scores = min_dist.argsort()
        scores = scores[::-1]
        # calculating auc score of anomly detection task, in case that all labels are 0's or all 1's
        if np.sum(labels) == 0:
            labels[0] = 1
        elif np.sum(labels) == len(labels):
            labels[0] = 0
        auc = roc_auc_score(labels, min_dist)
        # calculating distances for testing edge codes to centroids of clusters
        dist_center_tr = cdist(codes, c)
        min_dist_tr = np.min(dist_center_tr, 1)
        max_dist_tr = np.max(min_dist_tr)
        res = [1 if x > max_dist_tr else 0 for x in min_dist]
        # ab_score = np.sum(res)/(1e-10 + len(res))
        ab_score = np.sum(min_dist) / (1e-10 + len(min_dist))
        return scores, auc, n, c, res, ab_score


    def dynamic_anomaly_detection(self, embedding_path, sample_path, auc_path, k=2):
        # 字母前缀个数(要转成数字才能进行array操作，把前缀删掉)
        f_list = os.listdir(embedding_path)
        length = len(f_list)
        operation_list = ['Average', 'Hadamard', 'WeightedL1', 'WeightedL2']
        for i, embedding_name in enumerate(f_list):
            print(str(length - i) + " dir(s) left")
            embedding_vec_path = os.path.join(embedding_path, embedding_name)
            embedding_f_list = os.listdir(embedding_vec_path)
            time_length = len(embedding_f_list)

            auc_dict = dict(zip(operation_list, [[]] * 4))
            auc_dict['time'] = []
            for j, f_name in enumerate(embedding_f_list):
                print(str(time_length - j) + " file(s) left")
                df_embedding = pd.read_csv(os.path.join(embedding_vec_path, f_name), sep='\t')
                full_node_list = df_embedding.index.tolist()
                node_idx_list = np.arange(len(full_node_list)).tolist()
                node2idx_dict = dict(zip(full_node_list, node_idx_list))

                df_train = pd.read_csv(os.path.join(sample_path, os.path.splitext(f_name)[0] + '_train.csv'), sep='\t')
                df_test = pd.read_csv(os.path.join(sample_path, os.path.splitext(f_name)[0] + '_test.csv'), sep='\t')
                df_train['from_id'] = df_train['from_id'].map(lambda x: node2idx_dict[x])
                df_train['to_id'] = df_train['to_id'].map(lambda x: node2idx_dict[x])
                df_test['from_id'] = df_test['from_id'].map(lambda x: node2idx_dict[x])
                df_test['to_id'] = df_test['to_id'].map(lambda x: node2idx_dict[x])

                embedding, train, synthetic_test = df_embedding.values, df_train.values, df_test.values
                auc_dict['time'].append(os.path.splitext(f_name)[0])
                for operation in operation_list:
                    scores, auc, n, c, _, _ = self.anomaly_detection(embedding, train, synthetic_test, k,
                                                                encoding_method=operation)
                    auc_dict[operation].append(auc)
            auc_embedding_dir_path = os.path.join(auc_path, embedding_name)
            check_and_make_path(auc_embedding_dir_path)
            df_auc = pd.DataFrame(auc_dict)
            df_auc = df_auc[['time', 'Average', 'Hadamard', 'WeightedL1', 'WeightedL2']]
            df_auc.to_csv(os.path.join(auc_embedding_dir_path, embedding_name + '_auc.csv'), sep=',', index=False)


if __name__ == '__main__':
    data_path = "data\\facebook\\1.format"
    sample_path = "data\\facebook\\sample"
    embedding_path = "data\\facebook\\2.embedding"
    auc_path = "data\\facebook\\auc"

    # 1. 先生成异常检测的采样样本，得到带label的异常检测数据集
    dynamic_anomaly_generation(data_path, sample_path, init_percent=0.8, anomaly_rate=0.2)

    # 2. 对每个embedding方法，训练并测试其每个时间片的anomaly detection模型
    dynamic_anomaly_detection(embedding_path, sample_path, auc_path, k=2)
