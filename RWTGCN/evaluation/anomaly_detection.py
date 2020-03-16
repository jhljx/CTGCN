import numpy as np
import pandas as pd
import os, datetime, sys, time, multiprocessing
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, get_sp_adj_mat

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    node_num: int
    train_ratio: float
    val_ratio: float
    anomaly_ratio: float

    def __init__(self, base_path, input_folder, output_folder, node_file, train_ratio=0.5, val_ratio=0.2, anomaly_ratio=0.1):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        assert 0 < train_ratio < 1 and 0 < anomaly_ratio < 1
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.anomaly_ratio = anomaly_ratio

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)
        return

    #S = VS(A, 0.9, 100)
    # S = (S + S.T) / 2
    def get_vertex_similarity(self, A, alpha, iter_num=100, normalize=False):
        """the implement of Vertex similarity in networks"""
        import scipy
        assert 0 < alpha < 1
        assert type(A) is scipy.sparse.csr.csr_matrix
        lambda_1 = scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
        n = A.shape[0]
        d = np.array(A.sum(1)).flatten()
        def inv(x):
            if x == 0:
                return x
            return 1. / x
        inv_func = np.vectorize(inv)
        d_inv = np.diag(inv_func(d))
        dsd = np.random.normal(0, 1 / np.sqrt(n), (n, n))
        # dsd = np.zeros((n, n))
        I = np.eye(n)
        for i in range(iter_num):
            dsd = alpha / lambda_1 * A.dot(dsd) + I
            if i % 10 == 0:
                print('VS', i, '/', iter_num)
        if normalize == False:
            return dsd
        return d_inv.dot(dsd).dot(d_inv)

    def get_anomaly_edge_samples(self, pos_edges, anomaly_num, all_edge_dict, cluster_arr):
        anomaly_edge_dict = dict()
        anomaly_edge_list = []
        cnt = 0
        while cnt < anomaly_num:
            from_id = np.random.choice(self.node_num)
            to_id = np.random.choice(self.node_num)
            if from_id == to_id:
                continue
            if (from_id, to_id) in all_edge_dict or (to_id, from_id) in all_edge_dict:
                continue
            if (from_id, to_id) in anomaly_edge_dict or (to_id, from_id) in anomaly_edge_dict:
                continue
            if cluster_arr[from_id] == cluster_arr[to_id]:
                continue
            anomaly_edge_list.append([from_id, to_id, 0])
            cnt += 1
        anomaly_edges = np.array(anomaly_edge_list)
        all_edges = np.vstack([pos_edges, anomaly_edges])
        all_edge_idxs = np.arange(all_edges.shape[0])
        np.random.shuffle(all_edge_idxs)
        all_edges = all_edges[all_edge_idxs, :]
        return all_edges

    def generate_node_samples(self):
        print('Start generating node samples!')
        node_arr = np.arange(self.node_num)

        f_list = sorted(os.listdir(self.input_base_path))
        for i, file in enumerate(f_list):
            file_path = os.path.join(self.input_base_path, file)
            date = file.split('.')[0]
            spadj = get_sp_adj_mat(file_path, self.full_node_list)
            degrees = spadj.sum(axis=1).flatten().A[0].astype(np.int)
            spadj = spadj.tolil()
            node_neighbor_arr = spadj.rows
            degree_ranks = degrees.argsort()[::-1]

            anomaly_num = int(np.floor(self.node_num * self.anomaly_ratio))
            cnt, j = 0, 0
            anomaly_list = []
            anomaly_dict = dict()
            while cnt < anomaly_num:
                cur = degree_ranks[j]
                j += 1
                if cur not in anomaly_dict:
                    anomaly_dict[cur] = 1
                    anomaly_list.append(cur)
                    cnt += 1

                neighbor_arr = node_neighbor_arr[cur]
                neighbor_num = len(neighbor_arr)
                sample_num = np.random.randint(1, neighbor_num // 2)
                sample_arr = np.random.choice(neighbor_arr, sample_num, replace=False)
                for nidx in sample_arr:
                    if nidx not in anomaly_dict:
                        anomaly_dict[nidx] = 1
                        anomaly_list.append(nidx)
                        cnt += 1

            labels = np.zeros(self.node_num, dtype=np.int)
            labels[anomaly_list] = 1

            node_idxs = np.arange(self.node_num)
            data = np.vstack((node_idxs, labels)).T
            np.random.shuffle(node_idxs)
            train_num = int(np.floor(self.node_num * self.train_ratio))
            val_num = int(np.floor(self.node_num * self.val_ratio))

            train_data = data[node_idxs[ : train_num]]
            val_data = data[node_idxs[train_num: train_num + val_num]]
            test_data = data[node_idxs[train_num + val_num: ]]

            train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
            df_train = pd.DataFrame(train_data, columns=['node', 'label'])
            df_train.to_csv(train_output_path, sep='\t', index=False)

            test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
            df_test = pd.DataFrame(test_data, columns=['node', 'label'])
            df_test.to_csv(test_output_path, sep='\t', index=False)

            val_output_path = os.path.join(self.output_base_path, date + '_val.csv')
            df_val = pd.DataFrame(val_data, columns=['node', 'label'])
            df_val.to_csv(val_output_path, sep='\t', index=False)

    def generate_edge_samples(self):
        print('Start generating edge samples!')
        node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))

        f_list = sorted(os.listdir(self.input_base_path))
        for i, file in enumerate(f_list):
            file_path = os.path.join(self.input_base_path, file)
            date = file.split('.')[0]
            all_edge_dict = dict()
            edge_list = []
            adj = np.zeros((self.node_num, self.node_num), dtype=np.int)

            with open(file_path, 'r') as fp:
                content_list = fp.readlines()
                for line in content_list[1:]:
                    edge = line.strip().split('\t')
                    from_id = node2idx_dict[edge[0]]
                    to_id = node2idx_dict[edge[1]]
                    key = (from_id, to_id)
                    all_edge_dict[key] = 1
                    adj[from_id, to_id] = 1
                    edge_list.append([from_id, to_id, 1])
                    key = (to_id, from_id)
                    all_edge_dict[key] = 1
                    adj[to_id, from_id] = 1
                    edge_list.append([to_id, from_id, 1])
            # for i in range(self.node_num):
            #     adj[i, i] = 1
            kk = 10 # 10#42#42
            #model = KMeans(n_clusters=kk)
            model = SpectralClustering(kk, affinity='precomputed', n_init=100, assign_labels='discretize')
            cluster_arr = model.fit_predict(adj)

            edges = np.array(edge_list)
            del edge_list
            edge_num = edges.shape[0]

            all_edge_idxs = np.arange(edge_num)
            np.random.shuffle(all_edge_idxs)
            train_num = int(np.floor(edge_num * self.train_ratio))
            test_num = edge_num - train_num

            train_edges = edges[all_edge_idxs[ : train_num]]
            test_edges = edges[all_edge_idxs[train_num : ]]
            del edges
            anomaly_num = int(np.floor(test_num * self.anomaly_ratio))
            test_edges = self.get_anomaly_edge_samples(test_edges, anomaly_num, all_edge_dict, cluster_arr)

            train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
            df_train = pd.DataFrame(train_edges, columns=['from_id', 'to_id', 'label'])
            df_train.to_csv(train_output_path, sep='\t', index=False)

            test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
            df_test = pd.DataFrame(test_edges, columns=['from_id', 'to_id', 'label'])
            df_test.to_csv(test_output_path, sep='\t', index=False)
        print('Generate edge samples finish!')

class AnomalyDetector(object):
    base_path: str
    origin_base_path: str
    origin_base_path: str
    embedding_base_path: str
    ad_edge_base_path: str
    output_base_path: str
    full_node_list: list
    train_ratio: float
    test_ratio: float

    def __init__(self, base_path, origin_folder, embedding_folder, ad_edge_folder, output_folder, node_file):
        self.base_path = base_path
        self.origin_base_path = os.path.join(base_path, origin_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.ad_edge_base_path = os.path.join(base_path, ad_edge_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)
        return

    def train(self, train_nodes, val_nodes, embeddings):
        #print('Start training!')
        train_feature = embeddings[train_nodes[:, 0], :]
        val_feature = embeddings[val_nodes[:, 0], :]
        train_labels = train_nodes[:, 1]
        val_labels = val_nodes[:, 1]

        models = []
        for C in [0.01, 0.1, 1, 5, 10, 20]:
            model = LogisticRegression(C=C, solver='lbfgs', max_iter=5000, class_weight='balanced')
            model.fit(train_feature, train_labels)
            models.append(model)
        best_auc = 0
        model_idx = -1
        for i, model in enumerate(models):
            val_pred = model.predict_proba(val_feature)[:, 1]
            auc = roc_auc_score(val_labels, val_pred)
            if  auc >= best_auc:
                best_auc = auc
                model_idx = i
        # print('best acc: ', best_acc)
        best_model = models[model_idx]
        #print('Finish training!')
        return best_model

    def test(self, test_nodes, embeddings, model, date):
        test_feature = embeddings[test_nodes[:, 0], :]
        test_labels = test_nodes[:, 1]

        auc_list = [date]
        test_pred = model.predict_proba(test_feature)[:, 1]
        auc_list.append(roc_auc_score(test_labels, test_pred))
        return auc_list

    # def get_edge_feature(self, edge_arr, embedding_arr):
    #     avg_edges, had_edges, l1_edges, l2_edges = [], [], [], []
    #     for i, edge in enumerate(edge_arr):
    #         from_id, to_id = edge[0], edge[1]
    #         avg_edges.append((embedding_arr[from_id] + embedding_arr[to_id]) / 2)
    #         had_edges.append(embedding_arr[from_id] * embedding_arr[to_id])
    #         l1_edges.append(np.abs(embedding_arr[from_id] - embedding_arr[to_id]))
    #         l2_edges.append((embedding_arr[from_id] - embedding_arr[to_id]) ** 2)
    #     avg_edges = np.array(avg_edges)
    #     had_edges = np.array(had_edges)
    #     l1_edges = np.array(l1_edges)
    #     l2_edges = np.array(l2_edges)
    #     feature_dict = {'Avg': avg_edges, 'Had': had_edges, 'L1': l1_edges, 'L2': l2_edges}
    #     return feature_dict
    #
    # def train(self, train_edges, embeddings):
    #     #print('Start training!')
    #     train_feature_dict = self.get_edge_feature(train_edges, embeddings)
    #     k = 10
    #     measure_list = ['Avg', 'Had', 'L1', 'L2']
    #     centroid_dict = dict()
    #     for measure in measure_list:
    #         model = KMeans(n_clusters=k)
    #         model = model.fit(train_feature_dict[measure])
    #         centroids = model.cluster_centers_
    #         centroid_dict[measure] = centroids
    #     #print('Finish training!')
    #     return centroid_dict
    #
    # def test(self, test_edges, embeddings, centroid_dict, date):
    #     #print('Start testing!')
    #     test_labels = test_edges[:, 2]
    #     test_feature_dict = self.get_edge_feature(test_edges, embeddings)
    #     auc_list = [date]
    #     measure_list = ['Avg', 'Had', 'L1', 'L2']
    #     for measure in measure_list:
    #         dist_center_arr = cdist(test_feature_dict[measure], centroid_dict[measure])
    #         min_dist_arr =  np.min(dist_center_arr, 1)
    #         auc_list.append(roc_auc_score(test_labels, min_dist_arr))
    #     #print('Finish testing!')
    #     return auc_list

    def anomaly_detection_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.origin_base_path))

        all_auc_list = []
        for i, f_name in enumerate(f_list):
            if i == 0:
                continue
            print('Current date is: {}'.format(f_name))
            date = f_name.split('.')[0]
            train_data = pd.read_csv(os.path.join(self.ad_edge_base_path, date + '_train.csv'), sep='\t').values
            val_data = pd.read_csv(os.path.join(self.ad_edge_base_path, date + '_test.csv'), sep='\t').values
            test_data = pd.read_csv(os.path.join(self.ad_edge_base_path, date + '_test.csv'), sep='\t').values
            # train_edges = pd.read_csv(os.path.join(self.ad_edge_base_path, date + '_train.csv'), sep='\t').values
            # test_edges = pd.read_csv(os.path.join(self.ad_edge_base_path, date + '_test.csv'), sep='\t').values
            if not os.path.exists(os.path.join(self.embedding_base_path, method, f_name)):
                continue
            df_embedding = pd.read_csv(os.path.join(self.embedding_base_path, method, f_name), sep='\t', index_col=0)
            df_embedding  = df_embedding.loc[self.full_node_list]
            embeddings = df_embedding.values

            #lb = preprocessing.LabelBinarizer()
            #lb.fit([0, 1])
            model = self.train(train_data, val_data, embeddings)
            auc_list = self.test(test_data, embeddings, model, date)
            # centroid_dict = self.train(train_edges, embeddings)
            # auc_list = self.test(test_edges, embeddings, centroid_dict, date)
            all_auc_list.append(auc_list)

        # df_output = pd.DataFrame(all_auc_list, columns=['date', 'Avg', 'Had', 'L1', 'L2'])
        df_output = pd.DataFrame(all_auc_list, columns=['date', 'auc'])
        print(df_output)
        print('method = ', method, ', average AUC: ', df_output['auc'].mean())
        output_file_path = os.path.join(self.output_base_path, method + '_auc_record.csv')
        df_output.to_csv(output_file_path, sep=',', index=False)

    def anomaly_detection_all_method(self, method_list=None, worker=-1):
        print('Start anomaly_detection!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)

        if worker <= 0:
            for method in method_list:
                print('Current method is :{}'.format(method))
                self.anomaly_detection_all_time(method)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for method in method_list:
                pool.apply_async(self.anomaly_detection_all_time, (method,))
            pool.close()
            pool.join()
        print('Finish anomaly_detection!')

def process_result(dataset, rep_num, method_list):
    for method in method_list:
        base_path = os.path.join('../../data/' + dataset, 'anomaly_detection_res_0')
        res_path = os.path.join(base_path, method + '_auc_record.csv')
        df_method = pd.read_csv(res_path, sep=',', header=0, names=['date', 'avg0', 'had0', 'l1_0', 'l2_0'])
        df_avg = df_method.loc[:, ['date', 'avg0']].copy()
        df_had = df_method.loc[:, ['date', 'had0']].copy()
        df_l1 = df_method.loc[:, ['date', 'l1_0']].copy()
        df_l2 = df_method.loc[:, ['date', 'l2_0']].copy()
        for i in range(1, rep_num):
            base_path = os.path.join('../../data/' + dataset, 'anomaly_detection_res_' + str(i))
            res_path = os.path.join(base_path, method + '_auc_record.csv')
            df_rep = pd.read_csv(res_path, sep=',', header=0, names=['date', 'avg' + str(i), 'had' + str(i), 'l1_' + str(i), 'l2_' + str(i)])
            df_avg = pd.concat([df_avg, df_rep.loc[:, ['avg' + str(i)]]], axis=1)
            df_had = pd.concat([df_had, df_rep.loc[:, ['had' + str(i)]]], axis=1)
            df_l1 = pd.concat([df_l1, df_rep.loc[:, ['l1_' + str(i)]]], axis=1)
            df_l2 = pd.concat([df_l2, df_rep.loc[:, ['l2_' + str(i)]]], axis=1)
        output_base_path = os.path.join('../../data/' + dataset, 'anomaly_detection_res')
        check_and_make_path(output_base_path)

        avg_list = ['avg' + str(i) for i in range(rep_num)]
        df_avg['avg'] = df_avg.loc[:, avg_list].mean(axis=1)
        df_avg['max'] = df_avg.loc[:, avg_list].max(axis=1)
        df_avg['min'] = df_avg.loc[:, avg_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_avg_record.csv')
        df_avg.to_csv(output_path, sep=',', index=False)

        had_list = ['had' + str(i) for i in range(rep_num)]
        df_had['avg'] = df_had.loc[:, had_list].mean(axis=1)
        df_had['max'] = df_had.loc[:, had_list].max(axis=1)
        df_had['min'] = df_had.loc[:, had_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_had_record.csv')
        df_had.to_csv(output_path, sep=',', index=False)

        l1_list = ['l1_' + str(i) for i in range(rep_num)]
        df_l1['avg'] = df_l1.loc[:, l1_list].mean(axis=1)
        df_l1['max'] = df_l1.loc[:, l1_list].max(axis=1)
        df_l1['min'] = df_l1.loc[:, l1_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_l1_record.csv')
        df_l1.to_csv(output_path, sep=',', index=False)

        l2_list = ['l2_' + str(i) for i in range(rep_num)]
        df_l2['avg'] = df_l2.loc[:, l2_list].mean(axis=1)
        df_l2['max'] = df_l2.loc[:, l2_list].max(axis=1)
        df_l2['min'] = df_l2.loc[:, l2_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_l2_record.csv')
        df_l2.to_csv(output_path, sep=',', index=False)

if __name__ == '__main__':
    dataset = 'email-eu'
    rep_num = 1

    method_list = ['deepwalk', 'node2vec', 'struct2vec', 'GCN', 'GAT', 'dyGEM', 'timers', 'EvolveGCNH', 'RWTGCN_C', 'RWTGCN_S',  'CGCN_C', 'CGCN_S']

    for i in range(0, rep_num):
        data_generator = DataGenerator(base_path="../../data/" + dataset, input_folder="1.format",
                                       output_folder="anomaly_detection_data_" + str(i), node_file="nodes_set/nodes.csv")
        data_generator.generate_node_samples()
        anomaly_detector = AnomalyDetector(base_path="../../data/" + dataset, origin_folder='1.format', embedding_folder="2.embedding",
                                       ad_edge_folder="anomaly_detection_data_" + str(i), output_folder="anomaly_detection_res_" + str(i), node_file="nodes_set/nodes.csv")
        t1 = time.time()
        anomaly_detector.anomaly_detection_all_method(method_list=method_list, worker=12)
        t2 = time.time()
        print('anomaly detection cost time: ', t2 - t1, ' seconds!')

    # process_result(dataset, rep_num, method_list)