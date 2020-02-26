import numpy as np
import pandas as pd
import os, time, sys, multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.append("..")
from RWTGCN.utils import check_and_make_path

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    node_num: int
    test_ratio: float
    val_ratio: float

    def __init__(self, base_path, input_folder, output_folder, node_file, test_ratio=0.3, val_ratio=0.2):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        assert test_ratio + val_ratio < 1.0
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)
        return

    def get_neg_edge_samples(self, pos_edges, edge_num, all_edge_dict):
        neg_edge_dict = dict()
        neg_edge_list = []
        cnt = 0
        while cnt < edge_num:
            from_id = np.random.choice(self.node_num)
            to_id = np.random.choice(self.node_num)
            if from_id == to_id:
                continue
            if (from_id, to_id) in all_edge_dict or (to_id, from_id) in all_edge_dict:
                continue
            if (from_id, to_id) in neg_edge_dict or (to_id, from_id) in neg_edge_dict:
                continue
            neg_edge_list.append([from_id, to_id, 0])
            cnt += 1
        neg_edges = np.array(neg_edge_list)
        all_edges = np.vstack([pos_edges, neg_edges])
        return all_edges

    def generate_edge_samples(self):
        print('Start generating edge samples!')
        node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))

        f_list = os.listdir(self.input_base_path)
        for i, file in enumerate(f_list):
            # print('i = ', i, ', file = ', file)
            file_path = os.path.join(self.input_base_path, file)
            date = file.split('.')[0]
            all_edge_dict = dict()
            edge_list = []

            with open(file_path, 'r') as fp:
                content_list = fp.readlines()
                for line in content_list[1:]:
                    edge = line.strip().split('\t')
                    from_id = node2idx_dict[edge[0]]
                    to_id = node2idx_dict[edge[1]]
                    key = (from_id, to_id)
                    all_edge_dict[key] = 1
                    edge_list.append([from_id, to_id, 1])
                    key = (to_id, from_id)
                    all_edge_dict[key] = 1
                    edge_list.append([to_id, from_id, 1])
                edges = np.array(edge_list)
                del edge_list
                edge_num = edges.shape[0]

                all_edge_idxs = np.arange(edge_num)
                np.random.shuffle(all_edge_idxs)
                test_num = int(np.floor(edge_num * self.test_ratio))
                val_num = int(np.floor(edge_num * self.val_ratio))
                train_num = edge_num - test_num - val_num

                val_edges = edges[all_edge_idxs[ : val_num]]
                test_edges = edges[all_edge_idxs[val_num : val_num + test_num]]
                train_edges = edges[all_edge_idxs[val_num + test_num : ]]
                del edges

                train_edges = self.get_neg_edge_samples(train_edges, train_num, all_edge_dict)
                test_edges = self.get_neg_edge_samples(test_edges, test_num, all_edge_dict)
                val_edges = self.get_neg_edge_samples(val_edges, val_num, all_edge_dict)

            train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
            df_train = pd.DataFrame(train_edges, columns=['from_id', 'to_id', 'label'])
            df_train.to_csv(train_output_path, sep='\t', index=False)

            test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
            df_test = pd.DataFrame(test_edges, columns=['from_id', 'to_id', 'label'])
            df_test.to_csv(test_output_path, sep='\t', index=False)

            val_output_path = os.path.join(self.output_base_path, date + '_val.csv')
            df_val = pd.DataFrame(val_edges, columns=['from_id', 'to_id', 'label'])
            df_val.to_csv(val_output_path, sep='\t', index=False)
        print('Generate edge samples finish!')

class LinkPredictor(object):
    base_path: str
    edge_base_path: str
    embedding_base_path: str
    lp_edge_base_path: str
    output_base_path: str
    full_node_list: list
    train_ratio: float
    test_ratio: float

    def __init__(self, base_path, edge_folder, embedding_folder, lp_edge_folder, output_folder, node_file, train_ratio=1.0, test_ratio=1.0):
        self.base_path = base_path
        self.edge_base_path = os.path.join(base_path, edge_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.lp_edge_base_path = os.path.join(base_path, lp_edge_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.edge_base_path)
        check_and_make_path(self.output_base_path)
        return

    def get_edge_feature(self, edge_arr, embedding_arr):
        avg_edges, had_edges, l1_edges, l2_edges = [], [], [], []
        for i, edge in enumerate(edge_arr):
            from_id, to_id = edge[0], edge[1]
            avg_edges.append((embedding_arr[from_id] + embedding_arr[to_id]) / 2)
            had_edges.append(embedding_arr[from_id] * embedding_arr[to_id])
            l1_edges.append(np.abs(embedding_arr[from_id] - embedding_arr[to_id]))
            l2_edges.append((embedding_arr[from_id] - embedding_arr[to_id]) ** 2)
        avg_edges = np.array(avg_edges)
        had_edges = np.array(had_edges)
        l1_edges = np.array(l1_edges)
        l2_edges = np.array(l2_edges)
        feature_dict = {'Avg': avg_edges, 'Had': had_edges, 'L1': l1_edges, 'L2': l2_edges}
        return feature_dict

    def train(self, train_edges, val_edges, embeddings):
        #print('Start training!')
        train_edge_num = train_edges.shape[0]
        if self.train_ratio < 1.0:
            # print('train ratio < 1. 0')
            sample_num = int(train_edge_num * self.ratio)
            sampled_idxs = np.random.choice(np.arange(train_edge_num), sample_num).tolist()
            train_edges = train_edges[sampled_idxs, :]
        train_labels = train_edges[:, 2]
        val_labels = val_edges[:, 2]
        train_feature_dict = self.get_edge_feature(train_edges, embeddings)
        val_feature_dict = self.get_edge_feature(val_edges, embeddings)

        measure_list = ['Avg', 'Had', 'L1', 'L2']
        model_dict = dict()
        for measure in measure_list:
            models = []
            for C in [0.01, 0.1, 1, 10]:
                model = LogisticRegression(C=C, solver='lbfgs', max_iter=5000, class_weight='balanced')
                model.fit(train_feature_dict[measure], train_labels)
                models.append(model)
            best_auc = 0
            model_idx = -1
            for i, model in enumerate(models):
                val_pred = model.predict_proba(val_feature_dict[measure])[:, 1]
                auc = roc_auc_score(val_labels, val_pred)
                if  auc >= best_auc:
                    best_auc = auc
                    model_idx = i
            #print('model_idx = ', model_idx, ', best_auc=', best_auc)
            model_dict[measure] = models[model_idx]
        #print('Finish training!')
        return model_dict

    def test(self, test_edges, embeddings, model_dict, date):
        #print('Start testing!')
        test_edge_num = test_edges.shape[0]
        if self.test_ratio < 1.0:
            # print('test ratio < 1. 0')
            sample_num = int(test_edge_num * self.ratio)
            sampled_idxs = np.random.choice(np.arange(test_edge_num), sample_num).tolist()
            test_edges = test_edges[sampled_idxs, :]
        test_labels = test_edges[:, 2]
        test_feature_dict = self.get_edge_feature(test_edges, embeddings)
        auc_list = [date]
        measure_list = ['Avg', 'Had', 'L1', 'L2']
        for measure in measure_list:
            test_pred = model_dict[measure].predict_proba(test_feature_dict[measure])[:, 1]
            auc_list.append(roc_auc_score(test_labels, test_pred))
        #print('Finish testing!')
        return auc_list

    def link_prediction_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.edge_base_path))
        f_num = len(f_list)

        # model_dict = dict()
        all_auc_list = []
        for i, f_name in enumerate(f_list):
            if i == 0:
                continue
            print('Current date is: {}'.format(f_name))
            date = f_name.split('.')[0]
            train_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_train.csv'), sep='\t').values
            val_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_val.csv'), sep='\t').values
            test_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_test.csv'), sep='\t').values
            pre_f_name = f_list[i - 1]
            #print('pre_f_name: ', f_list[i - 1], ', f_name: ', f_name)
            if not os.path.exists(os.path.join(self.embedding_base_path, method, pre_f_name)):
                continue
            df_embedding = pd.read_csv(os.path.join(self.embedding_base_path, method, pre_f_name), sep='\t', index_col=0)
            df_embedding  = df_embedding.loc[self.full_node_list]
            node_num = len(self.full_node_list)
            for j in range(node_num):
                assert df_embedding.index[j] == self.full_node_list[j]
            embeddings = df_embedding.values
            #print('YES')
            model_dict = self.train(train_edges, val_edges, embeddings)
            auc_list = self.test(test_edges, embeddings, model_dict, date)
            all_auc_list.append(auc_list)

        df_output = pd.DataFrame(all_auc_list, columns=['date', 'Avg', 'Had', 'L1', 'L2'])
        print(df_output)
        print('method = ', method, ', average AUC of Had: ', df_output['Had'].mean())
        output_file_path = os.path.join(self.output_base_path, method + '_auc_record.csv')
        df_output.to_csv(output_file_path, sep=',', index=False)

    def link_prediction_all_method(self, method_list=None, worker=-1):
        print('Start link prediction!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)

        if worker <= 0:
            for method in method_list:
                print('Current method is :{}'.format(method))
                self.link_prediction_all_time(method)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for method in method_list:
                pool.apply_async(self.link_prediction_all_time, (method,))
            pool.close()
            pool.join()
        print('Finish link prediction!')


if __name__ == '__main__':
    dataset = 'math'
    data_generator = DataGenerator(base_path="../../data/" + dataset, input_folder="1.format",
                                   output_folder="link_prediction_data", node_file="nodes_set/nodes.csv")
    # data_generator.generate_edge_samples()

    link_predictor = LinkPredictor(base_path="../../data/" + dataset, edge_folder='1.format', embedding_folder="2.embedding",
                                   lp_edge_folder="link_prediction_data", output_folder="link_prediction_res", node_file="nodes_set/nodes.csv",
                                   train_ratio=1.0, test_ratio=1.0)
    method_list = ['deepwalk', 'node2vec', 'struct2vec', 'dyGEM', 'timers']
    method_list = ['dyGEM']
    # for neg_num in [10, 20, 50, 80, 100, 150, 200]:
    #     for Q in [0, 10, 20, 50, 100, 200, 500, 1000]:
    #         method_list.append('MRGCN_neg_' + str(neg_num) + '_Q_' + str(Q))
    t1 = time.time()
    link_predictor.link_prediction_all_method(method_list=method_list, worker=-1)
    t2 = time.time()
    print('link prediction cost time: ', t2 - t1, ' seconds!')