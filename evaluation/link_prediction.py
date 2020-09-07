# coding: utf-8
import numpy as np
import pandas as pd
import os
import time
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils import check_and_make_path, get_neg_edge_samples, sigmoid


# Generate data used for link prediction
class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    file_sep: str
    full_node_list: list
    node_num: int
    node2idx_dict: dict
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __init__(self, base_path, input_folder, output_folder, node_file, file_sep='\t', train_ratio=0.5, val_ratio=0.2, test_ratio=0.3):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        self.file_sep = file_sep

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        self.node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num)))

        assert train_ratio + test_ratio + val_ratio <= 1.0
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)

    def generate_edge_sample(self, file, sep='\t'):
        file_path = os.path.join(self.input_base_path, file)
        date = file.split('.')[0]
        all_edge_dict = dict()
        edge_list = []

        with open(file_path, 'r') as fp:
            content_list = fp.readlines()
            for line in content_list[1:]:
                edge = line.strip().split(sep)
                from_id = self.node2idx_dict[edge[0]]
                to_id = self.node2idx_dict[edge[1]]
                key = (from_id, to_id)
                all_edge_dict[key] = 1
                edge_list.append([from_id, to_id, 1])
                key = (to_id, from_id)
                all_edge_dict[key] = 1
                edge_list.append([to_id, from_id, 1])

        all_edges = np.array(edge_list)
        del edge_list
        edge_num = all_edges.shape[0]

        np.random.shuffle(all_edges)
        test_num = int(np.floor(edge_num * self.test_ratio))
        val_num = int(np.floor(edge_num * self.val_ratio))
        train_num = int(np.floor((edge_num - test_num - val_num) * self.train_ratio))

        val_edges = all_edges[: val_num]
        test_edges = all_edges[val_num: val_num + test_num]
        train_edges = all_edges[val_num + test_num: val_num + test_num + train_num]
        del all_edges

        train_edges = get_neg_edge_samples(train_edges, train_num, all_edge_dict, self.node_num)
        test_edges = get_neg_edge_samples(test_edges, test_num, all_edge_dict, self.node_num)
        val_edges = get_neg_edge_samples(val_edges, val_num, all_edge_dict, self.node_num)

        train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
        df_train = pd.DataFrame(train_edges, columns=['from_id', 'to_id', 'label'])
        df_train.to_csv(train_output_path, sep=self.file_sep, index=False)

        test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
        df_test = pd.DataFrame(test_edges, columns=['from_id', 'to_id', 'label'])
        df_test.to_csv(test_output_path, sep=self.file_sep, index=False)

        val_output_path = os.path.join(self.output_base_path, date + '_val.csv')
        df_val = pd.DataFrame(val_edges, columns=['from_id', 'to_id', 'label'])
        df_val.to_csv(val_output_path, sep=self.file_sep, index=False)

    def generate_edge_samples_all_time(self, sep='\t', worker=-1):
        print('Start generating edge samples!')
        f_list = sorted(os.listdir(self.input_base_path))

        if worker <= 0:
            for i, file_name in enumerate(f_list):
                print('Current file is :{}'.format(file_name))
                self.generate_edge_sample(file_name, sep=sep)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, file_name in enumerate(f_list):
                pool.apply_async(self.generate_edge_sample, (file_name, sep))
            pool.close()
            pool.join()

        print('Generate edge samples finish!')


# Link predictor class
class LinkPredictor(object):
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    lp_edge_base_path: str
    output_base_path: str
    file_sep: str
    full_node_list: list
    C_list: list
    max_iter: int

    def __init__(self, base_path, origin_folder, embedding_folder, lp_edge_folder, output_folder, node_file, file_sep='\t', C_list=None, measure_list=None, max_iter=5000):
        self.base_path = base_path
        self.origin_base_path = os.path.join(base_path, origin_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.lp_edge_base_path = os.path.join(base_path, lp_edge_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        self.file_sep = file_sep
        self.measure_list = measure_list

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.C_list = C_list
        self.max_iter = max_iter

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)

    def get_edge_feature(self, edge_arr, embedding_arr):
        feature_dict = dict()
        for measure in self.measure_list:
            assert measure in ['Avg', 'Had', 'L1', 'L2', 'sigmoid']
            feature_dict[measure] = []
        for i, edge in enumerate(edge_arr):
            from_id, to_id = edge[0], edge[1]
            for measure in self.measure_list:
                if measure == 'Avg':
                    feature_dict[measure].append((embedding_arr[from_id] + embedding_arr[to_id]) / 2)
                elif measure == 'Had':
                    feature_dict[measure].append(embedding_arr[from_id] * embedding_arr[to_id])
                elif measure == 'L1':
                    feature_dict[measure].append(np.abs(embedding_arr[from_id] - embedding_arr[to_id]))
                elif measure == 'L2':
                    feature_dict[measure].append((embedding_arr[from_id] - embedding_arr[to_id]) ** 2)
                elif measure == 'sigmoid':
                    feature_dict[measure].append(sigmoid(np.sum(embedding_arr[from_id] * embedding_arr[to_id])))
        for measure in self.measure_list:
            feature_dict[measure] = np.array(feature_dict[measure])
        return feature_dict

    def train(self, train_edges, val_edges, embeddings):
        print('Start training!')
        train_labels = train_edges[:, 2]
        val_labels = val_edges[:, 2]
        train_feature_dict = self.get_edge_feature(train_edges, embeddings)
        val_feature_dict = self.get_edge_feature(val_edges, embeddings)

        # measure_list = ['Avg', 'Had', 'L1', 'L2']
        model_dict = dict()
        for measure in self.measure_list:
            if measure == 'sigmoid':
                continue
            models = []
            # for C in [0.01, 0.1, 1, 10]:
            for C in self.C_list:
                model = LogisticRegression(C=C, solver='lbfgs', max_iter=self.max_iter, class_weight='balanced')
                model.fit(train_feature_dict[measure], train_labels)
                models.append(model)
            best_auc = 0
            model_idx = -1
            for i, model in enumerate(models):
                val_pred = model.predict_proba(val_feature_dict[measure])[:, 1]
                auc = roc_auc_score(val_labels, val_pred)
                if auc >= best_auc:
                    best_auc = auc
                    model_idx = i
            #print('model_idx = ', model_idx, ', best_auc=', best_auc)
            model_dict[measure] = models[model_idx]
        print('Finish training!')
        return model_dict

    def test(self, test_edges, embeddings, model_dict, date):
        test_labels = test_edges[:, 2]
        test_feature_dict = self.get_edge_feature(test_edges, embeddings)
        auc_list = [date]
        for measure in self.measure_list:
            if measure == 'sigmoid':
                test_pred = test_feature_dict[measure]
            else:
                test_pred = model_dict[measure].predict_proba(test_feature_dict[measure])[:, 1]
            auc_list.append(roc_auc_score(test_labels, test_pred))
        return auc_list

    def link_prediction_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.origin_base_path))
        # f_num = len(f_list)

        all_auc_list = []
        for i, f_name in enumerate(f_list):
            if i == 0:
                continue
            date = f_name.split('.')[0]
            train_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_train.csv'), sep=self.file_sep).values
            val_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_val.csv'), sep=self.file_sep).values
            test_edges = pd.read_csv(os.path.join(self.lp_edge_base_path, date + '_test.csv'), sep=self.file_sep).values
            pre_f_name = f_list[i - 1]
            pre_embedding_path = os.path.join(self.embedding_base_path, method, pre_f_name)
            if not os.path.exists(pre_embedding_path):
                continue
            # print('pre_f_name: ', f_list[i - 1], ', f_name: ', f_name)
            print('Current date is: {}'.format(f_name))
            df_embedding = pd.read_csv(pre_embedding_path, sep=self.file_sep, index_col=0)
            df_embedding = df_embedding.loc[self.full_node_list, :]
            # node_num = len(self.full_node_list)
            # for j in range(node_num):
            #     assert df_embedding.index[j] == self.full_node_list[j]
            embeddings = df_embedding.values
            model_dict = self.train(train_edges, val_edges, embeddings)
            auc_list = self.test(test_edges, embeddings, model_dict, date)
            all_auc_list.append(auc_list)
        # print('all auc list len: ', len(all_auc_list)
        column_names = ['date'] + self.measure_list
        df_output = pd.DataFrame(all_auc_list, columns=column_names)
        print(df_output)
        print(df_output.iloc[-4:, 2])
        print('method = ', method, ', average AUC of Had: ', df_output.iloc[-4:, 2].mean())
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


# Aggregate all link prediction results and write into a single result file for each graph embedding method
def aggregate_results(base_path, lp_res_folder, start_idx, rep_num, method_list, measure_list):
    if rep_num <= 0:
        return
    # Aggregate link prediction results when rep_num > 0
    for method in method_list:
        res_base_path = os.path.join(base_path, lp_res_folder + '_' + str(start_idx))
        res_path = os.path.join(res_base_path, method + '_auc_record.csv')
        column_names = ['date'] + [measure + '_' + str(start_idx) for measure in measure_list]
        df_method = pd.read_csv(res_path, sep=',', header=0, names=column_names)
        measure_df_dict = dict()
        for measure in measure_list:
            df_measure = df_method.loc[:, ['date', measure + '_' + str(start_idx)]].copy()
            measure_df_dict[measure] = df_measure
        for i in range(start_idx + 1, start_idx + rep_num):
            res_base_path = os.path.join(base_path, lp_res_folder + '_' + str(i))
            res_path = os.path.join(res_base_path, method + '_auc_record.csv')
            column_names = ['date'] + [measure + '_' + str(i) for measure in measure_list]
            df_rep = pd.read_csv(res_path, sep=',', header=0, names=column_names)
            for measure in measure_list:
                measure_df_dict[measure] = pd.concat([measure_df_dict[measure], df_rep.loc[:, [measure + '_' + str(i)]]], axis=1)
        output_base_path = os.path.join(base_path, lp_res_folder)
        check_and_make_path(output_base_path)

        for measure in measure_list:
            measure_column = [measure + '_' + str(i) for i in range(start_idx, start_idx + rep_num)]
            df_measure = measure_df_dict[measure]
            df_measure['avg'] = df_measure.loc[:, measure_column].mean(axis=1)
            df_measure['max'] = df_measure.loc[:, measure_column].max(axis=1)
            df_measure['min'] = df_measure.loc[:, measure_column].min(axis=1)
            output_path = os.path.join(output_base_path, method + '_' + measure + '_record.csv')
            df_measure.to_csv(output_path, sep=',', index=False)


def link_prediction(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    node_file = args['node_file']
    lp_edge_folder = args['lp_edge_folder']
    lp_res_folder = args['lp_res_folder']
    file_sep = args['file_sep']
    start_idx = args['start_idx']
    rep_num = args['rep_num']
    train_ratio = args['train_ratio']
    val_ratio = args['val_ratio']
    test_ratio = args['test_ratio']
    do_lp = args['do_lp']
    generate = args['generate']
    aggregate = args['aggregate']
    method_list = args['method_list']
    C_list = args['c_list']
    measure_list = args['measure_list']
    max_iter = args['max_iter']
    worker = args['worker']

    if do_lp:
        for i in range(start_idx, start_idx + rep_num):
            data_generator = DataGenerator(base_path=base_path, input_folder=origin_folder,  output_folder=lp_edge_folder + '_' + str(i), node_file=node_file,
                                           file_sep=file_sep, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            if generate:
                data_generator.generate_edge_samples_all_time(sep=file_sep, worker=worker)
            link_predictor = LinkPredictor(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, lp_edge_folder=lp_edge_folder + '_' + str(i),
                                           output_folder=lp_res_folder + '_' + str(i), node_file=node_file, file_sep=file_sep, C_list=C_list, measure_list=measure_list, max_iter=max_iter)
            t1 = time.time()
            link_predictor.link_prediction_all_method(method_list=method_list, worker=worker)
            t2 = time.time()
            print('link prediction cost time: ', t2 - t1, ' seconds!')

    if aggregate:
        aggregate_results(base_path, lp_res_folder, start_idx, rep_num, method_list, measure_list)