# coding: utf-8
import numpy as np
import pandas as pd
import os
import time
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from utils import check_and_make_path


# Generate data used for node classification
class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    label_base_path: str
    file_sep: str
    full_node_list: list
    node2idx_dict: dict
    node_num: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __init__(self, base_path, input_folder, output_folder, node_file, label_folder, file_sep='\t', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.base_path = base_path
        self.input_base_path = os.path.abspath(os.path.join(base_path, input_folder))
        self.output_base_path = os.path.abspath(os.path.join(base_path, output_folder))
        self.label_base_path = os.path.abspath(os.path.join(base_path, label_folder))
        self.file_sep = file_sep

        node_file_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_file_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        self.node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num)))

        assert train_ratio + test_ratio + val_ratio <= 1.0
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)

    def generate_node_samples(self, file_name, sep='\t'):
        date = file_name.split('.')[0]
        file_path = os.path.join(self.label_base_path, file_name)

        df_nodes = pd.read_csv(file_path, sep=sep, header=0, names=['node', 'label'])
        node_num = df_nodes.shape[0]
        df_nodes['node'] = df_nodes['node'].apply(lambda x: self.node2idx_dict[x])
        node_arr = df_nodes['node'].values
        label_arr = df_nodes['label'].values

        node_indices = np.arange(node_num)
        np.random.shuffle(node_indices)
        train_num = int(np.floor(node_num * self.train_ratio))
        val_num = int(np.floor(node_num * self.val_ratio))
        test_num = int(np.floor(node_num * self.test_ratio))

        train_indices = node_indices[: train_num]
        train_nodes, train_labels = node_arr[train_indices], label_arr[train_indices]
        val_indices = node_indices[train_num: train_num + val_num]
        val_nodes, val_labels = node_arr[val_indices], label_arr[val_indices]
        test_indices = node_indices[train_num + val_num: train_num + val_num + test_num]
        test_nodes, test_labels = node_arr[test_indices], label_arr[test_indices]

        train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
        df_train = pd.DataFrame({'node': train_nodes, 'label': train_labels})
        df_train.to_csv(train_output_path, sep=self.file_sep, index=False)
        test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
        df_test = pd.DataFrame({'node': test_nodes, 'label': test_labels})
        df_test.to_csv(test_output_path, sep=self.file_sep, index=False)
        val_output_path = os.path.join(self.output_base_path, date + '_val.csv')
        df_val = pd.DataFrame({'node': val_nodes, 'label': val_labels})
        df_val.to_csv(val_output_path, sep=self.file_sep, index=False)

    def generate_node_samples_all_time(self, sep='\t', worker=-1):
        print('Start generating node samples!')
        f_list = os.listdir(self.input_base_path)

        if worker <= 0:
            for i, file_name in enumerate(f_list):
                print('Current file is :{}'.format(file_name))
                self.generate_node_samples(file_name, sep)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, file_name in enumerate(f_list):
                pool.apply_async(self.generate_node_samples, (file_name, sep))
            pool.close()
            pool.join()

        print('Generate node samples finish!')


# Node classifier class
class NodeClassifier(object):
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    nodecls_base_path: str
    output_base_path: str
    file_sep: str
    full_node_list: list
    unique_labels: list
    C_list: list
    max_iter: int

    def __init__(self, base_path, origin_folder, embedding_folder, nodeclas_folder, output_folder, node_file, label_folder, file_sep='\t', C_list=None, max_iter=5000):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        self.nodecls_base_path = os.path.abspath(os.path.join(base_path, nodeclas_folder))
        self.output_base_path = os.path.abspath(os.path.join(base_path, output_folder))
        self.file_sep = file_sep

        node_file_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_file_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        label_base_path = os.path.abspath(os.path.join(base_path, label_folder))
        f_list = os.listdir(label_base_path)
        assert len(f_list) > 0
        label_path = os.path.join(label_base_path, f_list[0])
        df_label = pd.read_csv(label_path, sep=file_sep)
        self.unique_labels = df_label['label'].unique()
        self.C_list = C_list
        self.max_iter = max_iter

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)

    def train(self, train_nodes, val_nodes, embeddings, lb):
        #print('Start training!')
        train_feature = embeddings[train_nodes[:, 0], :]
        val_feature = embeddings[val_nodes[:, 0], :]
        train_labels = train_nodes[:, 1]
        val_labels = val_nodes[:, 1]

        train_labels = lb.transform(train_labels)
        val_labels = lb.transform(val_labels)

        models = []
        # for C in [0.01, 0.1, 1, 5, 10, 20]:
        for C in self.C_list:
            lr = LogisticRegression(C=C, solver='lbfgs', max_iter=self.max_iter, class_weight='balanced')
            model = OneVsRestClassifier(lr)
            model.fit(train_feature, train_labels)
            models.append(model)
        best_acc = 0
        model_idx = -1
        for i, model in enumerate(models):
            val_pred = model.predict_proba(val_feature)
            val_pred = lb.transform(np.argmax(val_pred, 1))
            acc = accuracy_score(val_labels, val_pred)
            if  acc >= best_acc:
                best_acc = acc
                model_idx = i
        # print('best acc: ', best_acc)
        best_model = models[model_idx]
        #print('Finish training!')
        return best_model

    @staticmethod
    def test(test_nodes, embeddings, model, lb, date):
        test_feature = embeddings[test_nodes[:, 0], :]
        test_labels = test_nodes[:, 1]
        test_labels = lb.transform(test_labels)

        acc_list = [date]
        test_pred = model.predict_proba(test_feature)
        test_pred = lb.transform(np.argmax(test_pred, 1))
        acc_list.append(accuracy_score(test_labels, test_pred))
        return acc_list

    def node_classification_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.origin_base_path))
        all_acc_list = []
        for i, f_name in enumerate(f_list):
            print('Current date is: {}'.format(f_name))
            date = f_name.split('.')[0]
            train_nodes = pd.read_csv(os.path.join(self.nodecls_base_path, date + '_train.csv'), sep=self.file_sep).values
            val_nodes = pd.read_csv(os.path.join(self.nodecls_base_path, date + '_val.csv'), sep=self.file_sep).values
            test_nodes = pd.read_csv(os.path.join(self.nodecls_base_path, date + '_test.csv'), sep=self.file_sep).values
            cur_embedding_path = os.path.join(self.embedding_base_path, method, f_name)
            if not os.path.exists(cur_embedding_path):
                continue
            df_embedding = pd.read_csv(cur_embedding_path, sep=self.file_sep, index_col=0)
            df_embedding = df_embedding.loc[self.full_node_list]
            embeddings = df_embedding.values

            lb = preprocessing.LabelBinarizer()
            lb.fit(self.unique_labels)
            model = self.train(train_nodes, val_nodes, embeddings, lb)
            acc_list = self.test(test_nodes, embeddings, model, lb, date)
            all_acc_list.append(acc_list)

        df_output = pd.DataFrame(all_acc_list, columns=['date', 'acc'])
        print(df_output)
        print('method = ', method, ', average accuracy: ', df_output['acc'].mean())
        output_file_path = os.path.join(self.output_base_path, method + '_acc_record.csv')
        df_output.to_csv(output_file_path, sep=',', index=False)

    def node_classification_all_method(self, method_list=None, worker=-1):
        print('Start node classification!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)

        if worker <= 0:
            for method in method_list:
                print('Current method is :{}'.format(method))
                self.node_classification_all_time(method)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for method in method_list:
                pool.apply_async(self.node_classification_all_time, (method,))
            pool.close()
            pool.join()
        print('Finish node classification!')


# Aggregate all node classification results and write into a single result file for each graph embedding method
def aggregate_results(base_path, nodecls_res_folder, start_idx, rep_num, method_list):
    if rep_num <= 0:
        return
    # Aggregate node classification results when rep_num > 0
    for method in method_list:
        res_base_path = os.path.join(base_path, nodecls_res_folder + '_' + str(start_idx))
        res_path = os.path.join(res_base_path, method + '_acc_record.csv')

        column_names = ['date', 'acc_' + str(start_idx)]
        df_method = pd.read_csv(res_path, sep=',', header=0, names=column_names)
        for i in range(start_idx + 1, start_idx + rep_num):
            res_base_path = os.path.join(base_path, nodecls_res_folder + '_' + str(i))
            res_path = os.path.join(res_base_path, method + '_acc_record.csv')
            column_names = ['date', 'acc_' + str(i)]
            df_rep = pd.read_csv(res_path, sep=',', header=0, names=column_names)
            df_method = pd.concat([df_method, df_rep.iloc[:, [1]]], axis=1)
        output_base_path = os.path.join(base_path, nodecls_res_folder)
        check_and_make_path(output_base_path)
        acc_list = ['acc_' + str(i) for i in range(start_idx, start_idx + rep_num)]
        df_method['avg'] = df_method.loc[:, acc_list].mean(axis=1)
        df_method['max'] = df_method.loc[:, acc_list].max(axis=1)
        df_method['min'] = df_method.loc[:, acc_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_acc_record.csv')
        df_method.to_csv(output_path, sep=',', index=False)


def node_classification(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    node_file = args['node_file']
    nlabel_folder = args['nlabel_folder']
    nodecls_data_folder = args['nodecls_data_folder']
    nodecls_res_folder = args['nodecls_res_folder']
    file_sep = args['file_sep']
    start_idx = args['start_idx']
    rep_num = args['rep_num']
    train_ratio = args['train_ratio']
    val_ratio = args['val_ratio']
    test_ratio = args['test_ratio']
    do_nodecls = args['do_nodecls']
    generate = args['generate']
    aggregate = args['aggregate']
    method_list = args['method_list']
    C_list = args['c_list']
    max_iter = args['max_iter']
    worker = args['worker']

    t1 = time.time()
    if do_nodecls:
        for i in range(start_idx, start_idx + rep_num):
            print('idx = ', i)
            data_generator = DataGenerator(base_path=base_path, input_folder=origin_folder, output_folder=nodecls_data_folder + '_' + str(i), node_file=node_file, label_folder=nlabel_folder,
                                           file_sep=file_sep, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            if generate:
                data_generator.generate_node_samples_all_time(sep=file_sep, worker=worker)

            node_classifier = NodeClassifier(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, nodeclas_folder=nodecls_data_folder + '_' + str(i),
                                             output_folder=nodecls_res_folder + '_' + str(i), node_file=node_file, label_folder=nlabel_folder, file_sep=file_sep, C_list=C_list, max_iter=max_iter)
            node_classifier.node_classification_all_method(method_list=method_list, worker=worker)

    t2 = time.time()
    print('node classification cost time: ', t2 - t1, ' seconds!')

    if aggregate:
        aggregate_results(base_path, nodecls_res_folder, start_idx, rep_num, method_list)