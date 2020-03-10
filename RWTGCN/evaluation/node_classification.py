import numpy as np
import pandas as pd
import os, time, sys, multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
sys.path.append("..")
from RWTGCN.utils import check_and_make_path

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    label_list: list
    node_num: int
    test_ratio: float
    val_ratio: float

    def __init__(self, base_path, input_folder, output_folder, node_file, label_file, trans_label_file, sep=' ', test_ratio=0.1, val_ratio=0.2):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)

        node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))
        df_label = pd.read_csv(os.path.join(base_path, label_file), sep=sep, header=0, names=['node', 'label'], dtype=str)
        df_label['node'] = df_label['node'].apply(lambda x: 'U' + x)
        df_label['label'] = df_label['label'].apply(np.int)
        df_label['node'] = df_label['node'].apply(lambda x: node2idx_dict[x])
        # print(node_idx_list)
        df_label.index = df_label['node'].tolist()
        df_label = df_label.loc[np.arange(self.node_num).tolist(), :]
        # print(df_label)
        self.label_list = df_label['label'].tolist()
        df_label.to_csv(os.path.join(base_path, trans_label_file), sep='\t', index=False)

        assert test_ratio + val_ratio < 1.0
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)
        return

    def generate_node_samples(self):
        print('Start generating node samples!')
        node_arr = np.arange(self.node_num)
        label_arr = np.array(self.label_list)

        f_list = os.listdir(self.input_base_path)
        for i, file in enumerate(f_list):
            date = file.split('.')[0]
            node_idxs = np.arange(self.node_num)
            np.random.shuffle(node_idxs)
            test_num = int(np.floor(self.node_num * self.test_ratio))
            val_num = int(np.floor(self.node_num * self.val_ratio))
            # train_num = self.node_num - test_num - val_num

            val_nodes = node_arr[node_idxs[: val_num]]
            test_nodes = node_arr[node_idxs[val_num : val_num + test_num]]
            train_nodes = node_arr[node_idxs[val_num + test_num: ]]

            val_labels = label_arr[node_idxs[: val_num]]
            test_labels = label_arr[node_idxs[val_num : val_num + test_num]]
            train_labels = label_arr[node_idxs[val_num + test_num: ]]

            train_output_path = os.path.join(self.output_base_path, date + '_train.csv')
            df_train = pd.DataFrame({'node': train_nodes, 'label': train_labels})
            df_train.to_csv(train_output_path, sep='\t', index=False)

            test_output_path = os.path.join(self.output_base_path, date + '_test.csv')
            df_test = pd.DataFrame({'node': test_nodes, 'label': test_labels})
            df_test.to_csv(test_output_path, sep='\t', index=False)

            val_output_path = os.path.join(self.output_base_path, date + '_val.csv')
            df_val = pd.DataFrame({'node': val_nodes, 'label': val_labels})
            df_val.to_csv(val_output_path, sep='\t', index=False)
        print('Generate node samples finish!')

class NodeClassifier(object):
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    nodeclas_base_path: str
    output_base_path: str
    full_node_list: list
    label_list: list

    def __init__(self, base_path, origin_folder, embedding_folder, nodeclas_folder, output_folder, node_file, trans_label_file):
        self.base_path = base_path
        self.origin_base_path = os.path.join(base_path, origin_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.nodeclas_base_path = os.path.join(base_path, nodeclas_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        df_label = pd.read_csv(os.path.join(base_path, trans_label_file), sep='\t')
        self.label_list = df_label['label'].tolist()

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)
        return

    def train(self, train_nodes, val_nodes, embeddings, lb):
        #print('Start training!')
        train_feature = embeddings[train_nodes[:, 0], :]
        val_feature = embeddings[val_nodes[:, 0], :]
        train_labels = train_nodes[:, 1]
        val_labels = val_nodes[:, 1]

        train_labels = lb.transform(train_labels)
        val_labels = lb.transform(val_labels)

        models = []
        for C in [0.01, 0.1, 1, 5, 10, 20]:
            lr = LogisticRegression(C=C, solver='lbfgs', max_iter=5000, class_weight='balanced')
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

    def test(self, test_nodes, embeddings, model, lb, date):
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
            train_nodes = pd.read_csv(os.path.join(self.nodeclas_base_path, date + '_train.csv'), sep='\t').values
            val_nodes = pd.read_csv(os.path.join(self.nodeclas_base_path, date + '_val.csv'), sep='\t').values
            test_nodes = pd.read_csv(os.path.join(self.nodeclas_base_path, date + '_test.csv'), sep='\t').values
            if not os.path.exists(os.path.join(self.embedding_base_path, method, f_name)):
                continue
            df_embedding = pd.read_csv(os.path.join(self.embedding_base_path, method, f_name), sep='\t', index_col=0)
            df_embedding  = df_embedding.loc[self.full_node_list]
            embeddings = df_embedding.values

            lb = preprocessing.LabelBinarizer()
            lb.fit(self.label_list)
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

def process_result(dataset, rep_num, method_list):

    for method in method_list:
        base_path = os.path.join('../../data/' + dataset, 'node_classification_res_0')
        res_path = os.path.join(base_path, method + '_acc_record.csv')
        df_method = pd.read_csv(res_path, sep=',', header=0, names=['date', 'acc0'])
        for i in range(1, rep_num):
            base_path = os.path.join('../../data/' + dataset, 'node_classification_res_' + str(i))
            res_path = os.path.join(base_path, method + '_acc_record.csv')
            df_rep = pd.read_csv(res_path, sep=',', header=0, names=['date', 'acc' + str(i)])
            df_method = pd.concat([df_method, df_rep.iloc[:, [1]]], axis=1)
        output_base_path = os.path.join('../../data/' + dataset, 'node_classification_res')
        check_and_make_path(output_base_path)
        acc_list = ['acc' + str(i) for i in range(rep_num)]
        df_method['avg'] = df_method.loc[:, acc_list].mean(axis=1)
        df_method['max'] = df_method.loc[:, acc_list].max(axis=1)
        df_method['min'] = df_method.loc[:, acc_list].min(axis=1)
        output_path = os.path.join(output_base_path, method + '_acc_record.csv')
        df_method.to_csv(output_path, sep=',', index=False)

if __name__ == '__main__':
    dataset = 'america_air'
    rep_num = 20

    # method_list = ['deepwalk', 'node2vec', 'struct2vec', 'GCN', 'dyGEM', 'timers', 'EvolveGCNH', 'EvolveGCNO']

    # prob_list = [0]
    # for prob in prob_list:
    #     method_list.append('RWTGCN_prob_' + str(prob))
    method_list = ['RWTGCN_S']

    t1 = time.time()
    for i in range(rep_num):
        data_generator = DataGenerator(base_path="../../data/" + dataset, input_folder="1.format",
                                       output_folder="node_classification_data_" + str(i), node_file="nodes_set/nodes.csv", label_file="nodes_set/labels.csv",
                                       trans_label_file="nodes_set/trans_label.csv")
        # data_generator.generate_node_samples()

        node_classifier = NodeClassifier(base_path="../../data/" + dataset, origin_folder='1.format', embedding_folder="2.embedding",
                                         nodeclas_folder="node_classification_data_" + str(i), output_folder="node_classification_res_" + str(i), node_file="nodes_set/nodes.csv",
                                         trans_label_file="nodes_set/trans_label.csv")
        node_classifier.node_classification_all_method(method_list=method_list, worker=11)

    t2 = time.time()
    print('node classification cost time: ', t2 - t1, ' seconds!')

    process_result(dataset, rep_num, method_list)