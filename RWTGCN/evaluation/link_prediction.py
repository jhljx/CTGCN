import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from RWTGCN.utils import check_and_make_path

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list

    def __init__(self, base_path, input_folder, output_folder, node_file):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)
        return

    def generate_edge_samples(self):
        print('Start generating edge samples!')
        node_num = len(self.full_node_list)
        node2idx_dict = dict(zip(self.full_node_list, np.arange(node_num).tolist()))

        f_list = os.listdir(self.input_base_path)
        for file in f_list:
            file_path = os.path.join(self.input_base_path, file)
            graph_dict = dict(zip(self.full_node_list, [{}] * node_num))

            with open(file_path, 'r') as fp:
                content_list = fp.readlines()
                for line in content_list[1:]:
                    edge = line.strip().split('\t')
                    from_id = node2idx_dict[edge[0]]
                    to_id = node2idx_dict[edge[1]]
                    graph_dict[from_id][to_id] = 1
                    graph_dict[to_id][from_id] = 1
                output_list = []
                for from_id, neighbor_dict in graph_dict.items():
                    for to_id, val in neighbor_dict.items():
                        output_list.append([from_id, to_id, 1])
                        for k in range(10):
                            sample_id = np.random.choice(node_num)
                            if sample_id not in neighbor_dict:
                                output_list.append([from_id, sample_id, 0])
                                break
                output_path = os.path.join(self.output_base_path, file)
            df_output = pd.DataFrame(output_list, columns=['from_id', 'to_id', 'label'])
            df_output.to_csv(output_path, sep=',', index=False)
        print('Generate edge samples finish!')


class LinkPredictor(object):
    base_path: str
    embedding_base_path: str
    edge_base_path: str
    output_base_path: str
    ratio: float

    def __init__(self, base_path, embedding_folder, edge_folder, output_folder, ratio=1.0):
        self.base_path = base_path
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.edge_base_path = os.path.join(base_path, edge_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        self.ratio = ratio

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.edge_base_path)
        check_and_make_path(self.output_base_path)
        return

    def get_edge_feature(self, edge_arr, embedding_arr):
        avg_edges, had_edges, l1_edges, l2_edges = np.array([]), np.array([]), np.array([]), np.array([])
        for i, edge in enumerate(edge_arr):
            from_id, to_id = edge[0], edge[1]
            avg_edges = np.concatenate((avg_edges, (embedding_arr[from_id] + embedding_arr[to_id]) / 2), axis=0)
            had_edges = np.concatenate((had_edges, (embedding_arr[from_id] * embedding_arr[to_id])), axis=0)
            l1_edges = np.concatenate((l1_edges, np.abs(embedding_arr[from_id] - embedding_arr[to_id])), axis=0)
            l2_edges = np.concatenate((l2_edges, (embedding_arr[from_id] - embedding_arr[to_id]) ** 2), axis=0)
        feature_dict = {'Avg': avg_edges, 'Had': had_edges, 'L1': l1_edges, 'L2': l2_edges}
        return feature_dict

    def train(self, edge_arr, embedding_arr):
        print('Start training!')
        model_dict = {}

        edge_num = edge_arr.shape[0]
        if self.ratio < 1.0:
            sample_num = int(edge_num * self.ratio)
            sampled_idxs = np.random.choice(np.arange(edge_num), sample_num).tolist()
            edge_arr = edge_arr[sampled_idxs, :]
        labels = edge_arr[:, 2]

        feature_dict = self.get_edge_feature(edge_arr, embedding_arr)
        measure_list = ['Avg', 'Had', 'L1', 'L2']
        for measure in measure_list:
            model_dict[measure] = LogisticRegression()
            model_dict[measure].fit(feature_dict[measure], labels)
        print('Finish training!')
        return model_dict

    def test(self, edge_arr, embedding_arr, model_dict, date):
        print('Start testing!')
        labels = edge_arr[:, 2]
        feature_dict = self.get_edge_feature(edge_arr, embedding_arr)
        auc_list = [date]
        measure_list = ['Avg', 'Had', 'L1', 'L2']
        for measure in measure_list:
            pred = model_dict[measure].predict_proba(feature_dict[measure])[:, 1]
            auc_list.append(roc_auc_score(labels, pred))
        print('Finish testing!')
        return auc_list

    def link_prediction_all_time(self, method_list=None):
        print('Start link prediction!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)
        for method in method_list:
            print('Current method is :{}'.format(method))
            f_list = os.listdir(self.edge_base_path)
            f_num = len(f_list)

            model_dict = dict()
            all_auc_list = []
            for i, f_name in enumerate(f_list):
                print('Current date is :{}'.format(f_name))
                date = f_name.split('.')[0]
                edge_arr = pd.read_csv(os.path.join(self.edge_base_path, f_name)).values
                embedding_arr = pd.read_csv(os.path.join(self.embedding_base_path, method, f_name)).values
                if i > 0:
                    auc_list = self.test(edge_arr, embedding_arr, model_dict, date)
                    all_auc_list.append(auc_list)
                if i < f_num - 1:
                    model_dict = self.train(edge_arr, embedding_arr)
            df_output = pd.DataFrame(all_auc_list, columns=['date', 'Avg', 'Had', 'L1', 'L2'])
            output_file_path = os.path.join(self.output_base_path, method + '_auc_record.csv')
            df_output.to_csv(output_file_path, sep=',', index=False)
        print('Finish link prediction!')


if __name__ == '__main__':
    data_set_name = 'enron_test'
    data_generator = DataGenerator(base_path="../../data/facebook", input_folder="1.format",
                                   output_folder="link_prediction_data", node_file="nodes_set/nodes.csv")
    data_generator.generate_edge_samples()

    link_predictor = LinkPredictor(base_path="../../data/facebook", embedding_folder="2.embedding",
                                   edge_folder="link_prediction_data", output_folder="link_prediction", ratio=1.0)
    method_list = ['deepwalk', 'node2vec', 'struc2vec', 'dyGEM', 'timers', 'rwtgcn']
    link_predictor.link_prediction_all_time(method_list=None)