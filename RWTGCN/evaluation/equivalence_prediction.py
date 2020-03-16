import numpy as np
import pandas as pd
import os, time, sys, multiprocessing
import networkx as nx
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
sys.path.append("..")
from RWTGCN.utils import check_and_make_path

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    label_list: list
    node_num: int

    def __init__(self, base_path, input_folder, output_folder, node_file):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)
        return

    def get_centrality(self, network, type='degree', undirected=True):
        if type == 'degree':
            if undirected:
                return nx.degree_centrality(network)
            else:
                return nx.in_degree_centrality(network)
        elif type == 'closeness':
            return nx.closeness_centrality(network)
        elif type == 'betweenness':
            return nx.betweenness_centrality(network)
        elif type == 'eigenvector':
            return nx.eigenvector_centrality(network)
        elif type == 'kcore':
            return nx.core_number(network)
        else:
            raise Exception('Unsupported centrality type!')

    def generate_node_samples(self, file, sep='\t'):
        centrality_list = ['closeness', 'betweenness', 'eigenvector', 'kcore']

        file_path = os.path.join(self.input_base_path, file)
        date = file.split('.')[0]
        if os.path.exists(os.path.join(self.output_base_path, date + '_centrality.csv')):
            print('\t', date + '_centrality.csv exist')
            return
        print('start generating', date + '_centrality.csv')

        t1 =time.time()
        df = pd.read_csv(file_path, sep=sep)
        if df.shape[1] == 2:
            df['weight'] = 1.0
        graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
        graph.add_nodes_from(self.full_node_list)
        centrality_dict = dict()
        for centrality in centrality_list:
            centrality_dict[centrality] = self.get_centrality(graph, type=centrality)
        centrality_res = []
        for nidx in range(self.node_num):
            node = self.full_node_list[nidx]
            node_res = [nidx]
            for centrality in centrality_list:
                node_res.append(centrality_dict[centrality][node])
            centrality_res.append(node_res)
        df_centrality = pd.DataFrame(centrality_res, columns=['node', 'closeness', 'betweenness', 'eigenvector', 'kcore'])
        output_path = os.path.join(self.output_base_path, date + '_centrality.csv')
        df_centrality.to_csv(output_path, sep='\t', index=False)
        t2 = time.time()
        print('finish generating', date + '_centrality.csv')
        print('cost time: ', t2 - t1, ' seconds!')

    def generate_all_node_samples(self, sep='\t', worker=-1):
        f_list = sorted(os.listdir(self.input_base_path))
        length = len(f_list)

        if worker <= 0:
            for i, file in enumerate(f_list):
                self.generate_node_samples(file, sep=sep)
        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("start " + str(worker) + " worker(s)")

            for i, file in enumerate(f_list):
                pool.apply_async(self.generate_node_samples, (file, sep, ))
            pool.close()
            pool.join()

class EquivalencePredictor(object):
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    equ_base_path: str
    output_base_path: str
    full_node_list: list

    def __init__(self, base_path, origin_folder, embedding_folder, equ_folder, output_folder, node_file):
        self.base_path = base_path
        self.origin_base_path = os.path.join(base_path, origin_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.equ_base_path = os.path.join(base_path, equ_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)
        return

    def get_prediction_error(self, centrality_data, embeddings, date):
        centrality_list = ['closeness', 'betweenness', 'eigenvector', 'kcore']
        mse_list = [date]
        for i, centrality in enumerate(centrality_list):
            # model = LinearRegression(n_jobs=-1)
            # min_error = float("inf")
            # # only using a certain cv value may cause some method have very large MSE, so we need to try different cv values
            # # for cv in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]:
            # y_pred = cross_val_predict(model, embeddings, centrality_data[:, i], cv=5)
            # error = mean_squared_error(centrality_data[:, i], y_pred) / np.mean(centrality_data[:, i])
            # min_error = min(min_error, error)
            min_error = float("inf")
            for alpha in [0.05, 0.5, 1, 2, 5, 10]:
                model = Ridge(alpha=alpha)
                y_pred = cross_val_predict(model, embeddings, centrality_data[:, i], cv=5)
                error = mean_squared_error(centrality_data[:, i], y_pred) / np.mean(centrality_data[:, i])
                min_error = min(min_error, error)
            mse_list.append(min_error)
        return mse_list

    def equivalence_prediction_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.origin_base_path))
        all_mse_list = []
        for i, f_name in enumerate(f_list):
            print('Current date is: {}'.format(f_name))
            date = f_name.split('.')[0]
            df_centrality = pd.read_csv(os.path.join(self.equ_base_path, date + '_centrality.csv'), sep='\t')
            centrality_data= df_centrality.iloc[:, 1:].values
            if not os.path.exists(os.path.join(self.embedding_base_path, method, f_name)):
                continue
            df_embedding = pd.read_csv(os.path.join(self.embedding_base_path, method, f_name), sep='\t', index_col=0)
            df_embedding  = df_embedding.loc[self.full_node_list]
            embeddings = df_embedding.values
            mse_list = self.get_prediction_error(centrality_data, embeddings, date)
            all_mse_list.append(mse_list)

        df_output = pd.DataFrame(all_mse_list, columns=['date', 'closeness', 'betweenness', 'eigenvector', 'kcore'])
        print(df_output)
        print('closeness avg: ', df_output['closeness'].mean())
        print('betweenness avg: ', df_output['betweenness'].mean())
        print('eigenvector avg: ', df_output['eigenvector'].mean())
        print('kcore avg: ', df_output['kcore'].mean())
        output_file_path = os.path.join(self.output_base_path, method + '_mse_record.csv')
        df_output.to_csv(output_file_path, sep=',', index=False)

    def equivalence_prediction_all_method(self, method_list=None, worker=-1):
        print('Start equivalence_prediction!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)

        if worker <= 0:
            for method in method_list:
                print('Current method is :{}'.format(method))
                self.equivalence_prediction_all_time(method)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("start " + str(worker) + " worker(s)")

            for method in method_list:
                pool.apply_async(self.equivalence_prediction_all_time, (method,))
            pool.close()
            pool.join()
        print('Finish equivalence_prediction!')


if __name__ == '__main__':
    dataset = 'blogcatalog'
    data_generator = DataGenerator(base_path="../../data/" + dataset, input_folder="1.format",
                                   output_folder="equ_prediction_data", node_file="nodes_set/nodes.csv")
    # data_generator.generate_all_node_samples(worker=10)

    equivalence_predictor = EquivalencePredictor(base_path="../../data/" + dataset, origin_folder='1.format', embedding_folder="2.embedding",
                                   equ_folder="equ_prediction_data", output_folder="equ_prediction_res", node_file="nodes_set/nodes.csv")

    # method_list = ['deepwalk', 'node2vec', 'struct2vec', 'GCN', 'dyGEM', 'timers', 'EvolveGCNH']
    method_list = ['RWTGCN_C', 'RWTGCN_S']

    t1 = time.time()
    equivalence_predictor.equivalence_prediction_all_method(method_list=method_list, worker=10)
    t2 = time.time()
    print('equivalence prediction cost time: ', t2 - t1, ' seconds!')