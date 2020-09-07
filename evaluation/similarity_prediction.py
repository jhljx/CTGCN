# coding: utf-8
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import os
import time
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from utils import check_and_make_path, get_sp_adj_mat, sigmoid
from scipy.spatial.distance import pdist, squareform


# Generate data used for structural similarity prediction
class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    file_sep: str
    full_node_list: list
    node_num: int
    alpha: float
    iter_num: int

    def __init__(self, base_path, input_folder, output_folder, node_file, file_sep='\t', alpha=0.5, iter_num=100):
        self.base_path = base_path
        self.input_base_path = os.path.abspath(os.path.join(base_path, input_folder))
        self.output_base_path = os.path.abspath(os.path.join(base_path, output_folder))
        self.file_sep = file_sep

        node_file_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_file_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)
        self.alpha = alpha
        self.iter_num = iter_num
        assert 0 < self.alpha < 1

        check_and_make_path(self.input_base_path)
        check_and_make_path(self.output_base_path)

    def generate_node_similarity(self, file):
        """the implement of Vertex similarity in networks"""
        #  Vertex similarity in networks(https://arxiv.org/abs/physics/0510143)
        print('file = ', file)
        file_path = os.path.join(self.input_base_path, file)
        date = file.split('.')[0]
        output_file_path = os.path.join(self.output_base_path, date + '_similarity.npz')
        A = get_sp_adj_mat(file_path, self.full_node_list, sep=self.file_sep)
        A = A.tocsr()
        lambda_1 = scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
        print('lambda 1: ', lambda_1)
        rows, cols = A.nonzero()
        edge_num = rows.shape[0]
        n = A.shape[0]
        d = np.array(A.sum(1)).flatten()
        d_inv = np.zeros(n)  # dtype is float
        indices = np.where(d > 0)[0]
        d_inv[indices] = 1. / d[indices]
        d_inv = np.diag(d_inv)
        # dsd = np.random.normal(0, 1 / np.sqrt(n), (n, n))
        dsd = np.zeros((n, n))
        I = np.eye(n)
        for i in range(self.iter_num):
            # if i == 0:
            #      dsd = self.alpha / lambda_1 * A
            # else:
            #      dsd = self.alpha / lambda_1 * A + self.alpha / lambda_1 * dsd.dot(A)
            dsd = self.alpha / lambda_1 * A.dot(dsd) + I
            if i % 10 == 0:
                print('VS', i, '/', self.iter_num)
        # coeff = 2 * edge_num * lambda_1
        # S = d_inv.dot(dsd).dot(d_inv)
        S = dsd
        S = (S + S.T) / 2
        S = S - I
        S = (S - S.min()) / (S.max() - S.min())
        print(type(S))
        print('S max: ', S.max(), ', min: ', S.min())
        eps = 1e-6
        S[S < eps] = 0
        # S[S > 1] = 1
        S = sp.coo_matrix(S)
        sp.save_npz(output_file_path, S)
        # exit(0)

    def generate_node_similarity_all_time(self, worker=-1):
        f_list = sorted(os.listdir(self.input_base_path))
        length = len(f_list)

        if worker <= 0:
            for i, file in enumerate(f_list):
                self.generate_node_similarity(file)
        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("start " + str(worker) + " worker(s)")

            for i, file in enumerate(f_list):
                pool.apply_async(self.generate_node_similarity, (file, ))
            pool.close()
            pool.join()


# Centrality predictor class
class SimilarityPredictor(object):
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    similarity_base_path: str
    output_base_path: str
    file_sep: str
    full_node_list: list

    def __init__(self, base_path, origin_folder, embedding_folder, similarity_folder, output_folder, node_file, file_sep='\t'):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        self.similarity_base_path = os.path.abspath(os.path.join(base_path, similarity_folder))
        self.output_base_path = os.path.abspath(os.path.join(base_path, output_folder))
        self.file_sep = file_sep

        node_file_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_file_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.origin_base_path)
        check_and_make_path(self.output_base_path)


    def get_prediction_error(self, method, node_sim_mat, embedding_mat, date):
        mse_list = [date]
        # node_sim_mat = node_sim_mat / node_sim_mat.sum()
        # node_sim = pd.Series(node_sim_mat.flatten())
        # node_sim_mat = sigmoid(node_sim_mat)
        #pred_sim_mat = pred_sim_mat / pred_sim_mat.sum()
        #print('node sim max:', node_sim_mat.max(), 'pred sim max: ', pred_sim_mat.max())
        #print('node sim min: ', node_sim_mat.min(), 'pred sim min: ', pred_sim_mat.min())
        #print('node sim avg: ', node_sim_mat.mean(), 'pred sim avg: ', pred_sim_mat.mean())
        # pred_sim_mat = pred_sim_mat / pred_sim_mat.sum()
        #embedding_mat = (embedding_mat - embedding_mat.min()) / (embedding_mat.max() - embedding_mat.min())
        # print('pred sim: ', pred_sim_mat.max(), pred_sim_mat.min())
        pred_sim_mat = embedding_mat.dot(embedding_mat.T)
        # node_sim_mat = (node_sim_mat - node_sim_mat.min()) / (node_sim_mat.max() - node_sim_mat.min())
        # pred_sim_mat = (pred_sim_mat - pred_sim_mat.min()) / (pred_sim_mat.max() - pred_sim_mat.min())
        #pred_sim_mat = pred_sim_mat / pred_sim_mat.sum()
        # pred_sim = pd.Series(pred_sim_mat.flatten())
        # pred_sim_mat = sigmoid(pred_sim_mat)
        #print(node_sim_mat.mean(), pred_sim_mat.mean())
        # exit(0)
        # np.savetxt(os.path.join(self.base_path, date + 'pred_sim.txt'), pred_sim_mat)
        # error = mean_squared_error(node_sim_mat, pred_sim_mat)
        eps = 1e-6
        column_list = []
        n = pred_sim_mat.shape[0]
        for i in range(n):
            if node_sim_mat[i].sum() < eps:  # node is single node whose degree is 0
                continue
            column_list.append(i)
        real_sim_mat = node_sim_mat[column_list, :][:, column_list]
        real_sim_mat = (real_sim_mat - real_sim_mat.min()) / (real_sim_mat.max() - real_sim_mat.min())
        real_sim_mat = real_sim_mat / real_sim_mat.sum()
        real_sim = pd.Series(real_sim_mat.flatten())
        pred_sim_mat = pred_sim_mat[column_list, :][:, column_list]
        pred_sim_mat = (pred_sim_mat - pred_sim_mat.min()) / (pred_sim_mat.max() - pred_sim_mat.min())
        pred_sim_mat = pred_sim_mat / pred_sim_mat.sum()
        pred_sim = pd.Series(pred_sim_mat.flatten())
        print('real sim min: ', real_sim.min(), 'max: ', real_sim.max(), 'avg: ', real_sim.mean())
        print('pred sim min: ', pred_sim.min(), 'max: ', pred_sim.max(), 'avg: ', pred_sim.mean())
        # import ot
        from scipy.stats import entropy

        #connected_node_num = len(column_list)
        #print('connected node number: ', connected_node_num)
        mse_list.append(real_sim.corr(pred_sim, method='spearman'))
        # mse_list.append(mutual_info_score(node_sim, pred_sim))
        return mse_list

    def similarity_prediction_all_time(self, method):
        print('method = ', method)
        f_list = sorted(os.listdir(self.origin_base_path))
        all_mse_list = []
        for i, f_name in enumerate(f_list):
            print('Current date is: {}'.format(f_name))
            date = f_name.split('.')[0]
            node_sim_mat = np.loadtxt(os.path.join(self.similarity_base_path, date + '_similarity.csv'))
            cur_embedding_path = os.path.join(self.embedding_base_path, method, f_name)
            if not os.path.exists(cur_embedding_path):
                continue
            df_embedding = pd.read_csv(cur_embedding_path, sep=self.file_sep, index_col=0)
            df_embedding = df_embedding.loc[self.full_node_list]
            embedding_mat = df_embedding.values
            mse_list = self.get_prediction_error(method, node_sim_mat, embedding_mat, date)
            all_mse_list.append(mse_list)

        df_output = pd.DataFrame(all_mse_list, columns=['date', 'mse'])
        print(df_output)
        output_file_path = os.path.join(self.output_base_path, method + '_mse_record.csv')
        df_output.to_csv(output_file_path, sep=',', index=False)

    def similarity_prediction_all_method(self, method_list=None, worker=-1):
        print('Start node similarity prediction!')
        if method_list is None:
            method_list = os.listdir(self.embedding_base_path)

        if worker <= 0:
            for method in method_list:
                print('Current method is :{}'.format(method))
                self.similarity_prediction_all_time(method)
        else:
            worker = min(worker, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("start " + str(worker) + " worker(s)")

            for method in method_list:
                pool.apply_async(self.similarity_prediction_all_time, (method,))
            pool.close()
            pool.join()
        print('Finish node similarity prediction!')


def similarity_prediction(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    node_file = args['node_file']
    similarity_data_folder = args['similarity_data_folder']
    similarity_res_folder = args['similarity_res_folder']
    file_sep = args['file_sep']
    generate = args['generate']
    method_list = args['method_list']
    alpha = args['alpha']
    iter_num = args['iter_num']
    worker = args['worker']

    data_generator = DataGenerator(base_path=base_path, input_folder=origin_folder, output_folder=similarity_data_folder, node_file=node_file, file_sep=file_sep,
                                   alpha=alpha, iter_num=iter_num)
    if generate:
        data_generator.generate_node_similarity_all_time(worker=worker)
    similarity_predictor = SimilarityPredictor(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, similarity_folder=similarity_data_folder,
                                               output_folder=similarity_res_folder, node_file=node_file, file_sep=file_sep)
    t1 = time.time()
    # similarity_predictor.similarity_prediction_all_method(method_list=method_list, worker=worker)
    t2 = time.time()
    print('node similarity prediction cost time: ', t2 - t1, ' seconds!')