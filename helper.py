# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import json
import torch
from utils import get_normalized_adj, get_sp_adj_mat, sparse_mx_to_torch_sparse_tensor


# A class which is designed for loading various kinds of data
class DataLoader:
    max_time_num: int
    full_node_list: list
    node2idx_dict: dict
    node_num: int
    has_cuda: bool

    def __init__(self, node_list, max_time_num, has_cuda=False):
        self.max_time_num = max_time_num
        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)
        self.node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num)))
        self.has_cuda = has_cuda

    # get adjacent matrices for a graph list, this function supports Tensor type-based adj and sparse.coo type-based adj.
    def get_date_adj_list(self, origin_base_path, start_idx, duration, sep='\t', normalize=False, row_norm=False, add_eye=False, data_type='tensor'):
        assert data_type in ['tensor', 'matrix']
        date_dir_list = sorted(os.listdir(origin_base_path))
        # print('adj list: ', date_dir_list)
        date_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            original_graph_path = os.path.join(origin_base_path, date_dir_list[i])
            spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep=sep)
            # spmat = sp.coo_matrix((np.exp(alpha * spmat.data), (spmat.row, spmat.col)), shape=(self.node_num, self.node_num))
            if add_eye:
                spmat = spmat + sp.eye(spmat.shape[0])
            if normalize:
                spmat = get_normalized_adj(spmat, row_norm=row_norm)
            # data type
            if data_type == 'tensor':
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                date_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            else:  # data_type == matrix
                date_adj_list.append(spmat)
        # print(len(date_adj_list))
        return date_adj_list

    # get k-core sub-graph adjacent matrices for a graph list, it is a 2-layer nested list, outer layer for graph, inner layer for k-cores.
    # k-core subgraphs will be automatically normalized by 'renormalization trick'(add_eye=True)
    def get_core_adj_list(self, core_base_path, start_idx, duration, max_core=-1):
        date_dir_list = sorted(os.listdir(core_base_path))
        time_stamp_num = len(date_dir_list)
        assert start_idx < time_stamp_num
        core_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            date_dir_path = os.path.join(core_base_path, date_dir_list[i])
            f_list = sorted(os.listdir(date_dir_path))
            core_file_num = len(f_list)
            tmp_adj_list = []
            if max_core == -1:
                max_core = core_file_num
            f_list = f_list[:max_core]  # select 1 core to max core
            f_list = f_list[::-1]  # reverse order, max core, (max - 1) core, ..., 1 core

            # get k-core adjacent matrices at the i-th timestamp
            spmat_list = []
            for j, f_name in enumerate(f_list):
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                spmat_list.append(spmat)
                if j == 0:
                    spmat = spmat + sp.eye(spmat.shape[0])
                else:
                    delta = spmat - spmat_list[j - 1]    # reduce subsequent computation complexity and reduce memory cost!
                    if delta.sum() == 0:  # reduce computation complexity and memory cost!
                        continue
                # Normalization will reduce the self weight, hence affect its performance! So we omit normalization.
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                tmp_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            # print('time: ', i, 'core len: ', len(tmp_adj_list))
            core_adj_list.append(tmp_adj_list)
        return core_adj_list

    # get node co-occurrence pairs of random walk for a graph list, the node pair list is used for negative sampling
    def get_node_pair_list(self, walk_pair_base_path, start_idx, duration):
        walk_file_list = sorted(os.listdir(walk_pair_base_path))
        # print('walk file list: ', walk_file_list)
        node_pair_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            walk_file_path = os.path.join(walk_pair_base_path, walk_file_list[i])
            walk_spadj = sp.load_npz(walk_file_path)
            neighbor_arr = walk_spadj.tolil().rows
            node_pair_list.append(neighbor_arr)
        return node_pair_list

    # get node frequencies of random walk for a graph list, the node frequency list is used for negative sampling
    def get_node_freq_list(self, node_freq_base_path, start_idx, duration):
        freq_file_list = sorted(os.listdir(node_freq_base_path))
        # print('node freq list: ', freq_file_list)
        node_freq_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            freq_file_path = os.path.join(node_freq_base_path, freq_file_list[i])
            with open(freq_file_path, 'r') as fp:
                node_freq_arr = json.load(fp)
                node_freq_list.append(node_freq_arr)
        return node_freq_list

    # load node features, use degree related features
    def get_degree_feature_list(self, origin_base_path, start_idx, duration, sep='\t', init_type='gaussian', std=1e-4):
        assert init_type in ['gaussian', 'adj', 'combine', 'one-hot']
        x_list = []
        max_degree = 0
        adj_list = []
        degree_list = []
        date_dir_list = sorted(os.listdir(origin_base_path))
        # find the maximal degree for a list of graphs
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            original_graph_path = os.path.join(origin_base_path, date_dir_list[i])
            adj = get_sp_adj_mat(original_graph_path, self.full_node_list, sep=sep)
            adj_list.append(adj)
            degrees = adj.sum(axis=1).astype(np.int)
            max_degree = max(max_degree, degrees.max())
            degree_list.append(degrees)
        # generate degree_based features
        input_dim = 0
        for i, degrees in enumerate(degree_list):
            # other structural feature initialization techniques can also be tried to improve performance
            if init_type == 'gaussian':
                fea_list = []
                for degree in degrees:
                    fea_list.append(np.random.normal(degree, std, max_degree + 1))
                fea_arr = np.array(fea_list).astype(np.float32)
                input_dim = fea_arr.shape[1]
                fea_tensor = torch.from_numpy(fea_arr).float()
                x_list.append(fea_tensor.cuda() if self.has_cuda else fea_tensor)
            elif init_type == 'adj':
                input_dim = self.node_num
                feat_tensor = sparse_mx_to_torch_sparse_tensor(adj_list[i])
                x_list.append(feat_tensor.cuda() if self.has_cuda else feat_tensor)
            elif init_type == 'combine':
                fea_list = []
                for degree in degrees:
                    fea_list.append(np.random.normal(degree, std, max_degree + 1))
                sp_feat = sp.coo_matrix(np.array(fea_list))
                sp_feat = sp.hstack((sp_feat, adj_list[i])).astype(np.float32)
                input_dim = sp_feat.shape[1]
                feat_tensor = sparse_mx_to_torch_sparse_tensor(sp_feat)
                x_list.append(feat_tensor.cuda() if self.has_cuda else feat_tensor)
            else:  # one-hot degree feature
                data = np.ones(degrees.shape[0], dtype=np.int)
                row = np.arange(degrees.shape[0])
                col = degrees.flatten().A[0]
                spmat = sp.csr_matrix((data, (row, col)), shape=(degrees.shape[0], max_degree + 1))
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                x_list.append(sptensor.cuda() if self.has_cuda else sptensor)
                # print('max degree: ', max_degree + 1)
                input_dim = max_degree + 1
        return x_list, input_dim

    # load node features from file, or create one-hot node feature
    def get_feature_list(self, feature_base_path, start_idx, duration, sep='\t', shuffle=False):
        if feature_base_path is None:
            x_list = []
            for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
                if shuffle:
                    node_indices = np.random.permutation(np.arange(self.node_num)) if shuffle else np.arange(self.node_num)
                    spmat = sp.coo_matrix((np.ones(self.node_num), (np.arange(self.node_num), node_indices)), shape=(self.node_num, self.node_num))
                else:
                    spmat = sp.eye(self.node_num)
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                x_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            input_dim = self.node_num
        else:
            feature_file_list = sorted(os.listdir(feature_base_path))
            x_list = []
            feature_arr_list = []
            max_feature_dim = 0
            # calculate max feature dimension
            for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
                feature_file_path = os.path.join(feature_base_path, feature_file_list[i])
                df_feature = pd.read_csv(feature_file_path, sep=sep, header=0)
                max_feature_dim = max(max_feature_dim, df_feature.shape[1])
                feature_arr = df_feature.values
                feature_arr_list.append(feature_arr)
            # expand feature matrix into the same dimension
            for feature_arr in feature_arr_list:
                batch_dim, feature_dim = feature_arr.shape
                expand_feature_arr = np.hstack((feature_arr, np.zeros((batch_dim, max_feature_dim - feature_dim)))).astype(np.float32)
                fea_tensor = torch.from_numpy(expand_feature_arr).float()
                x_list.append(fea_tensor.cuda() if self.has_cuda else fea_tensor)
            input_dim = max_feature_dim
        return x_list, input_dim

    def get_node_label_list(self, nlabel_base_path, start_idx, duration, sep='\t'):
        nlabel_file_list = sorted(os.listdir(nlabel_base_path))
        node_label_list = []
        label_dict = dict()
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            nlabel_file_path = os.path.join(nlabel_base_path, nlabel_file_list[i])
            df_nodes = pd.read_csv(nlabel_file_path, sep=sep, header=0, names=['node', 'label'])
            df_nodes['node'] = df_nodes['node'].apply(lambda x: self.node2idx_dict[x])
            unique_labels = df_nodes['label'].unique()
            for label in unique_labels:
                label_dict[label] = 1
            node_labels = torch.from_numpy(df_nodes.values).long()
            node_label_list.append(node_labels.cuda() if self.has_cuda else node_labels)
        return node_label_list, len(label_dict.keys())

    def get_edge_label_list(self, elabel_base_path, start_idx, duration, sep='\t'):
        elabel_file_list = sorted(os.listdir(elabel_base_path))
        edge_label_list = []
        label_dict = dict()
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            elabel_file_path = os.path.join(elabel_base_path, elabel_file_list[i])
            df_edges = pd.read_csv(elabel_file_path, sep=sep, header=0, names=['from_id', 'to_id', 'label'])
            df_edges[['from_id', 'to_id']] = df_edges[['from_id', 'to_id']].applymap(lambda x: self.node2idx_dict[x])
            unique_labels = df_edges['label'].unique()
            for label in unique_labels:
                label_dict[label] = 1
            edge_labels = torch.from_numpy(df_edges.values).long()
            edge_label_list.append(edge_labels.cuda() if self.has_cuda else edge_labels)
        return edge_label_list, len(label_dict.keys())
