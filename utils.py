# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import torch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Check the existence of directory(file) path, if not, create one
def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)


# Get networkx graph object from file path. If the graph is unweighted, then add the 'weight' attribute
def get_nx_graph(file_path, full_node_list, sep='\t'):
    df = pd.read_csv(file_path, sep=sep)
    if df.shape[1] == 2:
        df['weight'] = 1.0
    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
    graph.add_nodes_from(full_node_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# Get sparse.lil_matrix type adjacent matrix
# Note that if you want to use this function, please transform the sparse matrix type, i.e. sparse.coo_matrix, sparse.csc_matrix if needed!
def get_sp_adj_mat(file_path, full_node_list, sep='\t'):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num)))
    A = sp.lil_matrix((node_num, node_num))
    with open(file_path, 'r') as fp:
        content_list = fp.readlines()
        # ignore header
        for line in content_list[1:]:
            line_list = line.split(sep)
            col_num = len(line_list)
            assert col_num in [2, 3]
            if col_num == 2:
                from_node, to_node, weight = line_list[0], line_list[1], 1
            else:
                from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
            from_id = node2idx_dict[from_node]
            to_id = node2idx_dict[to_node]
            # remove self-loop data
            if from_id == to_id:
                continue
            A[from_id, to_id] = weight
            A[to_id, from_id] = weight
    A = A.tocoo()
    return A


# Generate a row-normalized adjacent matrix from a sparse matrix
# If add_eye=True, then the renormalization trick would be used.
# For the renormalization trick, please refer to the "Semi-supervised Classification with Graph Convolutional Networks" paper,
# The paper can be viewed in https://arxiv.org/abs/1609.02907
def get_normalized_adj(adj, row_norm=False):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    p = -1 if row_norm else -0.5

    def inv(x, p):
        if p >= 0:
            return np.power(x, p)
        if x == 0:
            return x
        if x < 0:
            raise ValueError('invalid value encountered in power, x is negative, p is negative!')
        return np.power(x, p)
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum, p).flatten()
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    if not row_norm:
        adj = adj.dot(r_mat_inv)
    adj = adj.tocoo()
    return adj


# Transform a sparse matrix into a torch.sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Transform a sparse matrix into a tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# Generate negative links(edges) in a graph
def get_neg_edge_samples(pos_edges, edge_num, all_edge_dict, node_num, add_label=True):
    neg_edge_dict = dict()
    neg_edge_list = []
    cnt = 0
    while cnt < edge_num:
        from_id = np.random.choice(node_num)
        to_id = np.random.choice(node_num)
        if from_id == to_id:
            continue
        if (from_id, to_id) in all_edge_dict or (to_id, from_id) in all_edge_dict:
            continue
        if (from_id, to_id) in neg_edge_dict or (to_id, from_id) in neg_edge_dict:
            continue
        if add_label:
            neg_edge_list.append([from_id, to_id, 0])
        else:
            neg_edge_list.append([from_id, to_id])
        cnt += 1
    neg_edges = np.array(neg_edge_list)
    all_edges = np.vstack([pos_edges, neg_edges])
    return all_edges


# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# Get formatted number str, which is used for dynamic graph(or embedding) file name. File order is important!
def get_format_str(cnt):
    max_bit = 0
    while cnt > 0:
        cnt //= 10
        max_bit += 1
    format_str = '{:0>' + str(max_bit) + 'd}'
    return format_str


# Print separate lines
def separate(info='', sep='=', num=8):
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)


def get_static_gnn_methods():
    gnn_list = ['GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN', 'CGCN-C', 'CGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_dynamic_gnn_methods():
    gnn_list = ['GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_core_based_methods():
    gnn_list = ['CGCN-C', 'CGCN-S', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_supported_gnn_methods():
    gnn_list = ['GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN', 'CGCN-C', 'CGCN-S', 'GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_supported_methods():
    method_list = ['DynGEM', 'DynAE', 'DynRNN', 'DynAERNN', 'TIMERS', 'GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN',
                   'CGCN-C', 'CGCN-S', 'GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(method_list, np.ones(len(method_list), dtype=np.int)))
