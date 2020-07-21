import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import torch


# Check the existence of directory(file) path, if not, create one
def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)


# Get networkx graph object from file path. If the graph is unweighted, then add the 'weight' attribute
def get_nx_graph(file_path, full_node_list, sep='\t'):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num)))
    df = pd.read_csv(file_path, sep=sep)
    if df.shape[1] == 2:
        df['weight'] = 1.0
    df['from_id'] = df['from_id'].apply(lambda x: node2idx_dict[x])
    df['to_id'] = df['to_id'].apply(lambda x: node2idx_dict[x])
    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
    graph.add_nodes_from(np.arange(node_num))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# Get sparse.lil_matrix type adjacent matrix
# Note that if you want to use this function, please transform the sparse matrix type, i.e. sparse.coo_matrix, sparse.csc_matrix if needed!
def get_sp_adj_mat(file_path, full_node_list, sep='\t', weight_flag=False):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num)))
    A = sp.lil_matrix((node_num, node_num))
    with open(file_path, 'r') as fp:
        content_list = fp.readlines()
        # ignore header
        for line in content_list[1:]:
            line_list = line.split(sep)
            from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
            from_id = node2idx_dict[from_node]
            to_id = node2idx_dict[to_node]
            # remove self-loop data
            if from_id == to_id:
                continue
            if not weight_flag:
                A[from_id, to_id] = 1
                A[to_id, from_id] = 1
            else:
                A[from_id, to_id] = weight
                A[to_id, from_id] = weight
    return A


# Generate a row-normalized sparse matrix
def normalize(mat):
    rowsum = np.array(mat.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1 / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = sp.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat


# Calculate accuracy of 1-dimension prediction result and its corresponding label
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# Transform a sparse matrix into a torch.sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
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


# Generate a row-normalized adjacent matrix from a sparse matrix
# If add_eye=True, then the renormalization trick would be used.
# For the renormalization trick, please refer to the "Semi-supervised Classification with Graph Convolutional Networks" paper,
# The paper can be viewed in https://arxiv.org/abs/1609.02907
def get_normalize_adj(spmat, add_eye=True):
    if add_eye:
        adj = normalize(spmat + sp.eye(spmat.shape[0]))
    else:
        adj = normalize(spmat)
    adj = adj.tocoo()
    return adj

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