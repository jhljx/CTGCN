import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os, json, time, random
import torch
from sympy import sieve

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)

def build_graph(file_path, full_node_list, sep='\t', weight_flag=False, save_weight=True):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num).tolist()))
    graph_dict = dict(zip(full_node_list, [{}] * node_num))
    with open(file_path, 'r') as fp:
        content_list = fp.readlines()
        # ignore header
        for line in content_list[1:]:
            line_list = line.split(sep)
            from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
            from_id = node2idx_dict[from_node]
            to_id = node2idx_dict[to_node]
            if not weight_flag:
                graph_dict[from_id][to_id] = 1
                graph_dict[to_id][from_id] = 1
            else:
                graph_dict[from_id][to_id] = max(graph_dict[from_id].get(to_id, np.NINF), weight)
                graph_dict[to_id][from_id] = max(graph_dict[to_id].get(from_id, np.NINF), weight)
    for node in full_node_list:
        neighbor_dict = graph_dict[node]
        neighbor_list = list(neighbor_dict.keys())
        if save_weight == True:
            weight_arr = np.array(list(neighbor_dict.values()))
            weight_arr = weight_arr / weight_arr.sum()
            graph_dict[node] = {'neighbor': neighbor_list, 'weight': weight_arr.tolist()}
        else:
            graph_dict[node] = neighbor_list
    print(graph_dict[full_node_list[0]])
    return graph_dict

def get_sp_adj_mat(file_path, full_node_list, sep='\t', weight_flag=False):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num).tolist()))
    A = lil_matrix((node_num, node_num))
    with open(file_path, 'r') as fp:
        content_list = fp.readlines()
        # ignore header
        for line in content_list[1:]:
            line_list = line.split(sep)
            from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
            from_id = node2idx_dict[from_node]
            to_id = node2idx_dict[to_node]
            if not weight_flag:
                A[from_id, to_id] = 1
                A[to_id, from_id] = 1
            else:
                A[from_id, to_id] = weight
                A[to_id, from_id] = weight
    A = A.tocsr()
    return A

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_normalize_PPMI_adj(spmat):
    adj = normalize(spmat + sp.eye(spmat.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

# def round_func(val):
#     from decimal import Decimal, ROUND_HALF_UP
#     decimal_val = Decimal(val).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP) \
#                                 .quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
#     return str(decimal_val)

def wl_transform(spadj, labels, cluster=False):
    # the ith entry is equal to the 2 ^ (i - 1)'th prime
    prime_list = [2, 3, 7, 19, 53, 131, 311, 719, 1619, 3671, 8161, 17863, 38873, 84017, 180503,
                  386093, 821641, 1742537, 3681131, 7754077, 16290047]

    label_num = np.max(labels)
    # generate enough primes to have one for each label in the graph
    max_prime = prime_list[int(np.ceil(np.log2(label_num)))]
    primes = list(sieve.primerange(1, max_prime + 1))
    prime_dict = dict(zip(np.arange(1, len(primes) + 1).tolist(), primes))

    def map_func(val, map_dict):
        return np.log(map_dict[val])
    vfunc = np.vectorize(map_func)
    log_primes = vfunc(labels, prime_dict).reshape(-1, 1)

    signatures = labels + spadj.dot(log_primes).reshape(-1)
    import RWTGCN.preprocessing.helper as helper
    return helper.uniquetol(signatures, cluster=cluster)

def separate(info='', sep='=', num=5):
    print()
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)
    print()

if __name__ == '__main__':
    A = np.array([[0, 1, 1, 1, 1],
                  [1, 0, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 0]])
    spmat = sp.csr_matrix(A)
    labels = np.ones(len(A))
    new_label = wl_transform(spmat, labels)
    print(new_label)

    A = np.array([[0, 1, 0, 1, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0]])
    spmat = sp.csr_matrix(A)
    labels = np.ones(len(A))
    new_label = wl_transform(spmat, labels)
    print(new_label)
    new_label = wl_transform(spmat, new_label)
    print(new_label)