import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import torch, os, traceback
from time import time
from sympy import sieve
from numpy import random

def check_and_make_path(to_make):
    if not os.path.exists(to_make):
        os.makedirs(to_make)


def read_edgelist_from_dataframe(filename, full_node_list):
    df = pd.read_csv(filename, sep='\t')
    # dataframe['weight'] = 1.0
    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight',
                                    create_using=nx.Graph)
    graph.add_nodes_from(full_node_list)
    return graph

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

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

# get unique values from array with tolerance=1e-12
def uniquetol(data_arr, cluster=False):
    idx_arr = np.argsort(data_arr)
    data_num = len(idx_arr)
    idx_order_dict = dict(zip(idx_arr.tolist(), np.ones(data_num).tolist()))
    pos = 0
    value = 1
    max_num = data_num
    while pos < max_num:
        idx_order_dict[idx_arr[pos]] = value
        while(pos + 1 < max_num):
            if np.abs(data_arr[idx_arr[pos]] - data_arr[idx_arr[pos + 1]]) >= 1e-12:
                value += 1
                break
            idx_order_dict[idx_arr[pos + 1]] = value
            pos += 1
        pos += 1
    cluster_dict = dict()
    def map_func(idx):
        label = idx_order_dict[idx]
        if cluster == True:
            if label not in cluster_dict:
                cluster_dict[label] = [idx]
            else:
                cluster_dict[label].append(idx)
        return label
    vfunc = np.vectorize(map_func)
    labels = vfunc(np.arange(data_num))
    if cluster == True:
        return labels, cluster_dict
    return labels

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
    #print(signatures)
    # map signatures to integers counting from 1
    try:
        import RWTGCN.preprocessing.helper as helper
        return helper.uniquetol(signatures, cluster=cluster)
    except:
        pass
    return uniquetol(signatures, cluster=cluster)

def random_walk(original_graph, structural_graph, node_list, output_dir_path,
                walk_length, walk_time, prob, weight):
    original_graph_dict, structural_graph_dict = {}, {}
    # preprocessing
    for node in node_list:
        original_neighbors = list(original_graph.neighbors(node))
        original_weight = np.array([original_graph[node][i]['weight'] for i in original_neighbors])
        original_graph_dict[node] = {'neighbor': original_neighbors}
        original_graph_dict[node]['weight'] = original_weight / original_weight.sum()

        structural_neighbors = list(structural_graph.neighbors(node))
        structural_weight = np.array([structural_graph[node][i]['weight'] for i in structural_neighbors])
        structural_graph_dict[node] = {'neighbor': structural_neighbors}
        structural_graph_dict[node]['weight'] = structural_weight / structural_weight.sum()

    node_num = len(node_list)
    nid2idx_dict = dict(zip(node_list, np.arange(node_num).tolist()))

    spmat_list, node_count_list, all_count_list = [sp.lil_matrix(1, 1)], [[]], [-1]
    spmat_list += [sp.lil_matrix((node_num, node_num)) for i in range(walk_length)]
    node_count_list += [np.zeros(node_num, dtype=int).tolist() for i in range(walk_length)]
    all_count_list += np.zeros(walk_length, dtype=int).tolist()

    # random walk
    for node in node_list:
        for iter in range(walk_time):
            eps = 1e-8
            walk = [node]
            while len(walk) < walk_length + 1:
                cur = walk[-1]
                rd = random.random()
                if rd <= prob + eps:
                    neighbors = original_graph_dict[cur]['neighbor']
                    weights = original_graph_dict[cur]['weight']
                else:
                    neighbors = structural_graph_dict[cur]['neighbor']
                    weights = structural_graph_dict[cur]['weight']
                if len(neighbors) == 0:
                    break
                walk.append(random.choice(neighbors, p=weights) if weight else random.choice(neighbors))
            seq_len = len(walk)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    step = j - i
                    left_idx = nid2idx_dict[walk[i]]
                    #print(j, walk[j])
                    right_idx = nid2idx_dict[walk[j]]
                    spmat = spmat_list[step]
                    node_count = node_count_list[step]

                    spmat[left_idx, right_idx] += 1
                    spmat[right_idx, left_idx] += 1
                    node_count[left_idx] += 1
                    node_count[right_idx] += 1
                    all_count_list[step] += 2
    # calculate PPMI values
    for i in range(1, walk_length + 1):
        spmat = spmat_list[i].tocoo()
        node_count = node_count_list[i]
        all_count = all_count_list[i]
        df_PPMI = pd.DataFrame({'row': spmat.row, 'col': spmat.col, 'data': spmat.data}, dtype=int)

        def calc_PPMI(series):
            res = np.log(series['data'] * all_count / (node_count[series['row']] * node_count[series['col']]))
            if res < 0:
                return 0
            return res

        df_PPMI['data'] = df_PPMI.apply(calc_PPMI, axis=1)
        spmat = sp.coo_matrix((df_PPMI['data'], (df_PPMI['row'], df_PPMI['col'])), shape=(node_num, node_num))
        sp.save_npz(os.path.join(output_dir_path, str(i) + ".npz"), spmat)


def separate(info='', sep='=', num=5):
    print()
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)
    print()


def time_filter_with_dict_param(func, **kwargs):
    try:
        t1 = time()
        func(**kwargs)
        t2 = time()
        print(func.__name__, " spends ", t2 - t1, 'ms')
    except Exception as e:
        traceback.print_exc()


def time_filter_with_tuple_param(func, *args):
    t1 = time()
    func(*args)
    t2 = time()
    print(func.__name__, " spends ", t2 - t1, 'ms')


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