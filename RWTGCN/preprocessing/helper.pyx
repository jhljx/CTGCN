import numpy as np
import pandas as pd
import os, json
import random
import scipy.sparse as sp
cimport numpy as np
from libc.math cimport log
import time

def random_walk(graph, node_list, walk_dir_path, freq_dir_path, f_name, int walk_length, int walk_time, bint weight):
    t0 = time.time()
    graph_dict = {}
    # preprocessing
    for node in node_list:
        node_neighbors = list(graph.neighbors(node))
        weight = np.array([graph[node][neighbor]['weight'] for neighbor in node_neighbors])
        graph_dict[node] = {'neighbor': node_neighbors}
        graph_dict[node]['weight'] = weight / weight.sum()
    t1 = time.time()
    print('build graph time: ', t1 - t0, ' seconds!')
    node_num = len(node_list)
    node2idx_dict = dict(zip(node_list, np.arange(node_num).tolist()))
    node_freq_arr = np.zeros(node_num, dtype=int)
    walk_pair_set =set()

    cdef int iter
    cdef int nd_num = node_num
    cdef int nidx
    cdef int seq_len
    cdef int i
    cdef int j
    cdef int cnt = 1
    cdef int walk_len = walk_length + 1
    # random walk
    for nidx in range(nd_num):
        for iter in range(walk_time):
            start_node = node_list[nidx]
            eps = 1e-8
            walk = [start_node]
            cnt = 1
            while cnt < walk_len:
                cur = walk[-1]
                neighbors = graph_dict[cur]['neighbor']
                weights = graph_dict[cur]['weight']
                if len(neighbors) == 0:
                    break
                walk.append(np.random.choice(neighbors, p=weights) if weight else np.random.choice(neighbors))
                cnt += 1
            # count walk pair
            seq_len = len(walk)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if walk[i] == walk[j]:
                        continue
                    walk_pair_set.add((walk[i], walk[j]))
                    walk_pair_set.add((walk[j], walk[i]))
                    from_id, to_id = node2idx_dict[wak[i]], node2idx_dict[walk[j]]
                    node_freq_arr[from_id] += 1
                    node_freq_arr[to_id] += 1
    t2 = time.time()
    print('build graph time: ', t2 - t1, ' seconds!')
    del graph_dict

    tot_freq = node_freq_arr.sum()
    Z = 0.00001
    neg_node_list = []
    for nidx in range(nd_num):
        rep_num = int(((node_freq_arr[nidx]/tot_freq)**0.75)/ Z)
        neg_node_list += [nidx] * rep_num
    walk_file_path = os.path.join(freq_dir_path, f_name.split('.')[0] + '.json')
    with open(walk_file_path, 'w') as fp:
        json.dump(neg_node_list, fp)
    del neg_node_list, node_freq_arr
    t3 = time.time()
    print('node freq time: ', t3 - t2, ' seconds!')

    walk_file_path = os.path.join(walk_dir_path, f_name.split('.')[0] + '.json')
    walk_pair_list = list(walk_pair_set)
    with open(walk_file_path, 'w') as fp:
        json.dump(walk_pair_list, fp)
    t4 = time.time()
    print('walk pair time: ', t4 - t3, ' seconds!')

def hybrid_random_walk(original_graph, structural_graph, node_list, walk_dir_path, freq_dir_path, f_name, tensor_dir_path,
                int walk_length, int walk_time, double prob, bint weight):
    t0 = time.time()
    original_graph_dict, structural_graph_dict = {}, {}
    # preprocessing
    for node in node_list:
        original_neighbors = list(original_graph.neighbors(node))
        original_weight = np.array([original_graph[node][neighbor]['weight'] for neighbor in original_neighbors])
        original_graph_dict[node] = {'neighbor': original_neighbors}
        original_graph_dict[node]['weight'] = original_weight / original_weight.sum()

        structural_neighbors = list(structural_graph.neighbors(node))
        structural_weight = np.array([structural_graph[node][neighbor]['weight'] for neighbor in structural_neighbors])
        structural_graph_dict[node] = {'neighbor': structural_neighbors}
        structural_graph_dict[node]['weight'] = structural_weight / structural_weight.sum()
    t1 =time.time()
    print('build graph time: ', t1 - t0, ' seconds!')

    node_num = len(node_list)
    node2idx_dict = dict(zip(node_list, np.arange(node_num).tolist()))
    node_freq_arr = np.zeros(node_num, dtype=int)
    walk_pair_set =set()
    walk_adj_list, node_count_list, all_count_list = [{}], [[]], [-1]

    cdef int iter
    cdef int nd_num = node_num
    cdef int nidx
    cdef int seq_len
    cdef int i
    cdef int j
    cdef int cnt = 1
    cdef int walk_len = walk_length + 1
    for i in range(1, walk_len):
        step_adj_list.append({})
        node_count_list.append({})
        all_count_list.append(0)
    # random walk
    for nidx in range(nd_num):
        for iter in range(walk_time):
            start_node = node_list[nidx]
            eps = 1e-8
            walk = [start_node]
            cnt = 1
            while cnt < walk_len:
                cur = walk[-1]
                rd = np.random.random()
                if rd <= prob + eps:
                    neighbors = original_graph_dict[cur]['neighbor']
                    weights = original_graph_dict[cur]['weight']
                else:
                    neighbors = structural_graph_dict[cur]['neighbor']
                    weights = structural_graph_dict[cur]['weight']
                if len(neighbors) == 0:
                    break
                walk.append(np.random.choice(neighbors, p=weights) if weight else np.random.choice(neighbors))
                cnt += 1

            seq_len = len(walk)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if walk[i] == walk[j]:
                        continue
                    from_id, to_id = node2idx_dict[wak[i]], node2idx_dict[walk[j]]
                    assert  from_id != to_id
                    edge_dict, node_count_dict = walk_adj_list[j - i], node_count_list[j - i]
                    edge, reverse_edge = (from_id, to_id), (to_id, from_id)

                    edge_dict[edge] = edge_dict.get(edge, 0) + 1
                    edge_dict[reverse_edge] = edge_dict.get(reverse_edge, 0) + 1
                    walk_pair_set.add(edge)
                    walk_pair_set.add(reverse_edge)

                    node_freq_arr[from_id] += 1
                    node_freq_arr[to_id] += 1
                    node_count_dict[from_id] = node_count_dict.get(from_id, 0) + 1
                    node_count_dict[to_id] = node_count_dict.get(to_id, 0) + 1
                    all_count_list[j - i] += 2
    t2 = time.time()
    print('random walk time: ', t2 - t1, ' seconds!')
    del original_graph_dict, structural_graph_dict

    tot_freq = node_freq_arr.sum()
    Z = 0.00001
    neg_node_list = []
    for nidx in range(nd_num):
        rep_num = int(((node_freq_arr[nidx]/tot_freq)**0.75)/ Z)
        neg_node_list += [nidx] * rep_num
    walk_file_path = os.path.join(freq_dir_path, f_name.split('.')[0] + '.json')
    with open(walk_file_path, 'w') as fp:
        json.dump(neg_node_list, fp)
    del neg_node_list, node_freq_arr
    t3 = time.time()
    print('node freq time: ', t3 - t2, ' seconds!')

    walk_file_path = os.path.join(walk_dir_path, f_name.split('.')[0] + '.json')
    walk_pair_list = list(walk_pair_set)
    with open(walk_file_path, 'w') as fp:
        json.dump(walk_pair_list, fp)
    t4 = time.time()
    print('walk pair time: ', t4 - t3, ' seconds!')

    for idx in range(1, walk_len):
        edge_dict = walk_adj_list[idx]
        node_count_dict = node_count_list[idx]
        all_count = all_count_list[idx]
        for edge, cnt in edge_dict.items():
            from_id = edge[0]
            to_id = edge[1]
            res = log(cnt * all_count / (node_count[from_id] * node_count[to_id]))
            edge_dict[edge] = res if res > 0 else res
        edge_arr = np.array(list(edge_dict.keys()))
        weight_arr = np.array(list(edge_dict.values()))
        spmat = sp.coo_matrix((weight_arr, (edge_arr[:,0], edge_arr[:,1])), shape=(node_num, node_num))
        sp.save_npz(os.path.join(tensor_dir_path, str(idx) + ".npz"), spmat)
    t5 = time.time()
    print('walk tensor time: ', t5 - t4, ' seconds!')

# get unique values from array with tolerance=1e-12
def uniquetol(data_arr, cluster=False):
    idx_arr = np.argsort(data_arr)
    data_num = len(idx_arr)
    idx_order_dict = dict(zip(idx_arr.tolist(), np.ones(data_num).tolist()))
    cdef int pos = 0
    value = 1
    cdef int max_num = data_num
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

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def get_structural_neighbors(color_arr, output_file, cluster_dict, cluster_len_dict, idx2nid_dict,
                             int max_neighbor_num):
    structural_edges_dict = dict()
    structural_edge_list = []

    t1 = time.time()
    cdef int i = 0
    cdef int node_num = color_arr.shape[0]
    cdef int sample_num = 0
    for i in range(node_num):
        row_arr = color_arr[i, :]
        from_node = idx2nid_dict[i]
        cluster_type = row_arr[0]
        cluster_list = cluster_dict[cluster_type]
        cluster_num = cluster_len_dict[cluster_type]
        if cluster_num == 1:
            continue
        cnt = random.randint(1, min(cluster_num, max_neighbor_num))
        sampled_nodes = random.sample(cluster_list, cnt)

        sample_num = cnt
        for j in range(sample_num):
            to_idx = sampled_nodes[j]
            to_node = idx2nid_dict[to_idx]
            key = (from_node, to_node)
            if from_node == to_node or key in structural_edges_dict:
                continue
            to_arr = color_arr[to_idx, :]
            weight = sigmoid(row_arr.dot(to_arr))
            structural_edge_list.append([from_node, to_node, weight])

    df_structural_edges = pd.DataFrame(structural_edge_list, columns=['from_id', 'to_id', 'weight'])
    print('edge num: ', df_structural_edges.shape[0])
    df_structural_edges.to_csv(output_file, sep='\t', index=False, header=True, float_format='%.3f')
    return