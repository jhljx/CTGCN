# coding: utf-8
import numpy as np
import networkx as nx
import random
import multiprocessing
import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch.nn import init

# Position-aware Graph Neural Networks. For more information, please refer to https://arxiv.org/abs/1906.04817
# We modify and simplify the code of PGNN from https://github.com/JiaxuanYou/P-GNN, and include this method in our graph embedding project framework.
# Author: jhljx
# Email: jhljx8918@gmail.com


####################### Utility Function #####################

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = multiprocessing.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


# approximate == -1 means exact shortest path (time consuming), approximate > 0 means shorted path with cut-off
def precompute_dist_data(edge_indices, num_nodes, approximate):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''
    if isinstance(edge_indices, list):
        is_list = True
        timestamp_num = len(edge_indices)
    else:  # tensor
        is_list = False
        timestamp_num = 1

    node_dist_list = []
    for i in range(timestamp_num):
        graph = nx.Graph()
        edge_index = edge_indices[i] if is_list else edge_indices
        assert edge_index.shape[0] == 2
        edge_arr = edge_index.transpose(0, 1).cpu().numpy()
        graph.add_edges_from(edge_arr)  # [edge_num, 2]
        graph.add_nodes_from(np.arange(num_nodes))
        # print('graph nodes: ', len(graph.nodes()))

        ##################
        # This block is quite memory consuming especially on large graphs
        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        # dist_tensor = torch.tensor(dists_array)
        node_dist_list.append(dists_array)
        #################
    if is_list:
        return node_dist_list
    return node_dist_list[0]


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_list = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchorset_list.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_list


# consider mutiple timestamps
def get_dist_max(anchorset_list, node_dist_list, device):
    anchor_set_num = len(anchorset_list)
    # print('anchor set num: ', anchor_set_num)
    if isinstance(node_dist_list, list):
        is_list = True
        timestamp = len(node_dist_list)
    else:
        is_list = False
        timestamp = 1

    dist_max_list = []
    dist_argmax_list = []
    for i in range(timestamp):
        node_dist = node_dist_list[i] if is_list else node_dist_list  # array
        dist_max = torch.zeros((node_dist.shape[0], anchor_set_num), device=device)
        dist_argmax = torch.zeros((node_dist.shape[0], anchor_set_num), device=device).long()
        for i in range(anchor_set_num):
            temp_id = anchorset_list[i]
            dist_temp = node_dist[:, temp_id]
            dist_max_temp, dist_argmax_temp = np.max(dist_temp, axis=1), np.argmax(dist_temp, axis=1)
            dist_max[:, i] = torch.from_numpy(dist_max_temp)
            dist_argmax[:, i] = torch.from_numpy(dist_argmax_temp)
        dist_max_list.append(dist_max)
        dist_argmax_list.append(dist_argmax)
    if is_list:
        return dist_max_list, dist_argmax_list
    return dist_max_list[0], dist_max_list[0]


# Select anchor sets
# element of dist_mat_list is np.ndarray
def preselect_anchor(node_num, node_dist_list, device):
    anchorset_list = get_random_anchorset(node_num, c=1)
    dists_max_list, dists_argmax_list = get_dist_max(anchorset_list, node_dist_list, device)
    return dists_max_list, dists_argmax_list


####################### Basic Ops #############################

# Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


# PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True, bias=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1, bias=bias)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim, bias=bias)
        self.linear_out_position = nn.Linear(output_dim, 1, bias=bias)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()  # [n, anchor_set_num]
        subset_features = feature[dists_argmax.flatten(), :]  # [n, anchor_set_num, input_dim]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1]))  # [n, anchor_set_num, input_dim]
        messages = subset_features * dists_max.unsqueeze(-1)  # [n, anchor_set_num, input_dim]
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)  # [n, anchor_set_num, input_dim]
        messages = torch.cat((messages, self_feature), dim=-1)  # [n, anchor_set_num, 2 * input_dim]
        messages = self.linear_hidden(messages).squeeze()   # [n, anchor_set_num, output_dim]
        messages = self.act(messages)  # [n, anchor_set_num, output_dim]
        out_position = self.linear_out_position(messages).squeeze(-1)  # [n, anchor_set_num]
        out_structure = torch.mean(messages, dim=1)  # [n, output_dim]
        return out_position, out_structure


# Position-aware graph neural network class
class PGNN(torch.nn.Module):
    input_dim: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    feature_pre: bool
    layer_num: int
    dropout: float
    bias: bool
    method_name: str

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=0.5, bias=True):
        super(PGNN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'PGNN'

        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=bias)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim, bias=bias)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim, bias=bias)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim, bias=bias)

    def forward(self, x, dists_max, dists_argmax):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.pgnn(x[i], dists_max[i], dists_argmax[i]))
            return output_list
        return self.pgnn(x, dists_max, dists_argmax)

    def pgnn(self, x, dists_max, dists_argmax):
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, dists_max, dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, dists_max, dists_argmax)
            # x = F.relu(x) # Note: optional!
            x = F.dropout(x, self.dropout, training=self.training)
        x_position, x = self.conv_out(x, dists_max, dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position
