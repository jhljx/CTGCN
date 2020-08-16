# coding: utf-8
import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import init
from torch.autograd import Variable

# Graph Sample And Aggregate(GraphSAGE). For more information, please refer to https://arxiv.org/abs/1706.02216
# We copy and modify GraphSAGE code from https://github.com/williamleif/graphsage-simple, and include this method in our graph embedding project framework.
# # Author: jhljx
# # Email: jhljx8918@gmail.com


class Aggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, num_sample=None, pooling_type='sum', gcn=False):
        """
        Initializes the aggregator for a specific graph.
        num_sample --- number of neighbors to sample. No sampling if None.
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(Aggregator, self).__init__()
        self.num_sample = num_sample
        self.pooling_type = pooling_type
        self.gcn = gcn
        assert pooling_type in ['sum', 'average', 'max']

    def forward(self, features, nodes, to_neighs):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        """
        # Local pointers to functions (speed hack)
        if not self.num_sample is None:
            samp_neighs = [random.sample(to_neigh, self.num_sample) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # print('gcn version!')
            samp_neighs = [samp_neigh + [nodes[i]] for i, samp_neigh in enumerate(samp_neighs)]
        sample_num = len(samp_neighs)

        if self.pooling_type in ['sum', 'average']:
            all_node_list = []
            for samp_neigh in samp_neighs:
                all_node_list += samp_neigh
            unique_nodes_list = np.unique(np.array(all_node_list))
            unique_node_num = len(unique_nodes_list)
            unique_nodes_dict = dict(zip(unique_nodes_list, np.arange(unique_node_num)))

            mask = torch.zeros(sample_num, unique_node_num)
            column_indices = [unique_nodes_dict[nid] for samp_neigh in samp_neighs for nid in samp_neigh]
            row_indices = [i for i in range(sample_num) for j in range(len(samp_neighs[i]))]
            mask[row_indices, column_indices] = 1
            mask = mask.to(features.device)

            if self.pooling_type == 'average':
                num_neigh = mask.sum(dim=1, keepdim=True)
                num_neigh[num_neigh == 0] = 1  # in case div operation produce nan
                mask = mask.div(num_neigh)
            unique_nodes_indices = torch.LongTensor(unique_nodes_list).to(features.device)
            embed_matrix = features[unique_nodes_indices]
            to_feats = mask.mm(embed_matrix)
        else:  # self.pooling_type == 'max'
            to_feats = torch.zeros_like(features).to(features.device)
            for i in range(sample_num):
                if len(samp_neighs[i]) == 0:
                    continue
                to_feats[i] = torch.max(features[samp_neighs[i]], 0)[0]
        return to_feats


class SAGE_Layer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_dim, output_dim, num_sample=10, pooling_type='sum', gcn=False, bias=True):
        super(SAGE_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_sample = num_sample
        self.pooling_type = pooling_type
        self.gcn = gcn
        self.bias = bias

        self.aggregator = Aggregator(num_sample=num_sample, pooling_type=pooling_type, gcn=gcn)
        self.linear = nn.Linear(input_dim if self.gcn else 2 * input_dim, output_dim, bias=bias)

    def forward(self, features, nodes, neighbor_list):
        """
        Generates embeddings for a batch of nodes.
        nodes -- list of nodes
        """
        to_neighbors = [neighbor_list[int(node)] for node in nodes]
        neigh_feats = self.aggregator(features, nodes, to_neighbors)
        if not self.gcn:
            node_indices = torch.LongTensor(nodes).to(features.device)
            # print('node indices: ', node_indices)
            self_feats = features[node_indices]
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.linear(combined))
        combined = F.normalize(combined, p=2)
        return combined


# Original version of GraphSAGE
class SAGE(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_samples: int
    pooling_type: str
    dropout: float
    bias: bool

    def __init__(self, input_dim, hidden_dim, output_dim, num_sample=10, pooling_type='sum', gcn=False, dropout=0.5, bias=True):
        super(SAGE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_sample = num_sample
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'SAGE'

        self.linear = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.sage1 = SAGE_Layer(hidden_dim, hidden_dim, num_sample, pooling_type=pooling_type, gcn=gcn, bias=bias)
        self.sage2 = SAGE_Layer(hidden_dim, output_dim, num_sample, pooling_type=pooling_type, gcn=gcn, bias=bias)

    def forward(self, x, adj):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.sage(x[i], adj[i]))
            return output_list
        return self.sage(x, adj)

    def sage(self, x, adj):
        # print('x shape: ', x.shape)
        data = adj._values().cpu().numpy()
        row = adj._indices()[0, :].cpu().numpy()
        col = adj._indices()[1, :].cpu().numpy()
        node_num = adj.shape[0]
        neighbor_list = sp.coo_matrix((data, (row, col)), shape=(node_num, node_num)).tolil().rows
        # print('neighbor_list: ', neighbor_list[0])
        nodes = np.arange(node_num)
        x = self.linear(x)   # [node_num, hidden_dim]
        h = self.sage1(x, nodes, neighbor_list)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.sage2(h, nodes, neighbor_list)
        return h


# Pytorch-Geometric version of GraphSAGE
class TgSAGE(torch.nn.Module):
    input_dim: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    feature_pre: bool
    layer_num: int
    dropout: float
    bias: bool
    method_name: str

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=0.5, bias=True, **kwargs):
        super(TgSAGE, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'TgSAGE'

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=bias)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim, bias=bias)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim, bias=bias)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim, bias=bias)

    def forward(self, x, edge_index):
        if isinstance(x, list):  # x: list, edge_index: list
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.sage(x[i], edge_index[i]))
            return output_list
        return self.sage(x, edge_index)

    def sage(self, x, edge_index):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv_out(x, edge_index)
        return x
