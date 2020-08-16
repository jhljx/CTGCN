# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

# Graph Convolutional Network. For more information, please refer to https://arxiv.org/abs/1609.02907
# We copy and modify GCN code from https://github.com/tkipf/pygcn, and include this method in our graph embedding project framework.
# # Author: jhljx
# # Email: jhljx8918@gmail.com


class GraphConvolution(nn.Module):
    input_dim: int
    output_dim: int
    weight: nn.Parameter

    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        del support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


# Original version of GCN
class GCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    bias: bool
    method_name: str
    gc1: GraphConvolution
    gc2: GraphConvolution

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, bias=True):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'GCN'

        self.gc1 = GraphConvolution(input_dim, hidden_dim, bias=bias)
        self.gc2 = GraphConvolution(hidden_dim, output_dim, bias=bias)

    def forward(self, x, adj):
        # GCN for static embedding
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gcn(x[i], adj[i]))
            return output_list
        return self.gcn(x, adj)

    def gcn(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# Pytorch-Geometric version of GCN
class TgGCN(torch.nn.Module):
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
        super(TgGCN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'TgGCN'

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=bias)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim, bias=bias)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim, bias=bias)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim, bias=bias)

    def forward(self, x, edge_index):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gcn(x[i], edge_index[i]))
            return output_list
        return self.gcn(x, edge_index)

    def gcn(self, x, edge_index):
        assert edge_index.shape[0] == 2
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
