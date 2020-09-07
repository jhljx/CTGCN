# coding: utf-8
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg


# Graph Isomorphism Network. For more information, please refer to https://arxiv.org/abs/1810.00826
# We copy and modify GIN code from https://github.com/weihua916/powerful-gnns, and include this method in our graph embedding project framework.
# # Author: jhljx
# # Email: jhljx8918@gmail.com


# MLP with lienar output
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True):
        '''
            layer_num: number of layers in the neural networks (EXCLUDING the input layer). If layer_num=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.layer_num = layer_num
        self.bias = bias

        if layer_num < 1:
            raise ValueError("number of layers should be positive!")
        elif layer_num == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(layer_num - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            for layer in range(layer_num - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.layer_num - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.layer_num - 1](h)


# Original version of GIN
class GIN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    mlp_layer_num: int
    learn_eps: bool
    neighbor_pooling_type: str
    dropout: float
    bias: bool

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, mlp_layer_num, learn_eps, neighbor_pooling_type='sum', dropout=0.5, bias=True):
        '''
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            layer_num: number of layers in the neural networks
            mlp_layer_num: number of layers in mlps
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (sum, average, or max)
            dropout: dropout ratio on the final linear layer
            bias: whether to add bias for MLP
        '''
        super(GIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.mlp_layer_num = mlp_layer_num
        self.learn_eps = learn_eps
        self.neighbor_pooling_type = neighbor_pooling_type
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'GIN'
        self.eps = nn.Parameter(torch.zeros(self.layer_num))
        assert neighbor_pooling_type in ['sum', 'average', 'max']

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mlps = torch.nn.ModuleList()
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.layer_num - 1):
            self.mlps.append(MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layer_num, bias=bias))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.mlps.append(MLP(hidden_dim, hidden_dim, output_dim, mlp_layer_num, bias=bias))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

    def __preprocess_neighbors_maxpool(self, adj):
        # create padded_neighbor_list in a graph
        data = adj._values().cpu().numpy()
        row = adj._indices()[0, :].cpu().numpy()
        col = adj._indices()[1, :].cpu().numpy()
        node_num = adj.shape[0]
        sp_neighbors = sp.coo_matrix((data, (row, col)), shape=(node_num, node_num)).tolil()
        neighbor_list = sp_neighbors.rows
        return neighbor_list

    def __preprocess_neighbors_sumavepool(self, adj):
        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        node_num = adj.shape[0]
        edge_indices = adj._indices()
        weights = adj._values()

        if not self.learn_eps:
            # num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(node_num), range(node_num)]).to(adj.device)
            elem = torch.ones(node_num).to(adj.device)
            Adj_block_idx = torch.cat([edge_indices, self_loop_edge], 1)
            Adj_block_elem = torch.cat([weights, elem], 0)
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([node_num, node_num]))
        return Adj_block.to(adj.device)

    @staticmethod
    def maxpool(h, neighbor_list):
        neighbor_list_len = len(neighbor_list)
        neighbor_feats = torch.zeros_like(h).to(h.device)
        for i in range(neighbor_list_len):
            if len(neighbor_list[i]) == 0:
                continue
            neighbor_feats[i] = torch.max(h[neighbor_list[i]], 0)[0]
        return neighbor_feats

    def next_layer_eps(self, h, layer, neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(h.device))
                pooled = pooled / degree
        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, neighbor_list)
        else:
            # If sum or average pooling
            # As torch.spmm doesn't support matrix multiplication between two sparse tensors, so we have to reduce h into a dense matrix. A linear layer is needed!
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(h.device))
                pooled = pooled / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, x, adj):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gin(x[i], adj[i]))
            return output_list
        return self.gin(x, adj)

    def gin(self, x, adj):
        if self.neighbor_pooling_type == "max":
            neighbor_list = self.__preprocess_neighbors_maxpool(adj)
            Adj_block = []
        else:
            neighbor_list = []
            Adj_block = self.__preprocess_neighbors_sumavepool(adj)

        h = self.linear(x)
        for layer in range(self.layer_num):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, neighbor_list=neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, neighbor_list=neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)
            if layer < self.layer_num - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return h


# Pytorch-Geometric version of GIN
class TgGIN(torch.nn.Module):
    input_dim: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    feature_pre: bool
    layer_num: int
    dropout: float
    bias: bool
    method_name: str

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=True, bias=True, **kwargs):
        super(TgGIN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'TgGIN'

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=bias)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim, bias=bias)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim, bias=bias)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)  # default: eps=0, train_eps=False
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self, x, edge_index):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gin(x[i], edge_index[i]))
            return output_list
        return self.gin(x, edge_index)

    def gin(self, x, edge_index):
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
