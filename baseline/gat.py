# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

# Graph Attention Networks. For more information, please refer to https://arxiv.org/abs/1710.10903
# We copy and modify GAT code from https://github.com/Diego999/pyGAT, and include this method in our graph embedding project framework.
# # Author: jhljx
# # Email: jhljx8918@gmail.com

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, input, adj):
        # dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        # adj = adj.tocsc()
        # edge = torch.LongTensor(np.array(adj.nonzero()))

        if input.layout == torch.sparse_coo:
            h = torch.sparse.mm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()
        # Self-attention on the nodes - Shared attention mechanism
        edge_index = adj._indices()  # [2, edge_num]
        edge_h = torch.cat((h[edge_index[0, :], :], h[edge_index[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge_index, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=input.device))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge_index, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Original version of GAT
class GAT(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    alpha: float
    head_num: int
    method_name: str

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6, alpha=0.2, head_num=8, learning_type='U-neg'):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.head_num = head_num
        assert learning_type in ['U-neg', 'S-node', 'S-edge', 'S-link-st', 'S-link-dy']
        self.learning_type = learning_type
        self.method_name = 'GAT'

        self.attentions = [SpGraphAttentionLayer(input_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(head_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(hidden_dim * head_num, output_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gat(x[i], adj[i]))
            return output_list
        return self.gat(x, adj)

    def gat(self, x, adj):
        if x.layout != torch.sparse_coo:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        if self.learning_type == 'U-neg':
            return F.log_softmax(x, dim=1)
        return x


# Pytorch-Geometric version of GAT
class TgGAT(torch.nn.Module):
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
        super(TgGAT, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'TgGAT'

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=bias)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim, bias=bias)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim, bias=bias)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim, bias=bias)

    def forward(self, x, edge_index):
        if isinstance(x, list):
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.gat(x[i], edge_index[i]))
            return output_list
        return self.gat(x, edge_index)

    def gat(self, x, edge_index):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            F.dropout(x, self.dropout, training=self.training)
        x = self.conv_out(x, edge_index)
        return x
