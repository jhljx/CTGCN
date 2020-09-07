# coding: utf-8
import torch
import torch.nn as nn
from layers import CoreDiffusion, MLP


# K-core diffusion network
class CDN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    diffusion_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super(CDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type

        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, output_dim, bias=bias, rnn_type=rnn_type))
        elif diffusion_num > 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(CoreDiffusion(hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(CoreDiffusion(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            raise ValueError("number of layers should be positive!")

    # x: node feature tensor
    # adj_list: k-core subgraph adj list
    def forward(self, x, adj_list):
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x


# MLP classifier
class MLPClassifier(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    duration: int
    bias: bool
    activate_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, activate_type='N'):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.duration = duration
        self.bias = bias
        self.activate_type = activate_type

        self.mlp_list = nn.ModuleList()
        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, layer_num, bias=bias, activate_type=activate_type))

    def forward(self, x, batch_indices=None):
        if isinstance(x, list) or len(x.size()) == 3:  # list or 3D tensor(GCRN, CTGCN output)
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                output_list.append(self.mlp_classifier(x[i], batch_indices[i]))
            return output_list
        return self.mlp_classifier(x, batch_indices)

    def mlp_classifier(self, x, batch_indices=None):
        # x is a tensor
        embedding_mat = x[batch_indices] if batch_indices is not None else x
        x = self.mlp_list[0](embedding_mat)
        return x


# This class supports inner product edge features!
class InnerProduct(nn.Module):
    reduce: bool

    def __init__(self, reduce=True):
        super(InnerProduct, self).__init__()
        self.reduce = reduce

    def forward(self, x, edge_index):
        if isinstance(x, list) or len(x.size()) == 3:  # list or 3D tensor(GCRN, CTGCN output)
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                embedding_mat = x[i]
                edge_mat = edge_index[i]
                output_list.append(self.inner_product(embedding_mat, edge_mat))
            return output_list
        # x is a tensor
        return self.inner_product(x, edge_index)

    def inner_product(self, x, edge_index):
        # x is a tensor
        assert edge_index.shape[0] == 2
        edge_index = edge_index.transpose(0, 1)  # [edge_num, 2]
        embedding_i = x[edge_index[:, 0]]
        embedding_j = x[edge_index[:, 1]]
        if self.reduce:
            return torch.sum(embedding_i * embedding_j, dim=1)
        return embedding_i * embedding_j


class EdgeClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, activate_type='N'):
        super(EdgeClassifier, self).__init__()
        self.conv = InnerProduct(reduce=False)
        self.classifier = MLPClassifier(input_dim, hidden_dim, output_dim, layer_num, duration, bias=bias, activate_type=activate_type)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return self.classifier(x)


# K-core based graph convolutional network
class CGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    trans_num: int
    diffusion_num: int
    bias: bool
    rnn_type: str
    model_type: str
    trans_activate_type: str
    method_name: str

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, bias=True, rnn_type='GRU', model_type='C', trans_activate_type='L'):
        super(CGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type
        self.model_type = model_type
        self.trans_activate_type = trans_activate_type
        self.method_name = 'CGCN' + '-' + model_type

        assert self.model_type in ['C', 'S']
        assert self.trans_activate_type in ['L', 'N']

        if self.model_type == 'C':
            # self.mlp = nn.Linear(input_dim, hidden_dim, bias=bias)
            self.mlp = MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, activate_type=trans_activate_type)
            self.duffision = CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type)
        else:
            self.mlp = MLP(input_dim, hidden_dim, output_dim, trans_num, bias=bias, activate_type=trans_activate_type)
            self.duffision = CDN(output_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type)

    def forward(self, x, adj):
        if isinstance(x, list):
            timestamp_num = len(x)
            embedding_list, structure_list = [], []
            if self.model_type == 'C':
                for i in range(timestamp_num):
                    embedding_mat = self.cgcn(x[i], adj[i])
                    embedding_list.append(embedding_mat)
                return embedding_list
            else:
                for i in range(timestamp_num):
                    embedding_mat, structure_mat = self.cgcn(x[i], adj[i])
                    embedding_list.append(embedding_mat)
                    structure_list.append(structure_mat)
                return embedding_list, structure_list
        return self.cgcn(x, adj)

    def cgcn(self, x, adj):
        trans = self.mlp(x)
        x = self.duffision(trans, adj)
        if self.model_type == 'S':
            return x, trans
        return x


# K-core based temporal graph convolutional network
class CTGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    duration: int
    trans_num: int
    diffusion_num: int
    bias: bool
    rnn_type: str
    model_type: str
    trans_activate_type: str
    method_name: str

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, duration, bias=True, rnn_type='GRU', model_type='C', trans_activate_type='L'):
        super(CTGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.model_type = model_type
        self.trans_activate_type = trans_activate_type
        self.method_name = 'CTGCN' + '-' + model_type

        assert self.model_type in ['C', 'S']
        assert self.trans_activate_type in ['L', 'N']

        self.duration = duration
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.mlp_list = nn.ModuleList()
        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            if self.model_type == 'C':
                self.mlp_list.append(MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, activate_type=trans_activate_type))
                self.duffision_list.append(CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))
            else:  # model_type == 'S'
                self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, trans_num, bias=bias, activate_type=trans_activate_type))
                self.duffision_list.append(CDN(output_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))
        assert self.rnn_type in ['LSTM', 'GRU']

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, adj_list):
        time_num = len(x_list)
        hx_list, trans_list = [], []
        for i in range(time_num):
            x = self.mlp_list[i](x_list[i])
            trans_list.append(x)
            x = self.duffision_list[i](x, adj_list[i])
            hx_list.append(x)
        hx = torch.stack(hx_list).transpose(0, 1)
        out, _ = self.rnn(hx)
        out = self.norm(out).transpose(0, 1)
        if self.model_type == 'C':
            return out
        return out, trans_list
