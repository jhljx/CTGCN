# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# K-core subgraph based diffusion layer
class CoreDiffusion(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.core_num = core_num
        self.rnn_type = rnn_type

        self.linear = nn.Linear(input_dim, output_dim)
        # self.att_weight = nn.Parameter(torch.FloatTensor(core_num))
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            # self.lstm = LSTMCell(input_dim, output_dim, bias=bias)
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            # self.gru = GRUCell(input_dim, output_dim, bias=bias)
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        # self.norm = nn.BatchNorm1d(output_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.att_weight.data.uniform_(0, 1)

    def forward(self, x, adj_list):
        # if self.rnn_type == 'GRU':
            # hx = Variable(torch.zeros(x.shape[0], self.output_dim, device=x.device))
        adj_list = adj_list[::-1]
        hx_list = []
        output = None
        # assert len(adj_list) == core_num
        for i, adj in enumerate(adj_list):
            res = torch.sparse.mm(adj, x)
            res = F.relu(res)
            # hx = self.linear(res)
            hx_list.append(res)
            # hx = torch.stack(hx_list, dim=0).transpose(0, 1)
            # out, _ = self.rnn(hx)
            # hx = self.gru(res, hx)
            # hx_list.append(hx)
            # if output is None:
            #      output = hx
            # #     # output = self.att_weight[i] * hx
            # #     # print('output shape: ', output)
            # else:
            # #     # output = output + self.att_weight[i] * hx
            #      output = output + hx
            # hx = hx + res
    # hx = F.normalize(hx, p=2, dim=-1)
    #Layer normalization could improve performance and make rnn stable
    #
        # hx = out[:, -1, :]
        # hx = F.normalize(hx, p=2, dim=-1)
        # output = self.linear(output)
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)
        output = self.norm(output)
        return output
        # else:
        #     hx = Variable(torch.zeros(x.shape[0], self.output_dim, device=x.device))
        #     cx = Variable(torch.zeros(x.shape[0], self.output_dim, device=x.device))
        #     adj_list = adj_list[::-1]
        #     for i, adj in enumerate(adj_list):
        #         res = torch.sparse.mm(adj, hx)
        #         res = self.linear(res)
        #         res = F.relu(res)
        #         # res = F.dropout(res, training=self.training)
        #         hx, cx = self.lstm(res, hx, cx)
        #     # hx = F.normalize(hx, p=2, dim=-1)
        #     #Layer normalization could improve performance and make rnn stable
        #     hx = self.norm(hx)
        #     # hx = F.dropout(hx, training=self.training)
        #     return hx


# Gated Recurrent Unit(GRU) cell
class GRUCell(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool

    def __init__(self, input_dim, output_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.x2h = nn.Linear(input_dim, 3 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 3 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        del i_r, i_i, i_n, h_r, h_i, h_n
        hy = newgate + inputgate * (hidden - newgate)
        return hy


# Long-Short Term Memory(LSTM) cell
class LSTMCell(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool

    def __init__(self, input_dim, output_dim, bias=True):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.x2h = nn.Linear(input_dim, 4 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x, hx, cx):
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        del ingate, forgetgate, cellgate, outgate
        return hy, cy


# # Linear layer
# class Linear(nn.Module):
#     input_dim: int
#     output_dim: int
#     bias: bool
#
#     def __init__(self, input_dim, output_dim, bias=True):
#         super(Linear, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.bias = bias
#
#         self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         if bias:
#             self.b = nn.Parameter(torch.FloatTensor(output_dim))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.output_dim)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.b is not None:
#             self.b.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         # sparse tensor
#         if x.layout == torch.sparse_coo:
#             support = torch.sparse.mm(x, self.weight)
#         else:
#             support = torch.mm(x, self.weight)
#         if self.b is not None:
#             return support + self.b
#         return support


# Multi-Layer Perceptron(MLP) 'layer'
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True, activate_type='N'):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.activate_type = activate_type
        assert self.activate_type in ['L', 'N']

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            # self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            #for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            x = self.linear(x)
            if self.activate_type == 'L':
                return x
            else:  # activate_type == 'N'
                return F.selu(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers):
                if self.activate_type == 'L':
                    h = self.linears[layer](h)
                else:
                    h = F.selu(self.linears[layer](h))
            return h
