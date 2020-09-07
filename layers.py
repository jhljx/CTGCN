# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.att_weight.data.uniform_(0, 1)

    def forward(self, x, adj_list):
        hx_list = []
        # output = None
        for i, adj in enumerate(adj_list):
            if i == 0:
                res = torch.sparse.mm(adj, x)
            else:
                res = hx_list[-1] + torch.sparse.mm(adj, x)
            # hx = self.linear(res)
            hx_list.append(res)
        hx_list = [F.relu(res) for res in hx_list]

        #################################
        # Simple Core Diffusion, no RNN
        # out = hx_list[0]
        # for i, res in enumerate(hx_list[1:]):
        #     out = out + res
        # output = self.linear(out)
        ##################################
        # Add RNN to improve performance, but this will reduce the computation efficiency a little.
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)
        # Layer normalization could improve performance and make rnn stable
        output = self.norm(output)
        return output


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    activate_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True, activate_type='N'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.bias = bias
        self.activate_type = activate_type
        assert self.activate_type in ['L', 'N']
        assert self.layer_num > 0

        if layer_num == 1:
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(layer_num - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, x):
        if self.layer_num == 1:  # Linear model
            x = self.linear(x)
            if self.activate_type == 'N':
                x = F.selu(x)
            return x
        h = x  # MLP
        for layer in range(self.layer_num):
            h = self.linears[layer](h)
            if self.activate_type == 'N':
                h = F.selu(h)
        return h
