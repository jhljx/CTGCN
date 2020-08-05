# coding: utf-8
import torch.nn as nn


# dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning. For more information, please refer to https://arxiv.org/abs/1809.02657
# We refer to the dyngraph2vec tensorflow source code https://github.com/palash1992/DynamicGEM, and implement a pytorch version of dyngraph2vec
# Author: jhljx
# Email: jhljx8918@gmail.com


# DynRNN model and its components
# Multi-layer LSTM class
class MLLSTM(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.LSTM(input_dim, n_units[0], bias=bias, batch_first=True))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.LSTM(n_units[i - 1], n_units[i], bias=bias, batch_first=True))
        self.layer_list.append(nn.LSTM(n_units[-1], output_dim, bias=bias, batch_first=True))
        self.layer_num = layer_num + 1

    def forward(self, x):
        for i in range(self.layer_num):
            x, _ = self.layer_list[i](x)
        # return outputs and the last hidden embedding matrix
        return x, x[:, -1, :]


# DynRNN class
class DynRNN(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLLSTM
    decoder: MLLSTM

    def __init__(self, input_dim, output_dim, look_back=3, n_units=None,  bias=True, **kwargs):
        super(DynRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'DynRNN'

        self.encoder = MLLSTM(input_dim, output_dim, n_units, bias=bias)
        self.decoder = MLLSTM(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, x):
        output, hx = self.encoder(x)
        _, x_pred = self.decoder(output)
        return hx, x_pred
