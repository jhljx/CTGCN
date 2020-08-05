# coding: utf-8
import torch
import torch.nn as nn
from baseline.dynAE import MLP
from baseline.dynRNN import MLLSTM


# dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning. For more information, please refer to https://arxiv.org/abs/1809.02657
# We refer to the dyngraph2vec tensorflow source code https://github.com/palash1992/DynamicGEM, and implement a pytorch version of dyngraph2vec
# Author: jhljx
# Email: jhljx8918@gmail.com


# DynAERNN model and its components
# Multi-timestamp MLP
class MTMLP(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, look_back, bias=True):
        super(MTMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias

        self.layer_list = nn.ModuleList()
        for timestamp in range(look_back):
            self.layer_list.append(MLP(input_dim, output_dim, n_units, bias=bias))

    # x dim: [batch_size, look_back, input_dim]
    def forward(self, x):
        hx_list = []
        for timestamp in range(self.look_back):
            hx = self.layer_list[timestamp](x[:, timestamp, :])
            hx_list.append(hx)
        return torch.stack(hx_list, dim=0).transpose(0, 1)


# DynAERNN class
class DynAERNN(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLLSTM
    decoder: MLLSTM

    def __init__(self, input_dim, output_dim, look_back=3, ae_units=None, rnn_units=None, bias=True, **kwargs):
        super(DynAERNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'DynAERNN'

        self.ae_encoder = MTMLP(input_dim, output_dim, ae_units, look_back, bias=bias)
        self.rnn_encoder = MLLSTM(output_dim, output_dim, rnn_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, ae_units[::-1], bias=bias)

    def forward(self, x):
        ae_hx = self.ae_encoder(x)
        output, hx = self.rnn_encoder(ae_hx)
        x_pred = self.decoder(hx)
        return hx, x_pred
