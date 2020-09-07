# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.gcn import GCN, TgGCN

# Graph Convolutional Recurrent Network = Graph Convolutional Network + Gated Recurrent Unit
# This model is similar to the model proposed in paper 'Structured Sequence Modeling with Graph Convolutional Recurrent Networks'.
# For more information, please refer to https://arxiv.org/abs/1612.07659
# We refer to the code of GCRN in https://github.com/IBM/EvolveGCN/blob/master/models.py and include this method in our graph embedding project framework.
# Author: jhljx
# Email: jhljx8918@gmail.com


class GCRN(nn.Module):
    input_dim: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    feature_pre: bool
    layer_num: int
    dropout: float
    duration: int
    rnn_type: str
    bias: bool
    method_name: str

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=0.5, bias=True, duration=1, rnn_type='GRU'):
        super(GCRN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.duration = duration
        self.rnn_type = rnn_type
        self.method_name = 'GCRN'

        self.gcn_list = nn.ModuleList()
        for i in range(self.duration):
            # self.gcn_list.append(TgGCN(input_dim, feature_dim, hidden_dim, output_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias))
            self.gcn_list.append(GCN(input_dim, hidden_dim, output_dim, dropout=dropout, bias=bias))
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        else:  # 'GRU'
            self.rnn = nn.GRU(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, edge_list):
        time_num = len(x_list)
        hx_list = []
        for i in range(time_num):
            x = self.gcn_list[i](x_list[i], edge_list[i])
            x = F.normalize(x, p=2)
            hx_list.append(x)
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)
        out, _ = self.rnn(hx)
        out = self.norm(out)
        return out.transpose(0, 1)
