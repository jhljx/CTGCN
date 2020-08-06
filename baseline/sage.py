# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

# Graph Sample And Aggregate(GraphSAGE). For more information, please refer to https://arxiv.org/abs/1706.02216
# We copy some code of GraphSAGE in https://github.com/JiaxuanYou/P-GNN, and include this method in our graph embedding project framework.
# # Author: jhljx
# # Email: jhljx8918@gmail.com


class SAGE(torch.nn.Module):
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
        super(SAGE, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias
        self.method_name = 'SAGE'

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
        x = F.normalize(x, p=2, dim=-1)
        return x
