import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.autograd import Variable
from RWTGCN.layers import GCGRUCell, GCLSTMCell

class RWTGCN(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    unit_type: str
    duration: int
    layer_num: int
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, duration, unit_type='GRU', bias=True):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.unit_type = unit_type
        self.duration = duration
        self.layer_num = layer_num
        self.bias = bias
        if self.unit_type == 'GRU':
            self.rnn_cell = GCGRUCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        elif self.unit_type == 'LSTM':
            self.rnn_cell = GCLSTMCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        else:
            raise AttributeError('unit type error!')

    def forward(self, x_list, adj_list):
        if torch.cuda.is_available():
            assert len(x_list) == self.duration and len(adj_list) == self.duration * self.layer_num
            # print('gpu input size: ', x_list[0].size())
            h0 = Variable(torch.zeros(x_list[0].size()[1], self.output_dim).cuda())
        else:
            assert len(x_list) == self.duration and len(adj_list) == self.duration
            # print('cpu input size: ', x_list[0].size())
            h0 = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
        hx_list = []
        hx = h0
        for i in range(len(x_list)):
            if torch.cuda.is_available():
                temp_adj_list = adj_list[i * self.layer_num: (i + 1) * self.layer_num]
                # temp_adj_list = nn.ParameterList()
                # for j in range(i * self.layer_num, (i + 1) * self.layer_num):
                #     temp_adj_list.append(adj_list[j])
                hx = self.rnn_cell(x_list[i], temp_adj_list, hx)
            else:
                hx = self.rnn_cell(x_list[i], adj_list[i], hx)
            hx_list.append(hx)
        return hx_list