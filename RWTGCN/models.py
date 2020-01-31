import torch
import torch.nn as nn
from torch.autograd import Variable
from RWTGCN.layers import GCGRUCell, GCLSTMCell


class RWTGCN(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    unit_type: str
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, unit_type='GRU', bias=True):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.unit_type = unit_type
        self.bias = bias
        if self.unit_type == 'GRU':
            self.rnn_cell = GCGRUCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        elif self.unit_type == 'LSTM':
            self.rnn_cell = GCLSTMCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        else:
            raise AttributeError('unit type error!')

    def forward(self, x_list, adj_list):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x_list[0].size(0), self.output_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x_list[0].size(0), self.output_dim))
        outs = []
        hn = h0
        for i in range(len(x_list)):
            hn = self.rnn_cell(x_list[i], adj_list[i], hn)
            outs.append(hn)
        return outs
