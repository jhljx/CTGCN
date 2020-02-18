import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.autograd import Variable
from RWTGCN.layers import GatedGraphConvolution, GraphConvolution, GCGRUCell, GCLSTMCell

class MRGCN(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    dropout: float
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, bias=True):
        super(MRGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias

        self.gc_list = nn.ModuleList()
        self.gc_list.append(GatedGraphConvolution(input_dim, output_dim, bias=bias))
        for i in range(1, layer_num):
            self.gc_list.append(GatedGraphConvolution(output_dim, output_dim, bias=bias))
        self.gc_list = nn.ModuleList(self.gc_list)

    def forward(self, x, adj_list):
        assert self.layer_num == len(adj_list)
        # gcn for static embedding
        if isinstance(x, list):
            assert len(x) == 1
            hx_list, xi = [], x[0]
            for i in range(self.layer_num):
                xi = self.gc_list[i](xi, adj_list[i])
                xi = F.dropout(xi, self.dropout, training=self.training)
            hx_list.append(xi)
            return hx_list
        # x is a sparse matrix, adj is a list(or nn.ParameterList)
        for i in range(self.layer_num):
            x = self.gc_list[i](x, adj_list[i])
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        # gcn for static embedding
        if isinstance(x, list):
            assert len(x) == 1
            hx_list = []
            x1 = F.relu(self.gc1(x[0], adj[0]))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj[0])
            hx_list.append(x2)
            return hx_list
        # x is a sparse tensor
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


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
        assert (len(x_list) == self.duration) and (len(adj_list) == self.duration)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x_list[0].size()[1], self.output_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
        hx, hx_list = h0, []
        for i in range(len(x_list)):
            hx = self.rnn_cell(x_list[i], adj_list[i], hx)
            hx_list.append(hx)
        return hx_list