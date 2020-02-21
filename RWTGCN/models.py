import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from RWTGCN.layers import GatedGraphConvolution, GraphConvolution

class MRGCN(nn.Module):
    node_num: int
    input_dim: int
    output_dim: int
    layer_num: int
    dropout: float
    bias: bool

    def __init__(self, node_num, input_dim, output_dim, layer_num, dropout, bias=True):
        super(MRGCN, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias

        self.gc_list = nn.ModuleList()
        self.gc_list.append(GatedGraphConvolution(node_num, input_dim, output_dim, bias=bias))
        for i in range(1, layer_num):
            self.gc_list.append(GatedGraphConvolution(node_num, output_dim, output_dim, bias=bias))

    def forward(self, x, adj_list):
        assert self.layer_num == len(adj_list)
        # MRGCN for static embedding
        if isinstance(x, list):
            assert len(x) == 1
            hx_list, xi = [], x[0]
            for i in range(self.layer_num):
                xi = self.gc_list[i](xi, adj_list[i])
                # xi = F.dropout(xi, self.dropout, training=self.training)
            hx_list.append(xi)
            return hx_list
        # x is a sparse matrix, adj is a list (component of RWTGCN)
        for i in range(self.layer_num):
            x = self.gc_list[i](x, adj_list[i])
            # x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        # GCN for static embedding
        if isinstance(x, list):
            assert len(x) == 1
            hx_list = []
            x1 = F.relu(self.gc1(x[0], adj[0]))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj[0])
            hx_list.append(x2)
            return hx_list
        # x is a sparse tensor, component of other model
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCGRUCell(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, bias=True):
        super(GCGRUCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias

        self.gcn = MRGCN(input_dim, output_dim, layer_num, dropout, bias=bias)
        self.x2h = nn.Linear(output_dim, 3 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 3 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, adj_list, hidden):
        x = self.gcn(x, adj_list)
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


class GCLSTMCell(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, bias=True):
        super(GCLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias

        self.gcn = MRGCN(input_dim, output_dim, layer_num, dropout, bias=bias)
        self.x2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, adj_list, hidden):
        x = self.gcn(x, adj_list)
        hx, cx = hidden
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

class RWTGCN(nn.Module):
    node_num: int
    input_dim: int
    output_dim: int
    dropout: float
    unit_type: str
    duration: int
    layer_num: int
    bias: bool

    def __init__(self, node_num, input_dim, output_dim, layer_num, dropout, duration, unit_type='GRU', bias=True):
        super(RWTGCN, self).__init__()
        self.node_num = node_num
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