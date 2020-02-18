import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from RWTGCN.models import GCN, MRGCN

class GatedGraphConvolution(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool

    def __init__(self, input_dim, output_dim, bias=True):
        super(GatedGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # linear transformation parameter
        self.w1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.w2 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # transform gate
        self.w3 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.b1 = nn.Parameter(torch.FloatTensor(output_dim))
            self.b2 = nn.Parameter(torch.FloatTensor(output_dim))
            self.b3 = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
            self.register_parameter('b3', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.output_dim)
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        self.w3.data.uniform_(-stdv, stdv)
        if self.b1 is not None:
            self.b1.data.uniform_(-stdv, stdv)
        if self.b2 is not None:
            self.b2.data.uniform_(-stdv, stdv)
        if self.b3 is not None:
            self.b3.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # sparse tensor
        if input.layout == torch.sparse_coo:
            support = torch.sparse.mm(input, self.w1)
            trans = torch.sparse.mm(input, self.w2)
            gate = torch.sparse.mm(input, self.w3)
        # dense tensor
        else:
            support = torch.mm(input, self.w1)
            trans = torch.mm(input, self.w2)
            gate = torch.mm(input, self.w3)
        output = torch.sparse.mm(adj, support)
        if self.b1 is not None:
            output += self.b1
        if self.b2 is not None:
            trans += self.b2
        if self.b3 is not None:
            gate += self.b3
        gate = torch.sigmoid(gate)
        output = F.relu(output)
        return trans + gate * (output - trans)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


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
        return hy, cy