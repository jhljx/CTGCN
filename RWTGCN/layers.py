import torch, math
import torch.nn as nn
import torch.nn.functional as F


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
        gate = F.sigmoid(gate)
        output = F.relu(output)
        return trans + gate * (output - trans)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'


class GatedGCN(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    dropout: float
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, bias=True):
        super(GatedGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.bias = bias

        step_dim = (input_dim - output_dim) / layer_num
        assert step_dim > 0

        self.gc_list = []
        if (input_dim - output_dim) % layer_num == 0:
            self.gc_list.append(GatedGraphConvolution(input_dim, input_dim - step_dim, bias=bias))
            input_dim -= step_dim
        else:
            remain_dim = (input_dim - output_dim) % layer_num
            self.gc_list.append(GatedGraphConvolution(input_dim, input_dim - remain_dim - step_dim, bias=bias))
            input_dim -= remain_dim + step_dim
        for i in range(1, layer_num):
            self.gc_list.append(GatedGraphConvolution(input_dim, input_dim - step_dim, bias=bias))
            input_dim -= step_dim

    def forward(self, x, adj_list):
        assert self.layer_num == len(adj_list)
        for i in range(self.layer_num):
            x = self.gc_list[i](x, adj_list[i])
            x = F.dropout(x, self.dropout, training=self.training)
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

        self.gcn = GatedGCN(input_dim, output_dim, layer_num, dropout, bias=bias)
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

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)
        return hy


class GCLSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
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

        self.gcn = GatedGCN(input_dim, output_dim, layer_num, dropout, bias=bias)
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

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return hy, cy


class Infomax(nn.Module):
    input_dim: int
    output_dim: int
    bias: int

    def __init__(self, input_dim, output_dim, bias=True):
        super(Infomax, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        # bilinear layer weight
        self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.output_dim)
        self.w.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x, pos_hx, neg_hx):
        # x is sparse tensor
        if x.layout == torch.sparse_coo:
            res = torch.sparse.mm(x, self.w)
            pos_score = torch.mm(res, pos_hx)
            neg_score = torch.mm(res, neg_hx)
        # x is dense tensor
        else:
            res = torch.mm(x, self.w)
            pos_score = torch.mm(res, pos_hx)
            neg_score = torch.mm(res, neg_hx)
        if self.b is not None:
            pos_score += self.b
            neg_score += self.b
        pos_neg = torch.cat((pos_score, neg_score), 1)
        return pos_neg
