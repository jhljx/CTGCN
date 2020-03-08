import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.w4 = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
        self.epsilo = nn.Parameter(torch.FloatTensor(1))
        if bias:
            self.b1 = nn.Parameter(torch.FloatTensor(output_dim))
            self.b2 = nn.Parameter(torch.FloatTensor(output_dim))
            self.b3 = nn.Parameter(torch.FloatTensor(output_dim))
            self.b4 = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
            self.register_parameter('b3', None)
            self.register_parameter('b4', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.output_dim)
        self.epsilo.data.uniform_(1, 2)
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        self.w3.data.uniform_(-stdv, stdv)
        self.w4.data.uniform_(-stdv, stdv)
        if self.b1 is not None:
            self.b1.data.uniform_(-stdv, stdv)
        if self.b2 is not None:
            self.b2.data.uniform_(-stdv, stdv)
        if self.b3 is not None:
            self.b3.data.uniform_(-stdv, stdv)
        if self.b4 is not None:
            self.b4.data.uniform_(-stdv, stdv)

    def forward(self, input, res_input, adj):
        # sparse tensor
        # print('input', input)
        if input.layout == torch.sparse_coo:
            support = torch.sparse.mm(input, self.w1)
            trans = torch.sparse.mm(res_input, self.w2)
            gate1 = torch.sparse.mm(input, self.w3)
        # dense tensor
        else:
            support = torch.mm(input, self.w1)
            trans = torch.mm(res_input, self.w2)
            gate1 = torch.mm(input, self.w3)
        if self.b2 is not None:
            trans += self.b2
        if self.b3 is not None:
            gate1 += self.b3
        # gate1 = torch.sigmoid(gate1)
        output = torch.sparse.mm(adj, support) + self.epsilo * support
        del support
        if self.b1 is not None:
            output += self.b1
        output = F.relu(output)
        trans = torch.sigmoid(trans)
        gate2 = torch.mm(output, self.w4)
        if self.b4 is not None:
            gate2 += self.b4
        # gate2 = torch.sigmoid(gate2)
        gate = torch.sigmoid(gate1 + gate2)
        del gate1, gate2
        # return output, trans
        #return trans + output
        # return torch.cat([trans, output], dim=1)
        # return trans * output
        #return gate1 * trans + gate2 * output
        return output + gate * (trans - output),  trans + gate * (output - trans)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
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
        del support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class CoreDiffusion(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, bias=True, rnn_type='GRU'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.rnn_type = rnn_type

        # self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # if bias:
        #     self.b = nn.Parameter(torch.FloatTensor(output_dim))
        # else:
        #     self.register_parameter('b', None)
        # self.alpha = nn.Parameter(torch.FloatTensor(1))
        # self.beta = nn.Parameter(torch.FloatTensor(1))

        if self.rnn_type == 'LSTM':
            self.rnn = LSTMCell(input_dim, output_dim, bias=bias)
            # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        elif self.rnn_type == 'GRU':
            self.rnn = GRUCell(input_dim, output_dim, bias=bias)
            #self.gru = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        else:
            raise AttributeError('Unsupported rnn type!')
        self.norm = nn.LayerNorm(output_dim)
        # self.linear = Linear(output_dim * 2, output_dim, bias=bias)
        # self.reset_parameters()

    def reset_parameters(self):
        # self.alpha.data.uniform_(0, 1)
        # self.beta.data.uniform_(0, 1)
        stdv = 1 / math.sqrt(self.output_dim)
        self.w.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_list):
        # hx = F.relu(torch.sparse.mm(adj_list[0], x))
        if torch.cuda.is_available():
            hx = Variable(torch.zeros(x.size()[0], self.output_dim).cuda())
        else:
            hx = Variable(torch.zeros(x.size()[0], self.output_dim))
        adj_list = adj_list[::-1]
        for i, adj in enumerate(adj_list):
            res = F.relu(torch.sparse.mm(adj, x))
            hx = self.rnn(res, hx)
        hx = self.norm(hx)
        return hx

class GRUCell(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool

    def __init__(self, input_dim, output_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.x2h = nn.Linear(input_dim, 3 * output_dim, bias=bias)
        self.h2h = nn.Linear(input_dim, 3 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
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


class LSTMCell(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool

    def __init__(self, input_dim, output_dim, bias=True):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.x2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
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


class Linear(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    def __init__(self, input_dim, output_dim, bias=True):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.output_dim)
        self.w.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # sparse tensor
        # print('input', input)
        if input.layout == torch.sparse_coo:
            support = torch.sparse.mm(input, self.w)
        else:
            support = torch.mm(input, self.w)
        if self.b is not None:
            return support + self.b
        return support


###MLP with linear output
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = Linear(input_dim, output_dim, bias=bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(num_layers - 2):
                self.linears.append(Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(Linear(hidden_dim, output_dim, bias=bias))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)