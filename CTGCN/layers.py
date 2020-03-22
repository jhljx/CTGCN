import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

        if self.rnn_type == 'LSTM':
            self.rnn = LSTMCell(input_dim, output_dim, bias=bias)
            # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        elif self.rnn_type == 'GRU':
            self.rnn = GRUCell(input_dim, output_dim, bias=bias)
            #self.gru = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        else:
            raise AttributeError('Unsupported rnn type!')
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj_list):
        if self.rnn_type == 'GRU':
            if torch.cuda.is_available():
                hx = Variable(torch.zeros(x.size()[0], self.output_dim).cuda())
            else:
                hx = Variable(torch.zeros(x.size()[0], self.output_dim))
            adj_list = adj_list[::-1]
            for i, adj in enumerate(adj_list):
                res = F.relu(torch.sparse.mm(adj, x))
                hx = self.rnn(res, hx)
            #Layer normalization could improve performance and make rnn stable
            hx = self.norm(hx)
            return hx
        elif self.rnn_type == 'LSTM':
            if torch.cuda.is_available():
                hx = Variable(torch.zeros(x.size()[0], self.output_dim).cuda())
                cx = Variable(torch.zeros(x.size()[0], self.output_dim).cuda())
            else:
                hx = Variable(torch.zeros(x.size()[0], self.output_dim))
                cx = Variable(torch.zeros(x.size()[0], self.output_dim))
            adj_list = adj_list[::-1]
            for i, adj in enumerate(adj_list):
                res = F.relu(torch.sparse.mm(adj, x))
                hx, cx = self.rnn(res, hx, cx)
            #Layer normalization could improve performance and make rnn stable
            hx = self.norm(hx)
            return hx
        else:
            raise AttributeError('Unsupported rnn type!')

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
        self.h2h = nn.Linear(output_dim, 3 * output_dim, bias=bias)
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

        self.x2h = nn.Linear(input_dim, 4 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 4 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hx, cx):
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True, trans_version='N'):
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
        self.trans_version = trans_version

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

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            x = self.linear(x)
            if self.trans_version == 'L':
                return x
            elif self.trans_version == 'N':
                return F.selu(x)
            else:
                raise ValueError("Unsupported trans version!")
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers):
                if self.trans_version == 'L':
                    h = self.linears[layer](h)
                elif self.trans_version == 'N':
                    h = F.selu(self.linears[layer](h))
                else:
                    raise ValueError("Unsupported trans version!")
            return h