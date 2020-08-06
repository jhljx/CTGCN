# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# K-core subgraph based diffusion layer
class CoreDiffusion(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.core_num = core_num
        self.rnn_type = rnn_type

        self.linear = nn.Linear(input_dim, output_dim)
        # self.att_weight = nn.Parameter(torch.FloatTensor(core_num))
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            # self.lstm = LSTMCell(input_dim, output_dim, bias=bias)
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            # self.gru = GRUCell(input_dim, output_dim, bias=bias)
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        # self.norm = nn.BatchNorm1d(output_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.att_weight.data.uniform_(0, 1)

    def forward(self, x, adj_list):
        hx_list = []
        # output = None
        for i, adj in enumerate(adj_list):
            if i == 0:
                res = torch.sparse.mm(adj, x)
            else:
                res = hx_list[-1] + torch.sparse.mm(adj, x)
            # hx = self.linear(res)
            hx_list.append(res)
        hx_list = [F.relu(res) for res in hx_list]
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)
        # Layer normalization could improve performance and make rnn stable
        output = self.norm(output)
        return output


# Multi-Layer Perceptron(MLP) 'layer'
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias=True, activate_type='N'):
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
        self.activate_type = activate_type
        assert self.activate_type in ['L', 'N']

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            # self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            #for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            x = self.linear(x)
            if self.activate_type == 'L':
                return x
            else:  # activate_type == 'N'
                return F.selu(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers):
                if self.activate_type == 'L':
                    h = self.linears[layer](h)
                else:
                    h = F.selu(self.linears[layer](h))
            return h
