import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from RWTGCN.layers import CoreDiffusion, GraphConvolution, Linear, MLP, GRUCell, LSTMCell

class CGDN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, rnn_type='GRU'):
        super(CGDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.rnn_type = rnn_type

        self.diffusion1 = CoreDiffusion(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type)
        self.diffusion2 = CoreDiffusion(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type)


    def forward(self, x, adj_list):
        x = self.diffusion1(x, adj_list)
        x = self.diffusion2(x, adj_list)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden_dim, bias=bias)
        self.gc2 = GraphConvolution(hidden_dim, output_dim, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        # GCN for static embedding
        if isinstance(x, list):
            assert len(x) == 1
            hx_list = []
            # if torch.cuda.is_available():
            #     x1 = F.relu(self.gc1(x[0].cuda(), adj[0].cuda()))
            # else:
            x1 = F.relu(self.gc1(x[0], adj[0]))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj[0])
            hx_list.append(x2)
            return hx_list
        # x is a sparse tensor, component of other model
        # if torch.cuda.is_available():
        #     x = F.relu(self.gc1(x.cuda(), adj.cuda()))
        # else:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class RWTGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    rnn_type: str
    duration: int
    layer_num: int
    bias: bool

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, rnn_type='GRU'):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.duration = duration
        self.layer_num = layer_num
        self.bias = bias

        self.mlp_list = nn.ModuleList()
        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, layer_num, bias=bias))
            self.duffision_list.append(CGDN(output_dim, output_dim, output_dim, rnn_type=rnn_type))
        if self.rnn_type == 'LSTM':
            self.rnn = LSTMCell(output_dim, output_dim, bias=bias)
            # self.lstm = nn.LSTM(input_size=output_dim, hidden_size=output_dim, num_layers=1, bidirectional=False)
        elif self.rnn_type == 'GRU':
            self.rnn = GRUCell(output_dim, output_dim, bias=bias)
            # self.gru = nn.GRU(input_size=output_dim, hidden_size=output_dim, num_layers=1, bidirectional=False)
        else:
            raise AttributeError('unit type error!')
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, adj_list):
        if torch.cuda.is_available():
            hx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim).cuda())
        else:
            hx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
        trans_list = []
        hx_list = []
        # reconstruct_list = []
        for i in range(len(x_list)):
            x = self.mlp_list[i](x_list[i])
            trans_list.append(x)
            x = self.duffision_list[i](x, adj_list[i])
            hx = self.rnn(x, hx)
            hx_list.append(hx)
        return hx_list, trans_list# , reconstruct_list