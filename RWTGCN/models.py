import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from RWTGCN.layers import CoreDiffusion, MLP, GRUCell, LSTMCell

class CDN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    diffusion_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super(CDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type

        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, output_dim, bias=bias, rnn_type=rnn_type))
        elif diffusion_num > 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(CoreDiffusion(hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(CoreDiffusion(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            raise ValueError("number of layers should be positive!")

    def forward(self, x, adj_list):
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x


class CGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    trans_num: int
    diffusion_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, bias=True, rnn_type='GRU'):
        super(CGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type

        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers=trans_num, bias=bias)

        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(output_dim, output_dim, bias=bias, rnn_type=rnn_type))
        elif diffusion_num > 1:
            self.diffusion_list = nn.ModuleList()
            for i in range(diffusion_num):
                self.diffusion_list.append(CoreDiffusion(output_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            raise ValueError("number of layers should be positive!")

    def forward(self, x, adj_list):
        if isinstance(x, list):
            assert len(x) == 1
            x, adj_list = x[0], adj_list[0]
            trans = self.mlp(x)
            x = trans
            for i in range(self.diffusion_num):
                x = self.diffusion_list[i](x, adj_list)
            return x, trans
        trans = self.mlp(x)
        x = trans
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x, trans


class RWTGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    rnn_type: str
    duration: int
    trans_num: int
    diffusion_num: int
    bias: bool

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, duration, bias=True, rnn_type='GRU'):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.duration = duration
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.mlp_list = nn.ModuleList()
        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, trans_num, bias=bias))
            self.duffision_list.append(CDN(output_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))
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
        for i in range(len(x_list)):
            x = self.mlp_list[i](x_list[i])
            trans_list.append(x)
            x = self.duffision_list[i](x, adj_list[i])
            hx = self.rnn(x, hx)
            hx_list.append(hx)
        return hx_list, trans_list