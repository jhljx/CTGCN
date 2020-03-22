import sys
sys.path.append("..")
import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from CTGCN.layers import CoreDiffusion, MLP, GRUCell, LSTMCell

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

class MLPClassifier(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    duration: int
    bias: bool
    trans_version: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, trans_version='N'):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.duration = duration
        self.bias = bias
        self.trans_version = trans_version

        self.mlp_list = nn.ModuleList()
        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, layer_num, bias=bias, trans_version=trans_version))

    def forward(self, x_list):
        if isinstance(x_list, list):
            output_list = []
            for i in range(len(x_list)):
                x = self.mlp_list[i](x_list[i])
                output_list.append(x)
            return output_list
        x = self.mlp_list[0](x_list)
        return x

class CGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    trans_num: int
    diffusion_num: int
    bias: bool
    rnn_type: str
    version: str
    trans_version: str

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, bias=True, rnn_type='GRU', version='C', trans_version='L'):
        super(CGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type
        self.version = version
        self.trans_version = trans_version

        if self.version == 'C':
            self.mlp = MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, trans_version=trans_version)
            self.duffision = CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type)
        elif self.version == 'S':
            self.mlp = MLP(input_dim, hidden_dim, output_dim, trans_num, bias=bias)
            self.duffision = CDN(output_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type)
        else:
            raise AttributeError('Unsupported CTGCN version!')

    def forward(self, x, adj_list):
        if isinstance(x, list):
            assert len(x) == 1
            x, adj_list = x[0], adj_list[0]
            trans = self.mlp(x)
            x = self.duffision(trans, adj_list)
            return [x], [trans]
        trans = self.mlp(x)
        x = self.duffision(trans, adj_list)
        return x, trans

class CTGCN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    rnn_type: str
    version: str
    trans_version: str
    duration: int
    trans_num: int
    diffusion_num: int
    bias: bool

    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, duration, bias=True, rnn_type='GRU', version='C', trans_version='L'):
        super(CTGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.version = version
        self.trans_version = trans_version
        self.duration = duration
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.mlp_list = nn.ModuleList()
        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            if self.version == 'C':
                self.mlp_list.append(MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, trans_version=trans_version))
                self.duffision_list.append(CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))
            elif self.version == 'S':
                self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, trans_num, bias=bias, trans_version=trans_version))
                self.duffision_list.append(CDN(output_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))
            else:
                raise AttributeError('Unsupported CTGCN version!')
        if self.rnn_type == 'LSTM':
            self.rnn = LSTMCell(output_dim, output_dim, bias=bias)
        elif self.rnn_type == 'GRU':
            self.rnn = GRUCell(output_dim, output_dim, bias=bias)
        else:
            raise AttributeError('unit type error!')

    def forward(self, x_list, adj_list):
        if self.rnn_type == 'GRU':
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
        elif self.rnn_type == 'LSTM':
            if torch.cuda.is_available():
                hx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim).cuda())
                cx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim).cuda())
            else:
                hx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
                cx = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
            trans_list = []
            hx_list = []
            for i in range(len(x_list)):
                x = self.mlp_list[i](x_list[i])
                trans_list.append(x)
                x = self.duffision_list[i](x, adj_list[i])
                hx, cx = self.rnn(x, hx, cx)
                hx_list.append(hx)
            return hx_list, trans_list
        else:
            raise AttributeError('Unsupported rnn type!')