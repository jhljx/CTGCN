import sys
sys.path.append("..")
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
        # return output, trans
        #return trans + output
        # return torch.cat([trans, output], dim=1)
        # return trans * output
        #return gate1 * trans + gate2 * output
        return output + gate * (trans - output),  trans + gate * (output - trans)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
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
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'