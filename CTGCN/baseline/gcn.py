import torch, math
import torch.nn as nn
import torch.nn.functional as F

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