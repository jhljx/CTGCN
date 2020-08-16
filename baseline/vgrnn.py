# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch_scatter
import inspect


# Variational Graph Recurrent Networks. For more information, please refer to https://arxiv.org/abs/1908.09710
# We modify and simplify the code of VGRNN from https://github.com/VGraphRNN/VGRNN, and include this method in our graph embedding project framework.
# Author: jhljx
# Email: jhljx8918@gmail.com


# utility functions
def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0
    out = op(src, index, 0, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]

    if name is 'max':
        out[out == fill_value] = 0

    return out


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


# layers

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_weight = torch.full((x.size(0),), 1 if not self.improved else 2, dtype=x.dtype, device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]
        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_index, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pool='mean', act=F.relu, normalize=False, bias=False):
        super(SAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.act = act
        self.pool = pool

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index

        if self.pool == 'mean':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_mean(out[col], row, dim=0, dim_size=out.size(0))

        elif self.pool == 'max':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out, _ = scatter_max(out[col], row, dim=0, dim_size=out.size(0))

        elif self.pool == 'add':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        else:
            print('pooling not defined!')

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GINConv(torch.nn.Module):
    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index

        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class graph_gru_sage(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_sage, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = nn.ModuleList()
        self.weight_hz = nn.ModuleList()
        self.weight_xr = nn.ModuleList()
        self.weight_hr = nn.ModuleList()
        self.weight_xh = nn.ModuleList()
        self.weight_hh = nn.ModuleList()

        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size(), device=h.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
                #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out


class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = nn.ModuleList()
        self.weight_hz = nn.ModuleList()
        self.weight_xr = nn.ModuleList()
        self.weight_hr = nn.ModuleList()
        self.weight_xh = nn.ModuleList()
        self.weight_hh = nn.ModuleList()

        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size(), device=h.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
                #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        # out = self.decoder(h_t.view(1,-1))

        out = h_out
        return out, h_out


# Inner product decoder(Only apply for small graphs and can not be apply into large scale graphs)
# This decoder is memory-consuming!
class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


# VGRNN model
class VGRNN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    rnn_layer_num: int
    conv_type: str
    bias: bool
    method_name: str

    def __init__(self, input_dim, hidden_dim, output_dim, rnn_layer_num, conv_type='GCN', bias=True):
        super(VGRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_layer_num = rnn_layer_num
        self.conv_type = conv_type
        self.bias = bias
        self.method_name = 'VGRNN'

        assert conv_type in ['GCN', 'SAGE', 'GIN']
        if conv_type == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(output_dim, hidden_dim, bias=bias), nn.ReLU())

            self.enc = GCNConv(hidden_dim + hidden_dim, hidden_dim, bias=bias)
            self.enc_mean = GCNConv(hidden_dim, output_dim, act=lambda x: x, bias=bias)
            self.enc_std = GCNConv(hidden_dim, output_dim, act=F.softplus, bias=bias)

            self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias))
            self.prior_std = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias), nn.Softplus())

            self.rnn = graph_gru_gcn(hidden_dim + hidden_dim, hidden_dim, rnn_layer_num, bias=bias)

        elif conv_type == 'SAGE':
            self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(output_dim, hidden_dim, bias=bias), nn.ReLU())

            self.enc = SAGEConv(hidden_dim + hidden_dim, hidden_dim, bias=bias)
            self.enc_mean = SAGEConv(hidden_dim, output_dim, act=lambda x: x, bias=bias)
            self.enc_std = SAGEConv(hidden_dim, output_dim, act=F.softplus, bias=bias)

            self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias))
            self.prior_std = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias), nn.Softplus())

            self.rnn = graph_gru_sage(hidden_dim + hidden_dim, hidden_dim, rnn_layer_num, bias=bias)

        else:  # 'GIN':
            self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(output_dim, hidden_dim, bias=bias), nn.ReLU())

            self.enc = GINConv(nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim, bias=bias), nn.ReLU()))
            self.enc_mean = GINConv(nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias)))
            self.enc_std = GINConv(nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias), nn.Softplus()))

            self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias))
            self.prior_std = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=bias), nn.Softplus())

            self.rnn = graph_gru_gcn(hidden_dim + hidden_dim, hidden_dim, rnn_layer_num, bias=bias)

        self.dec = InnerProductDecoder(act=lambda x: x)

    def forward(self, x_list, edge_idx_list, hx=None):
        assert len(x_list) == len(edge_idx_list)
        timestamp_num = len(x_list)

        if hx is None:
            h = Variable(torch.zeros(self.rnn_layer_num, self.input_dim, self.hidden_dim, device=x_list[0].device))
        else:
            h = Variable(hx)
        loss_data_list = [[], [], [], [], []]

        embedding_list = []
        for t in range(timestamp_num):
            phi_x_t = self.phi_x(x_list[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t])

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            # decoder
            dec_t = self.dec(z_t)
            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            # add embedding matrix of each timestamp
            embedding_list.append(enc_mean_t)

            # add loss related data for variational autoencoder loss module
            loss_data_list[0].append(enc_mean_t)
            loss_data_list[1].append(enc_std_t)
            loss_data_list[2].append(prior_mean_t)
            loss_data_list[3].append(prior_std_t)
            loss_data_list[4].append(dec_t)

        return embedding_list, h, loss_data_list

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    @staticmethod
    def _reparameterized_sample(mean, std):
        gaussian = torch.randn(std.size(), device=mean.device)
        sample = Variable(gaussian)
        return sample.mul(std).add_(mean)
