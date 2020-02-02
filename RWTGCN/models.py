import pandas as pd
import scipy.sparse as sp
import os, time
import networkx as nx
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from RWTGCN.layers import GCGRUCell, GCLSTMCell, Readout, Infomax
from RWTGCN.metrics import MainLoss
from RWTGCN.utils import check_and_make_path, get_normalize_PPMI_adj


class RWTGCN(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    unit_type: str
    duration: int
    bias: bool
    readout: Readout

    def __init__(self, input_dim, output_dim, layer_num, dropout, duration, unit_type='GRU', bias=True):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.unit_type = unit_type
        self.duration = duration
        self.bias = bias
        if self.unit_type == 'GRU':
            self.rnn_cell = GCGRUCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        elif self.unit_type == 'LSTM':
            self.rnn_cell = GCLSTMCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        else:
            raise AttributeError('unit type error!')
        self.readout = Readout()
        self.infomax_list = []
        for i in range(duration):
            self.infomax_list.append(Infomax(output_dim, bias=bias))

    def forward(self, x_list, adj_list):
        assert len(x_list) == self.duration and len(adj_list) == self.duration
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x_list[0].size(0), self.output_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x_list[0].size(0), self.output_dim))
        hx_list, score_list = [], []
        hx = h0
        for i in range(len(x_list)):
            hx = self.rnn_cell(x_list[i], adj_list[i], hx)
            hx_list.append(hx)

            shuffle_hx = hx.clone().detach()

            def shuffle_list(a):
                return a[torch.randperm(a.size(0))]

            output = list(map(shuffle_list, torch.unbind(shuffle_hx, 1)))
            shuffle_hx = torch.stack(output, 1)
            c = self.readout(hx)
            pos_score, neg_score = self.infomax_list[i](c, hx, shuffle_hx)
            score_list.append((pos_score, neg_score))

        return hx_list, score_list


class DynamicEmbedding:
    base_path: str
    input_base_path: str
    embedding_base_path: str
    model_base_path: str
    full_node_list: list

    timestamp_list: list
    node_file: str
    full_node_set: list
    node_num: int
    output_dim: int
    duration: int
    layer_num: int
    model: RWTGCN
    device: torch.device

    def __init__(self, base_path, input_folder, embedding_folder, node_file, output_dim, model_folder="model",
                 dropout=0.5, duration=-1, unit_type='GRU', bias=True):

        # file paths
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)  # tensor folder
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.model_base_path = os.path.join(base_path, model_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)  # node num
        self.output_dim = output_dim
        self.layer_num = len(os.listdir(os.path.join(self.input_base_path, os.listdir(self.input_base_path)[0])))
        self.timestamp_list = os.listdir(self.input_base_path)

        if duration == -1:
            self.duration = len(self.timestamp_list)
        else:
            assert duration > 0
            self.duration = duration
        # cpu gpu
        if torch.cuda.is_available():
            print("GPU")
            device = torch.device("cuda: 0")
        else:
            print("CPU")
            device = torch.device("cpu")
        self.device = device

        self.model = RWTGCN(self.node_num, self.output_dim, self.layer_num, dropout=dropout, duration=duration,
                            unit_type=unit_type, bias=bias)

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def get_date_adj_list(self, start_idx):
        date_dir_list = sorted(os.listdir(self.input_base_path))
        time_stamp_num = len(date_dir_list)
        assert start_idx < time_stamp_num

        date_adj_list = []

        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            date_dir_path = os.path.join(self.input_base_path, date_dir_list[i])
            f_list = os.listdir(date_dir_path)
            walk_length = len(f_list)
            adj_list = []

            for i, f_name in enumerate(f_list):
                print("\t\t" + str(walk_length - i) + "file(s) left")
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                adj_list.append(get_normalize_PPMI_adj(spmat))
            date_dir_list.append(adj_list)
        return date_adj_list

    # def _adjust_learning_rate(self, optimizer, epoch, initial_lr):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     lr = initial_lr * 0.1
    #     print('epoch', epoch + 1, 'learn rate', lr)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    def learn_embedding(self, epoch=50, lr=1e-3, start_idx=0, weight_decay=0, export=True):

        adj_list = self.get_date_adj_list(start_idx)
        time_stamp_num = len(adj_list)
        x_list = [sp.eye(self.node_num) for i in range(time_stamp_num)]

        if torch.cuda.is_available():
            device = torch.device("cuda: 0")
            model = self.model.to(device)
        else:
            model = self.model

        # 创建优化器（optimizer）
        # optimizer = optim.SGD(embedding_net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-3, weight_decay=weight_decay)
        train_loss = []
        criterion = MainLoss()
        embedding_list = []

        for i in range(epoch):
            ## 1. forward propagation
            embedding_list, score_list = model(x_list, adj_list)
            ## 2. loss calculation
            loss = criterion(score_list)
            optimizer.zero_grad()  # 清零梯度缓存
            ## 3. backward propagation
            loss.backward()
            ## 4. weight optimization
            optimizer.step()  # 更新参数
            train_loss.append(loss.item())
            print("epoch", i + 1, "loss:", loss)

        if export:
            for i in range(len(embedding_list)):
                embedding = embedding_list[i]
                timestamp = self.timestamp_list[start_idx + i]
                df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_set)
                embedding_path = os.path.join(self.embedding_base_path, timestamp + ".csv")
                df_export.to_csv(embedding_path, sep='\t', header=True, index=True)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.model_base_path, "RWTGCN_model"))
        return embedding_list
