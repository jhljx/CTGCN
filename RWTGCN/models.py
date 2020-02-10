import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, time, json, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.append("..")
from RWTGCN.layers import GCGRUCell, GCLSTMCell
from RWTGCN.metrics import MainLoss
from RWTGCN.utils import check_and_make_path, get_normalize_PPMI_adj, sparse_mx_to_torch_sparse_tensor

class RWTGCN(nn.Module):
    input_dim: int
    output_dim: int
    dropout: float
    unit_type: str
    duration: int
    layer_num: int
    bias: bool

    def __init__(self, input_dim, output_dim, layer_num, dropout, duration, unit_type='GRU', bias=True):
        super(RWTGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.unit_type = unit_type
        self.duration = duration
        self.layer_num = layer_num
        self.bias = bias
        if self.unit_type == 'GRU':
            self.rnn_cell = GCGRUCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        elif self.unit_type == 'LSTM':
            self.rnn_cell = GCLSTMCell(input_dim, output_dim, layer_num, dropout, bias=bias)
        else:
            raise AttributeError('unit type error!')

    def forward(self, x_list, adj_list):
        if torch.cuda.is_available():
            assert len(x_list) == self.duration and len(adj_list) == self.duration * self.layer_num
            # print('gpu input size: ', x_list[0].size())
            h0 = Variable(torch.zeros(x_list[0].size()[1], self.output_dim).cuda())
        else:
            assert len(x_list) == self.duration and len(adj_list) == self.duration
            # print('cpu input size: ', x_list[0].size())
            h0 = Variable(torch.zeros(x_list[0].size()[0], self.output_dim))
        hx_list = []
        hx = h0
        for i in range(len(x_list)):
            if torch.cuda.is_available():
                temp_adj_list = adj_list[i * self.layer_num: (i + 1) * self.layer_num]
                # temp_adj_list = nn.ParameterList()
                # for j in range(i * self.layer_num, (i + 1) * self.layer_num):
                #     temp_adj_list.append(adj_list[j])
                hx = self.rnn_cell(x_list[i], temp_adj_list, hx)
            else:
                hx = self.rnn_cell(x_list[i], adj_list[i], hx)
            hx_list.append(hx)
        return hx_list

class DynamicEmbedding:
    base_path: str
    walk_base_path: str
    freq_base_path: str
    tensor_base_path: str
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

    def __init__(self, base_path, walk_folder, freq_folder, tensor_folder, embedding_folder, node_file, output_dim, model_folder="model",
                 dropout=0.5, duration=-1, neg_num=50, Q=10, unit_type='GRU', bias=True):

        # file paths
        self.base_path = base_path
        self.walk_base_path = os.path.join(base_path, walk_folder)
        self.freq_base_path = os.path.join(base_path, freq_folder)
        self.tensor_base_path = os.path.join(base_path, tensor_folder)
        self.embedding_base_path = os.path.join(base_path, embedding_folder)
        self.model_base_path = os.path.join(base_path, model_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)  # node num
        self.output_dim = output_dim
        self.layer_num = len(os.listdir(os.path.join(self.tensor_base_path, os.listdir(self.tensor_base_path)[0])))
        self.timestamp_list = os.listdir(self.tensor_base_path)

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

        self.model = RWTGCN(self.node_num, self.output_dim, self.layer_num, dropout=dropout, duration=self.duration,
                            unit_type=unit_type, bias=bias)
        self.loss = MainLoss(neg_num=neg_num, Q=Q)

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def get_date_adj_list(self, start_idx):
        date_dir_list = sorted(os.listdir(self.tensor_base_path))
        time_stamp_num = len(date_dir_list)
        assert start_idx < time_stamp_num

        #nn.Parameter(adj, requires_grad=False)
        #for adj in adj_list_item])
        if torch.cuda.is_available():
            date_adj_list = nn.ParameterList()
        else:
            date_adj_list = []
        # node_pair_list = []
        # node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))

        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            date_dir_path = os.path.join(self.tensor_base_path, date_dir_list[i])
            f_list = os.listdir(date_dir_path)
            #if torch.cuda.is_available():
            #    adj_list = nn.ParameterList()
            #else:
            adj_list = []
            #edge_set = set()
            for i, f_name in enumerate(f_list):
                # print("\t\t" + str(walk_length - i) + "file(s) left")
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                #rows, cols = spmat.nonzero()
                #edge_set |= set(list(zip(rows, cols)))
                sptensor = get_normalize_PPMI_adj(spmat)
                if torch.cuda.is_available():
                    date_adj_list.append(nn.Parameter(sptensor, requires_grad=False))
                else:
                    adj_list.append(sptensor)
            if not torch.cuda.is_available():
                date_adj_list.append(adj_list)
        date_adj_list = date_adj_list.to(self.device)
            # neighbor_dict = dict()
            # edge_list = list(edge_set)
            # for edge in edge_list:
            #     if edge[0] not in neighbor_dict:
            #         neighbor_dict[self.full_node_list[edge[0]]] = [edge[1]]
            #     else:
            #         neighbor_dict[self.full_node_list[edge[0]]].append(edge[1])
            # node_pair_list.append(neighbor_dict)
        return date_adj_list

    def get_node_pair_list(self, start_idx):
        walk_file_list = sorted(os.listdir(self.walk_base_path))
        time_stamp_num = len(walk_file_list)
        assert start_idx < time_stamp_num

        node_pair_list = []
        node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            walk_file_path = os.path.join(self.walk_base_path, walk_file_list[i])
            with open(walk_file_path, 'r') as fp:
                #t1 = time.time()
                edge_list = json.load(fp)
                # t2 = time.time()
                # print('read file time: ', t2 - t1, ' seconds!')
                neighbor_dict = dict()
                for edge in edge_list:
                    if edge[0] not in neighbor_dict:
                        neighbor_dict[edge[0]] = [node2idx_dict[edge[1]]]
                    else:
                        neighbor_dict[edge[0]].append(node2idx_dict[edge[1]])
                node_pair_list.append(neighbor_dict)
                #t3 = time.time()
                #print('transform time: ', t3 - t2, ' seconds!')
                # graph = nx.Graph()
                # graph.add_edges_from(edge_list)
                # t3 = time.time()
                # print('add edges time: ', t3 - t2, ' seconds!')
                # # convert graph to dict {node: neighbor_list}
                # node_pair_list.append(nx.to_dict_of_lists(graph))
                # t4 = time.time()
                # print('to list time: ', t4 - t3, ' seconds!')
                # del graph
        return node_pair_list, node2idx_dict

    def get_neg_freq_list(self, start_idx):
        freq_file_list = sorted(os.listdir(self.freq_base_path))
        time_stamp_num = len(freq_file_list)
        assert start_idx < time_stamp_num

        node_freq_list = []
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            freq_file_path = os.path.join(self.freq_base_path, freq_file_list[i])
            with open(freq_file_path, 'r') as fp:
                node_freq_list.append(json.load(fp))
        return node_freq_list


    # def _adjust_learning_rate(self, optimizer, epoch, initial_lr):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     lr = initial_lr * 0.1
    #     print('epoch', epoch + 1, 'learn rate', lr)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    def learn_embedding(self, epoch=50, batch_size=10240, lr=1e-3, start_idx=0, weight_decay=0., export=True):
        t1 = time.time()
        adj_list = self.get_date_adj_list(start_idx)
        t2 = time.time()
        print('get adj list finish! cost time: ', t2 - t1, ' seconds!')
        node_pair_list, node2idx_dict = self.get_node_pair_list(start_idx)
        t3 = time.time()
        print('get node pair list finish! cost time: ', t3 - t2, ' seconds!')
        neg_freq_list = self.get_neg_freq_list(start_idx)
        t4 = time.time()
        print('get neg freq list finish! cost time: ', t4 - t3, ' seconds!')
        self.loss.set_node_info(node_pair_list, neg_freq_list, node2idx_dict)
        #t5 = time.time()
        #print("prepare finish! cost time: ", t5 - t4, ' seconds!')
        # print('time stamp num: ', time_stamp_num)
        x_list = [sparse_mx_to_torch_sparse_tensor(sp.eye(self.node_num)) for i in range(self.duration)]
        # print('x list len: ', len(x_list))
        model = self.model
        if torch.cuda.is_available():
            model = model.to(self.device)
            x_list_tensor = nn.ParameterList([nn.Parameter(x, requires_grad=False) for x in x_list])
            x_list = x_list_tensor.to(self.device)
            torch.cuda.empty_cache()

        # 创建优化器（optimizer）
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        # train_loss = []

        embedding_list = []
        batch_num = self.node_num // batch_size
        if self.node_num % batch_size != 0:
            batch_num += 1
        # node_list = self.full_node_list.copy()
        st = time.time()
        train_loss = []
        flag = False
        for i in range(epoch):
            node_list = np.random.permutation(self.full_node_list)
            for j in range(batch_num):
                ## 1. forward propagation
                t1 = time.time()
                embedding_list = model(x_list, adj_list)
                t2 = time.time()
                print('forward time: ', t2 - t1, ' seconds!')
                # print('finish forward!')
                batch_nodes = node_list[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                ## 2. loss calculation
                loss = self.loss(embedding_list, batch_nodes)
                t3 = time.time()
                print('loss calc time: ', t3 - t2, ' seconds!')
                ## 3. backward propagation
                loss.backward()
                t4 = time.time()
                print('backward time: ', t4 - t3, ' seconds!')
                # gradient accumulation
                if j == batch_num - 1:  # 重复多次前面的过程
                    optimizer.step()  # 更新梯度
                    model.zero_grad()
                ## 4. weight optimization
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清零梯度缓存
                torch.cuda.empty_cache()
                if len(train_loss) > 0:
                    if np.abs(train_loss[-1] - loss.item()) < 1e-12:
                        flag = 1
                        break
                train_loss.append(loss.item())
                t5 = time.time()
                print("epoch", i + 1, ', batch num = ', j + 1, ", loss:", loss.item(), ', cost time: ', t5 - t1, ' seconds!')
            if flag:
                break
        en = time.time()
        print('training total time: ',en - st, ' seconds!')
        if export:
            for i in range(len(embedding_list)):
                embedding = embedding_list[i]
                timestamp = self.timestamp_list[start_idx + i]
                df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
                embedding_path = os.path.join(self.embedding_base_path, timestamp + ".csv")
                df_export.to_csv(embedding_path, sep='\t', header=True, index=True)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.model_base_path, "RWTGCN_model"))
        return embedding_list

if __name__ == '__main__':
    thread_num = os.cpu_count() - 4
    torch.set_num_threads(thread_num)
    dyEmbedding = DynamicEmbedding(base_path="../../data/facebook/RWT-GCN", walk_folder='walk_pairs',
                                   freq_folder='node_freq', tensor_folder="walk_tensor",
                                   embedding_folder="embedding", node_file="../nodes_set/nodes.csv",
                                   output_dim=128, dropout=0.5, duration=5, neg_num=20, Q=10,
                                   unit_type='GRU', bias=True)
    # dyEmbedding = DynamicEmbedding(base_path="..\\data\\email-eu\\RWT-GCN", walk_folder='walk_pairs',
    #                                freq_folder='node_freq',  tensor_folder="walk_tensor",
    #                                embedding_folder="embedding", node_file="..\\nodes_set\\nodes.csv",
    #                                output_dim=128, dropout=0.5, duration=5, neg_num=50, Q=10,
    #                                unit_type='GRU', bias=True)
    dyEmbedding.learn_embedding(epoch=10, batch_size=1024*5, lr=0.001, start_idx=0, weight_decay=0.0005, export=True)