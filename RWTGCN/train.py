import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, time, json, sys
sys.path.append("..")
import torch
import torch.nn as nn
from RWTGCN.metrics import MainLoss
from RWTGCN.models import GCN, MRGCN, RWTGCN
from RWTGCN.utils import check_and_make_path, get_normalize_PPMI_adj, sparse_mx_to_torch_sparse_tensor, build_graph, get_sp_adj_mat, separate

class DynamicEmbedding:
    base_path: str
    walk_pair_base_path: str
    node_freq_base_path: str
    walk_tensor_base_path: str
    embedding_base_path: str
    origin_base_path: str
    model_base_path: str
    full_node_list: list

    timestamp_list: list
    node_file: str
    full_node_set: list
    node_num: int
    output_dim: int
    duration: int
    walk_len: int
    gcn_type: str
    #model: RWTGCN
    device: torch.device

    def __init__(self, base_path, walk_pair_folder, node_freq_folder, walk_tensor_folder, embedding_folder, node_file, origin_folder='', model_folder="model",
                 output_dim=128, hid_num=500, dropout=0.5, duration=-1, neg_num=50, Q=10, gcn_type='RWTGCN', unit_type='GRU', bias=True):

        # file paths
        self.base_path = base_path
        self.walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        #print('walk pair: ', self.walk_pair_base_path)
        self.node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        #print('node freq: ', self.node_freq_base_path)
        self.walk_tensor_base_path = '' if walk_tensor_folder == '' else os.path.abspath(os.path.join(base_path, walk_tensor_folder))
        print('walk tensor: ', self.walk_tensor_base_path)
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        print('embedding: ', self.embedding_base_path)
        self.origin_base_path = '' if origin_folder == '' else os.path.abspath(os.path.join(base_path, origin_folder))
        print('origin graph: ', self.origin_base_path)
        self.model_base_path = os.path.abspath(os.path.join(base_path, model_folder))
        #print('model: ', self.model_base_path)

        node_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)  # node num
        self.output_dim = output_dim
        self.timestamp_list = sorted(os.listdir(self.walk_pair_base_path))
        # print('timestamp list:', self.timestamp_list)

        if duration == -1:
            self.duration = len(self.timestamp_list)
        else:
            assert duration > 0
            self.duration = duration
        # cpu gpu
        if torch.cuda.is_available():
            print("GPU")
            device = torch.device("cuda: 0")
            self.set_thread()
        else:
            print("CPU")
            device = torch.device("cpu")
        self.device = device
        self.gcn_type = gcn_type


        if gcn_type == 'RWTGCN':
            self.walk_len = len(os.listdir(os.path.join(self.walk_tensor_base_path, os.listdir(self.walk_tensor_base_path)[0])))
            print('walk_len: ', self.walk_len)
            self.model = RWTGCN(self.node_num, self.output_dim, self.walk_len, dropout=dropout, duration=self.duration,
                                unit_type=unit_type, bias=bias)
        elif gcn_type == 'MRGCN':
            assert self.duration == 1 # duration must be 1 when calling static embeding
            self.walk_len = len(os.listdir(os.path.join(self.walk_tensor_base_path, os.listdir(self.walk_tensor_base_path)[0])))
            print('walk_len: ', self.walk_len)
            self.model = MRGCN(self.node_num, self.output_dim, self.walk_len, dropout=dropout, bias=bias)
        elif gcn_type == 'GCN':
            assert self.duration == 1  # duration must be 1 when calling static embeding
            self.model = GCN(self.node_num, hid_num, self.output_dim, dropout=dropout, bias=bias)
        self.loss = MainLoss(neg_num=neg_num, Q=Q)

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def set_thread(self, thread_num=None):
        if thread_num is None:
            thread_num = os.cpu_count() - 4
        torch.set_num_threads(thread_num)

    def get_date_adj_list(self, start_idx):
        if self.gcn_type == 'GCN':
            date_dir_list = sorted(os.listdir(self.origin_base_path))
            original_graph_path = os.path.join(self.origin_base_path, date_dir_list[start_idx])
            date_adj_list = []
            spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep='\t')
            sptensor = get_normalize_PPMI_adj(spmat)
            if torch.cuda.is_available():
                date_adj_list.append(sptensor.cuda())
            else:
                date_adj_list.append(sptensor)
            return date_adj_list
        # gcn_type == 'MRGCN' or 'RWTGCN'
        date_dir_list = sorted(os.listdir(self.walk_tensor_base_path))
        time_stamp_num = len(date_dir_list)
        assert start_idx < time_stamp_num
        date_adj_list = []
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            date_dir_path = os.path.join(self.walk_tensor_base_path, date_dir_list[i])
            f_list = sorted(os.listdir(date_dir_path))

            tmp_adj_list, adj_list = [], []
            for i, f_name in enumerate(f_list):
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                sptensor = get_normalize_PPMI_adj(spmat)
                if self.gcn_type == 'RWTGCN':
                    adj_list = tmp_adj_list
                else: # MRGCN
                    adj_list = date_adj_list
                adj_list.append(sptensor.cuda()  if torch.cuda.is_available() else sptensor)
            if self.gcn_type == 'RWGCN':
                date_adj_list.append(adj_list)
        return date_adj_list

    def get_node_pair_list(self, start_idx):
        walk_file_list = sorted(os.listdir(self.walk_pair_base_path))
        time_stamp_num = len(walk_file_list)
        assert start_idx < time_stamp_num

        node_pair_list = []
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            walk_file_path = os.path.join(self.walk_pair_base_path, walk_file_list[i])
            graph_dict = build_graph(walk_file_path, self.full_node_list, sep='\t', save_weight=False)
            node_pair_list.append(graph_dict)
        return node_pair_list

    def get_neg_freq_list(self, start_idx):
        freq_file_list = sorted(os.listdir(self.node_freq_base_path))
        time_stamp_num = len(freq_file_list)
        assert start_idx < time_stamp_num

        node_freq_list = []
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            freq_file_path = os.path.join(self.node_freq_base_path, freq_file_list[i])
            with open(freq_file_path, 'r') as fp:
                node_freq_list.append(json.load(fp))
        return node_freq_list

    def learn_embedding(self, epoch=50, batch_size=10240, lr=1e-3, start_idx=0, weight_decay=0., model_file='rwtgcn', export=True):
        t1 = time.time()
        adj_list = self.get_date_adj_list(start_idx)
        t2 = time.time()
        print('get adj list finish! cost time: ', t2 - t1, ' seconds!')
        node_pair_list = self.get_node_pair_list(start_idx)
        t3 = time.time()
        print('get node pair list finish! cost time: ', t3 - t2, ' seconds!')
        neg_freq_list = self.get_neg_freq_list(start_idx)
        t4 = time.time()
        print('get neg freq list finish! cost time: ', t4 - t3, ' seconds!')
        self.loss.set_node_info(node_pair_list, neg_freq_list)
        #t5 = time.time()
        #print("prepare finish! cost time: ", t5 - t4, ' seconds!')
        # print('time stamp num: ', time_stamp_num)
        x_list = [sparse_mx_to_torch_sparse_tensor(sp.eye(self.node_num)) for i in range(self.duration)]
        # print('x list len: ', len(x_list))
        model = self.model
        if torch.cuda.is_available():
            model = model.to(self.device)
            x_list = [x.cuda() for x in x_list]
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
            node_idx_list = np.random.permutation(np.arange(self.node_num))
            for j in range(batch_num):
                ## 1. forward propagation
                t1 = time.time()
                embedding_list = model(x_list, adj_list)
                t2 = time.time()
                print('forward time: ', t2 - t1, ' seconds!')
                # print('finish forward!')
                batch_node_idxs = node_idx_list[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                ## 2. loss calculation
                loss = self.loss(embedding_list, batch_node_idxs)
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
                timestamp = self.timestamp_list[start_idx + i].split('.')[0]
                df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
                embedding_path = os.path.join(self.embedding_base_path, timestamp + ".csv")
                df_export.to_csv(embedding_path, sep='\t', header=True, index=True)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        return embedding_list

def static_embedding():
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/facebook/RWT-GCN'))
    print(base_path)
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/GCN')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    t1 = time.time()
    print('start GCN embedding!')
    GCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='gcn_walk_pairs',
                                   node_freq_folder='gcn_node_freq', walk_tensor_folder='', embedding_folder=embedding_folder,
                                   node_file=node_file, origin_folder=origin_folder, model_folder='model',
                                   output_dim=128, hid_num=500, dropout=0.5, duration=1, neg_num=20, Q=10, gcn_type = 'GCN', bias=True)
    timestamp_num = len(GCN.timestamp_list)
    for idx in range(timestamp_num):
        GCN.learn_embedding(epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', export=True)
    t2 = time.time()
    print('finish GCN embedding! cost time: ', t2 - t1, ' seconds!')
    separate()
    t1 = time.time()
    print('start MRGCN embedding!')
    embedding_folder = os.path.join('..', '2.embedding/MRGCN')
    MRGCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='walk_pairs',
                                    node_freq_folder='node_freq', walk_tensor_folder="walk_tensor", embedding_folder=embedding_folder,
                                    node_file=node_file, origin_folder='', model_folder='model',
                                    output_dim=128, dropout=0.5, duration=1, neg_num=20, Q=10, gcn_type='MRGCN', bias=True)
    timestamp_num = len(MRGCN.timestamp_list)
    for idx in range(timestamp_num):
        MRGCN.learn_embedding(epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='mrgcn', export=True)
    t2 = time.time()
    print('finish MRGCN embedding! cost time: ', t2 - t1, ' seconds!')

def dynamic_embedding():
    dyEmbedding = DynamicEmbedding(base_path="../../data/facebook/RWT-GCN", walk_folder='walk_pairs',
                                   freq_folder='node_freq', tensor_folder="walk_tensor",
                                   embedding_folder="embedding", node_file="../nodes_set/nodes.csv",
                                   output_dim=128, dropout=0.5, duration=5, neg_num=20, Q=10,
                                   unit_type='GRU', bias=True)
    dyEmbedding.learn_embedding(epoch=10, batch_size=1024 * 5, lr=0.001, start_idx=0, weight_decay=0.0005, export=True)

if __name__ == '__main__':

   static_embedding()


