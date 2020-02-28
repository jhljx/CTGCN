import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, time, json, sys
import gc
sys.path.append("..")
import torch
import torch.nn as nn
from RWTGCN.metrics import MainLoss
from RWTGCN.baseline import EvolveGCN
from RWTGCN.models import GCN, MRGCN, RWTGCN
from RWTGCN.utils import check_and_make_path, get_walk_neighbor_dict, get_normalize_adj_tensor, sparse_mx_to_torch_sparse_tensor
from RWTGCN.utils import build_graph, get_sp_adj_mat, separate
from RWTGCN.evaluation.link_prediction import LinkPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2,3'

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
    hid_num: int
    dropout: float
    duration: int
    walk_len: int
    gcn_type: str
    egcn_type: str
    unit_type: str
    bias: bool
    #model: RWTGCN
    device: torch.device

    def __init__(self, base_path, walk_pair_folder, node_freq_folder, walk_tensor_folder, embedding_folder, node_file, origin_folder='', model_folder="model",
                 output_dim=128, hid_num=500, dropout=0.5, duration=-1, neg_num=50, Q=10, gcn_type='RWTGCN', egcn_type='EGCNO', unit_type='GRU', bias=True):
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
        self.hid_num = hid_num
        self.dropout = dropout
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
            #
        else:
            print("CPU")
            device = torch.device("cpu")
            self.set_thread()
        self.device = device
        self.gcn_type = gcn_type
        self.egcn_type = egcn_type
        self.unit_type = unit_type
        self.bias = bias
        self.loss = MainLoss(neg_num=neg_num, Q=Q)

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def set_thread(self, thread_num=None):
        if thread_num is None:
            thread_num = os.cpu_count() - 4
        torch.set_num_threads(thread_num)

    def reset_model(self):
        if self.gcn_type == 'RWTGCN':
            self.walk_len = len(
                os.listdir(os.path.join(self.walk_tensor_base_path, os.listdir(self.walk_tensor_base_path)[0])))
            print('walk_len: ', self.walk_len)
            self.model = RWTGCN(self.node_num, self.output_dim, self.walk_len, duration=self.duration,
                                unit_type=self.unit_type, bias=self.bias)
        elif self.gcn_type == 'MRGCN':
            assert self.duration == 1  # duration must be 1 when calling static embeding
            self.walk_len = len(
                os.listdir(os.path.join(self.walk_tensor_base_path, os.listdir(self.walk_tensor_base_path)[0])))
            print('walk_len: ', self.walk_len)
            self.model = MRGCN(self.node_num, self.output_dim, self.walk_len, bias=self.bias)
        elif self.gcn_type == 'GCN':
            assert self.duration == 1  # duration must be 1 when calling static embeding
            self.model = GCN(self.node_num, self.hid_num, self.output_dim, dropout=self.dropout, bias=self.bias)
        elif self.gcn_type == 'EvolveGCN':
            assert self.duration >= 1
            self.model = EvolveGCN(self.node_num, self.hid_num, self.output_dim, egcn_type=self.egcn_type)
        else:
            raise Exception('unsupported gcn type!')

    def get_date_adj_list(self, start_idx):
        if self.gcn_type in ['GCN', 'EvolveGCN']:
            date_dir_list = sorted(os.listdir(self.origin_base_path))
            time_stamp_num = len(self.timestamp_list)
            date_adj_list = []
            for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
                original_graph_path = os.path.join(self.origin_base_path, date_dir_list[i])
                date_adj_list = []
                spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep='\t')
                sptensor = get_normalize_adj_tensor(spmat)
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

            tmp_adj_list = []
            for f_name in f_list:
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                sptensor = get_normalize_adj_tensor(spmat)
                # original_graph_path = os.path.join(self.origin_base_path, date_dir_list[start_idx] + '.csv')
                # spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep='\t')
                # sptensor = get_normalize_adj_tensor(spmat)
                if self.gcn_type == 'RWTGCN':
                    tmp_adj_list.append(sptensor)
                    # tmp_adj_list.append(sptensor.cuda()  if torch.cuda.is_available() else sptensor)
                else: # MRGCN
                    date_adj_list.append(sptensor)
                    # date_adj_list.append(sptensor.cuda()  if torch.cuda.is_available() else sptensor)
            if self.gcn_type == 'RWTGCN':
                date_adj_list.append(tmp_adj_list)
        print('date adj list len: ', len(date_adj_list))
        return date_adj_list

    def get_node_pair_list(self, start_idx):
        walk_file_list = sorted(os.listdir(self.walk_pair_base_path))
        time_stamp_num = len(walk_file_list)
        assert start_idx < time_stamp_num

        node_pair_list = []
        for i in range(start_idx, min(start_idx + self.duration, time_stamp_num)):
            walk_file_path = os.path.join(self.walk_pair_base_path, walk_file_list[i])
            graph_dict = get_walk_neighbor_dict(walk_file_path, self.node_num)
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

    def learn_embedding(self, epoch=50, batch_size=10240, lr=1e-3, start_idx=0, weight_decay=0., model_file='rwtgcn', load_model=False, export=True):
        separate()
        time_stamp_num = len(self.timestamp_list)
        timestamps = ','.join([self.timestamp_list[i].split('.')[0] for i in range(start_idx, min(start_idx + self.duration, time_stamp_num))])
        print('train on ', timestamps)
        st = time.time()
        adj_list = self.get_date_adj_list(start_idx)
        print('adj list')
        node_pair_list = self.get_node_pair_list(start_idx)
        print('node pair list')
        neg_freq_list = self.get_neg_freq_list(start_idx)
        print('neg freq list')
        self.loss.set_node_info(node_pair_list, neg_freq_list)
        time_num = min(self.duration, time_stamp_num - start_idx)
        x_list = [sparse_mx_to_torch_sparse_tensor(sp.eye(self.node_num)) for i in range(time_num)]
        print('x list')
        self.reset_model()
        # print('yessssssssss')
        model = self.model
        if load_model:
            model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
            model.eval()

        if torch.cuda.is_available():
            model = model.to(self.device)
            # x_list = [x.cuda() for x in x_list]
            torch.cuda.empty_cache()

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()

        embedding_list = []
        batch_num = self.node_num // batch_size
        if self.node_num % batch_size != 0:
            batch_num += 1
        # print('start training!')
        for i in range(epoch):
            node_idx_list = np.random.permutation(np.arange(self.node_num))
            for j in range(batch_num):
                t1 = time.time()
                # print('forward!')
                embedding_list = model(x_list, adj_list)
                batch_node_idxs = node_idx_list[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                loss = self.loss(embedding_list, batch_node_idxs)
                loss.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                t2 = time.time()
                print("epoch", i + 1, ', batch num = ', j + 1, ", loss:", loss.item(), ', cost time: ', t2 - t1, ' seconds!')

        if export:
            if isinstance(embedding_list, list):
                for i in range(len(embedding_list)):
                    embedding = embedding_list[i]
                    timestamp = self.timestamp_list[start_idx + i].split('.')[0]
                    df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
                    embedding_path = os.path.join(self.embedding_base_path, timestamp + ".csv")
                    df_export.to_csv(embedding_path, sep='\t', header=True, index=True)
            else:
                timestamp = self.timestamp_list[start_idx].split('.')[0]
                df_export = pd.DataFrame(data=embedding_list.cpu().detach().numpy(), index=self.full_node_list)
                embedding_path = os.path.join(self.embedding_base_path, timestamp + ".csv")
                df_export.to_csv(embedding_path, sep='\t', header=True, index=True)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        del adj_list, x_list, node_pair_list, neg_freq_list, embedding_list, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        en = time.time()
        print('training total time: ',en - st, ' seconds!')
        return

def static_embedding():
    dataset = 'facebook'
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    print(base_path)
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/GCN')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    t1 = time.time()
    print('start GCN embedding!')
    GCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='gcn_walk_pairs',
                                   node_freq_folder='gcn_node_freq', walk_tensor_folder='', embedding_folder=embedding_folder,
                                   node_file=node_file, origin_folder=origin_folder, model_folder='model',
                                   output_dim=128, hid_num=500, dropout=0.5, duration=1, neg_num=20, Q=20, gcn_type = 'GCN', bias=True)
    timestamp_num = len(GCN.timestamp_list)
    for idx in range(timestamp_num):
        GCN.learn_embedding(epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', export=True)
    t2 = time.time()
    print('finish GCN embedding! cost time: ', t2 - t1, ' seconds!')

    # separate()
    # for neg_num in [10, 20, 50, 80, 100, 150, 200]:
    #     for Q in [500, 1000]:
    print('start MRGCN embedding!')
    t1 = time.time()
    embedding_folder = os.path.join('..', '2.embedding/MRGCN')
    MRGCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='mrgcn_walk_pairs',
                                    node_freq_folder='mrgcn_node_freq', walk_tensor_folder="mrgcn_walk_tensor", embedding_folder=embedding_folder,
                                    node_file=node_file, origin_folder=origin_folder, model_folder='model',
                                    output_dim=128, dropout=0.5, duration=1, neg_num=150, Q=10, gcn_type='MRGCN', bias=True)
    timestamp_num = len(MRGCN.timestamp_list)
    for idx in range(0, timestamp_num):
        MRGCN.learn_embedding(epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='mrgcn', export=True)
    t2 = time.time()
    print('finish MRGCN embedding! cost time: ', t2 - t1, ' seconds!')

def dynamic_embedding():
    dataset = 'enron'
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    print(base_path)
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/EvolveGCN')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    # t1 = time.time()
    # print('start EvolveGCN embedding!')
    # duration = 5
    # EvolveGCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='evolvegcn_walk_pairs',
    #                                node_freq_folder='evolvegcn_node_freq', walk_tensor_folder='', embedding_folder=embedding_folder,
    #                                node_file=node_file, origin_folder=origin_folder, model_folder='model',
    #                                output_dim=128, hid_num=128, dropout=0.5, duration=duration, neg_num=20, Q=20, gcn_type = 'EvolveGCN', bias=True)
    # timestamp_num = len(EvolveGCN.timestamp_list)
    # load_model = False
    # for idx in range(0, timestamp_num, duration):
    #     if idx > 0:
    #         load_model = True
    #     EvolveGCN.learn_embedding(epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, load_model=load_model, model_file='evolvegcn', export=True)
    # t2 = time.time()
    # print('finish EvolveGCN embedding! cost time: ', t2 - t1, ' seconds!')

    # link_predictor = LinkPredictor(base_path="../../data/" + dataset, edge_folder='1.format',
    #                                embedding_folder="2.embedding",
    #                                lp_edge_folder="link_prediction_data", output_folder="link_prediction_res",
    #                                node_file="nodes_set/nodes.csv",
    #                                train_ratio=1.0, test_ratio=1.0)
    # method_list = ['EvolveGCN']
    # link_predictor.link_prediction_all_method(method_list=method_list, worker=-1)

    #######################
    print('start RWTGCN embedding!')
    t1 = time.time()
    for prob in [1]:  #, 0.5, 0.8, 1
        embedding_folder = os.path.join('..', '2.embedding/RWTGCN_prob_' + str(prob))
        duration = 4
        RWTGCN = DynamicEmbedding(base_path=base_path, walk_pair_folder='rwtgcn_' + str(prob) + '_walk_pairs',
                                 node_freq_folder='rwtgcn_' + str(prob) + '_node_freq', walk_tensor_folder='rwtgcn_' + str(prob) + '_walk_tensor',
                                 embedding_folder=embedding_folder,
                                 node_file=node_file, origin_folder=origin_folder, model_folder='model',
                                 output_dim=128, dropout=0.5, duration=duration, neg_num=20, Q=10, gcn_type='RWTGCN', bias=True)
        timestamp_num = len(RWTGCN.timestamp_list)
        for idx in range(0, timestamp_num, duration - 1):
            RWTGCN.learn_embedding(epoch=50, batch_size=87036, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='rwtgcn_prob_' + str(prob), export=True)
        t2 = time.time()
        print('finish RWTGCN embedding! cost time: ', t2 - t1, ' seconds!')

        link_predictor = LinkPredictor(base_path="../../data/" + dataset, origin_folder='1.format',
                                       embedding_folder="2.embedding",
                                       lp_edge_folder="link_prediction_data", output_folder="link_prediction_res",
                                       node_file="nodes_set/nodes.csv",
                                       train_ratio=1.0, test_ratio=1.0)
        method_list = ['RWTGCN_prob_' + str(prob)]
        link_predictor.link_prediction_all_method(method_list=method_list, worker=-1)

if __name__ == '__main__':

   dynamic_embedding()


