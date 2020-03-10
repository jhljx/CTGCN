import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, time, json, sys
import gc
sys.path.append("..")
import torch
import torch.nn as nn
from RWTGCN.metrics import SupervisedLoss, UnsupervisedLoss
from RWTGCN.baseline.egcn import EvolveGCN
from RWTGCN.baseline.gcn import GCN
from RWTGCN.baseline.gat import SpGAT
from RWTGCN.embedding import  DataLoader, SupervisedEmbedding, UnsupervisedEmbedding
from RWTGCN.models import RWTGCN, CGCN
from RWTGCN.utils import check_and_make_path, sparse_mx_to_torch_sparse_tensor, get_sp_adj_mat, separate
from RWTGCN.evaluation.link_prediction import LinkPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2,3'

def gcn_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/GCN')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    duration = 1

    max_time_num = len(os.listdir(origin_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)

    t1 = time.time()
    print('start GCN embedding!')
    if learning_type == 'unsupervise':
        walk_pair_folder = 'gcn_walk_pairs'
        node_freq_folder = 'gcn_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        for idx in range(max_time_num):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            gcn_model = GCN(input_dim=node_num, hidden_dim=500, output_dim=128, dropout=0.5, bias=True)
            gcn_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            gcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                        model=gcn_model, loss=gcn_loss, max_time_num=max_time_num)
            gcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(max_time_num):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=1)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            gcn_model = GCN(input_dim=node_num, hidden_dim=500, output_dim=128, dropout=0.5, bias=True)
            gcn_loss = SupervisedLoss(label_list)
            gcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                      model=gcn_model, loss=gcn_loss, max_time_num=max_time_num)
            gcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', export=True)

    t2 = time.time()
    print('finish GCN embedding! cost time: ', t2 - t1, ' seconds!')
    return

def gat_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/GAT')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    duration = 1

    max_time_num = len(os.listdir(origin_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)

    t1 = time.time()
    print('start GAT embedding!')
    if learning_type == 'unsupervise':
        walk_pair_folder = 'gat_walk_pairs'
        node_freq_folder = 'gat_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        for idx in range(max_time_num):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration, data_type='matrix')
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            gat_model = SpGAT(node_num, 500, 128, nheads=1)
            gat_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            gat = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                        model=gat_model, loss=gat_loss, max_time_num=max_time_num)
            gat.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gat', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(max_time_num):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=1, data_type='matrix')
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            gat_model = SpGAT(node_num, 500, 128, nheads=1)
            gat_loss = SupervisedLoss(label_list)
            gat = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                      model=gat_model, loss=gat_loss, max_time_num=max_time_num)
            gat.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gat', export=True)

    t2 = time.time()
    print('finish GAT embedding! cost time: ', t2 - t1, ' seconds!')
    return

def evolvegcn_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/EvolveGCNH')
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    duration = 15

    max_time_num = len(os.listdir(origin_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)

    t1 = time.time()
    print('start EvolveGCN embedding!')
    if learning_type == 'unsupervise':
        walk_pair_folder='evolvegcn_walk_pairs'
        node_freq_folder='evolvegcn_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        for idx in range(0, max_time_num, duration - 1):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            evolvegcn_model = EvolveGCN(input_dim=max_degree, hidden_dim=128, output_dim=128, duration=duration, egcn_type='EGCNH')
            evolvegcn_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            evolvegcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                        node_list=nodes_set['node'].tolist(), model=evolvegcn_model, loss=evolvegcn_loss, max_time_num=max_time_num)
            evolvegcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='evolvegcnh', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(0, max_time_num, duration):
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=1)
            x_list, max_degree = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            evolvegcn_model = EvolveGCN(input_dim=max_degree, hidden_dim=128, output_dim=128, duration=duration, egcn_type='EGCNH')
            evolvegcn_loss = SupervisedLoss(label_list)
            evolvegcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                      node_list=nodes_set['node'].tolist(), model=evolvegcn_model, loss=evolvegcn_loss, max_time_num=max_time_num)
            evolvegcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='evolvegcnh', export=True)

    t2 = time.time()
    print('finish EvolveGCN embedding! cost time: ', t2 - t1, ' seconds!')
    return

def cgcn_connective_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/CGCN_C')
    core_folder = 'cgcn_cores'
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder))
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    duration = 1

    max_time_num = len(os.listdir(core_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)
    print('max time num: ', max_time_num)

    t1 = time.time()
    print('start CGCN_C embedding on ' + dataset)
    if learning_type == 'unsupervise':
        walk_pair_folder = 'cgcn_walk_pairs'
        node_freq_folder = 'cgcn_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            # print('time num: ', time_num)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_num)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=time_num)

            cgcn_model = CGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, bias=True, rnn_type='GRU')
            # cgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, layer_num=1, duration=time_num, bias=True, rnn_type='GRU')
            cgcn_loss = UnsupervisedLoss(neg_num=150, Q=10, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            cgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                           node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                           loss=cgcn_loss, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_c', embedding_type='connection', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)

            #cgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, layer_num=1, duration=time_num, bias=True, rnn_type='GRU')
            cgcn_model = CGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, bias=True, rnn_type='GRU')
            cgcn_loss = SupervisedLoss(label_list)
            cgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder,
                                         embedding_folder=embedding_folder,
                                         node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                         loss=cgcn_loss, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_c', embedding_type='connection', export=True)
    t2 = time.time()
    print('finish CGCN_C embedding! cost time: ', t2 - t1, ' seconds!')
    return


def cgcn_structural_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path =  os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/CGCN_S')
    core_folder = 'rwtgcn_cores'
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder))
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    duration = 1

    max_time_num = len(os.listdir(core_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)
    print('max time num: ', max_time_num)

    t1 = time.time()
    print('start CGCN_S embedding on ' + dataset)
    if learning_type == 'unsupervise':
        for idx in range(0, max_time_num, duration):
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            cgcn_model = CGCN(input_dim=max_degree, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, bias=True, rnn_type='GRU')
            cgcn_loss = UnsupervisedLoss()
            cgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                              loss=cgcn_loss, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=10, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_s', embedding_type='structure', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)

            cgcn_model = CGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=3,  diffusion_num=1, bias=True, rnn_type='GRU')
            cgcn_loss = SupervisedLoss(label_list)
            cgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                            loss=cgcn_loss, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='cgcn_s', embedding_type='structure', export=True)

    t2 = time.time()
    print('finish CGCN_S embedding! cost time: ', t2 - t1, ' seconds!')
    return


def rwtgcn_connective_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/RWTGCN_C')
    core_folder = 'rwtgcn_cores'
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder))
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    duration = 5

    max_time_num = len(os.listdir(core_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)
    print('max time num: ', max_time_num)

    t1 = time.time()
    print('start RWTGCN_C embedding on ' + dataset)
    if learning_type == 'unsupervise':
        walk_pair_folder = 'rwtgcn_walk_pairs'
        node_freq_folder = 'rwtgcn_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        for idx in range(0, max_time_num, duration - 1):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_num)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=time_num)

            rwtgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, duration=time_num, bias=True, rnn_type='GRU')
            rwtgcn_loss = UnsupervisedLoss(neg_num=150, Q=10, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            rwtgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=rwtgcn_model,
                                              loss=rwtgcn_loss, max_time_num=max_time_num)
            rwtgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='rwtgcn_c', embedding_type='connection', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=1)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            rwtgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, duration=duration, bias=True, rnn_type='GRU')
            rwtgcn_loss = SupervisedLoss(label_list)
            rwtgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=rwtgcn_model,
                                            loss=rwtgcn_loss, max_time_num=max_time_num)
            rwtgcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='rwtgcn_c', embedding_type='connection',  export=True)

    t2 = time.time()
    print('finish RWTGCN_C embedding! cost time: ', t2 - t1, ' seconds!')
    return

def rwtgcn_structural_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'data/' + dataset + '/RWT-GCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path =  os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/RWTGCN_S')
    core_folder = 'rwtgcn_cores'
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder))
    node_file = os.path.join('..', 'nodes_set/nodes.csv')

    duration = 10

    max_time_num = len(os.listdir(core_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]

    data_loader = DataLoader(nodes_set['node'].tolist(), max_time_num)
    print('max time num: ', max_time_num)

    t1 = time.time()
    print('start RWTGCN_S embedding on ' + dataset)
    if learning_type == 'unsupervise':
        for idx in range(0, max_time_num, duration):
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            rwtgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, duration=time_num, bias=True, rnn_type='GRU')
            rwtgcn_loss = UnsupervisedLoss()
            rwtgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=rwtgcn_model,
                                              loss=rwtgcn_loss, max_time_num=max_time_num)
            rwtgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=20, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='rwtgcn_s', embedding_type='structure', export=True)
    else:
        label_file = os.path.join('..', 'nodes_set/labels.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, names=['label'])
        label_list = df_label['label'].tolist()

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=1)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            rwtgcn_model = RWTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, duration=duration, bias=True, rnn_type='GRU')
            rwtgcn_loss = SupervisedLoss(label_list)
            rwtgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=rwtgcn_model,
                                            loss=rwtgcn_loss, max_time_num=max_time_num)
            rwtgcn.learn_embedding(adj_list, x_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='rwtgcn_s', embedding_type='structure', export=True)

    t2 = time.time()
    print('finish RWTGCN_S embedding! cost time: ', t2 - t1, ' seconds!')
    return

if __name__ == '__main__':
    dataset = 'blogcatalog'
    # gat_embedding(dataset=dataset)
    cgcn_connective_embedding(dataset=dataset)
    # rwtgcn_connective_embedding(dataset=dataset)
    #cgcn_structural_embedding(dataset=dataset)
   # rwtgcn_structural_embedding(dataset=dataset)