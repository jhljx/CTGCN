import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing
import os, time, json, sys
import gc
sys.path.append("..")
import torch
import torch.nn as nn
from CTGCN.metrics import SupervisedLoss, UnsupervisedLoss
from CTGCN.baseline.egcn import EvolveGCN
from CTGCN.baseline.gcn import GCN
from CTGCN.baseline.gat import SpGAT
from CTGCN.baseline.timers import timers
from CTGCN.embedding import  DataLoader, SupervisedEmbedding, UnsupervisedEmbedding
from CTGCN.models import CTGCN, CGCN, MLPClassifier
from CTGCN.utils import check_and_make_path, sparse_mx_to_torch_sparse_tensor, get_sp_adj_mat, separate
from CTGCN.evaluation.link_prediction import LinkPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2,3'

def gcn_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
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
        # time_list = []
        for idx in range(max_time_num):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            gcn_model = GCN(input_dim=node_num, hidden_dim=500, output_dim=128, dropout=0.5, bias=True)
            gcn_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            gcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                        model=gcn_model, loss=gcn_loss, max_time_num=max_time_num)
            # ta = time.time()
            gcn.learn_embedding(adj_list, x_list, epoch=5, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', export=True)
            if idx == 1:
                break
        #     tb = time.time()
        #     time_list.append(tb - ta)
        # df_output = pd.DataFrame({'time': time_list})
        # df_output.to_csv(dataset + '_gcn_time.csv', sep=',', index=False)
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(max_time_num):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            gcn_model = GCN(input_dim=node_num, hidden_dim=500, output_dim=128, dropout=0.5, bias=True)
            gcn_loss = SupervisedLoss()
            gcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=duration, bias=True, trans_version='L')
            gcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                      model=gcn_model, loss=gcn_loss, classifier=gcn_classifier, max_time_num=max_time_num)
            gcn.learn_embedding(adj_list, x_list, label_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gcn', classifier_file='gcn_cls', export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish GCN embedding! cost time: ', t2 - t1, ' seconds!')
    return

def gat_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
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
        # time_list = []
        for idx in range(max_time_num):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration, data_type='matrix')
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            gat_model = SpGAT(node_num, 500, 128, nheads=1)
            gat_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            gat = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                        model=gat_model, loss=gat_loss, max_time_num=max_time_num)
            # ta = time.time()
            gat.learn_embedding(adj_list, x_list, epoch=5, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gat', export=True)
            if idx == 1:
                break
        #     tb = time.time()
        #     time_list.append(tb - ta)
        # df_output = pd.DataFrame({'time': time_list})
        # df_output.to_csv(dataset + '_gat_time.csv', sep=',', index=False)
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(max_time_num):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration, data_type='matrix')
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            gat_model = SpGAT(node_num, 500, 128, nheads=1)
            gat_loss = SupervisedLoss()
            gat_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=duration, bias=True, trans_version='L')
            gat = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(),
                                      model=gat_model, loss=gat_loss, classifier=gat_classifier, max_time_num=max_time_num)
            gat.learn_embedding(adj_list, x_list, label_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='gat', classifier_file='gat_cls', export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish GAT embedding! cost time: ', t2 - t1, ' seconds!')
    return

def evolvegcn_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
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
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            evolvegcn_model = EvolveGCN(input_dim=max_degree, hidden_dim=128, output_dim=128, duration=duration, egcn_type='EGCNH')
            evolvegcn_loss = UnsupervisedLoss(neg_num=20, Q=20, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            evolvegcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                        node_list=nodes_set['node'].tolist(), model=evolvegcn_model, loss=evolvegcn_loss, max_time_num=max_time_num)
            evolvegcn.learn_embedding(adj_list, x_list, epoch=5, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='evolvegcnh', export=True)
            break
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            evolvegcn_model = EvolveGCN(input_dim=max_degree, hidden_dim=128, output_dim=128, duration=duration, egcn_type='EGCNH')
            evolvegcn_loss = SupervisedLoss()
            evolvegcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=duration, bias=True, trans_version='L')
            evolvegcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                      node_list=nodes_set['node'].tolist(), model=evolvegcn_model, loss=evolvegcn_loss, classifier=evolvegcn_classifier, max_time_num=max_time_num)
            evolvegcn.learn_embedding(adj_list, x_list, label_list, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx, weight_decay=5e-4, model_file='evolvegcnh', classifier_file='evolvegcnh_cls', export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish EvolveGCN embedding! cost time: ', t2 - t1, ' seconds!')
    return

def timers_embedding(dataset):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path =  os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/timers')
    embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
    node_file = os.path.join('..', 'nodes_set/nodes.csv')
    node_file_path = os.path.abspath(os.path.join(base_path, node_file))
    theta = 0.17
    dim = 128

    timers(nodes_file=node_file_path, input_path=origin_base_path, output_path=embedding_base_path, Theta=theta, dim=dim // 2, Update=True)

def cgcn_connective_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
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
        # time_list = []
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=duration)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=duration)

            cgcn_model = CGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, bias=True, rnn_type='GRU', version='C', trans_version='L')
            cgcn_loss = UnsupervisedLoss(neg_num=150, Q=10, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            cgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                           node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                           loss=cgcn_loss, max_time_num=max_time_num)
            # ta = time.time()
            cgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=5, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_c', embedding_type='connection', export=True)
            if idx == 1:
                break
        #     tb = time.time()
        #     time_list.append(tb - ta)
        # df_output = pd.DataFrame({'time': time_list})
        # df_output.to_csv(dataset + '_cgcn_c_time.csv', sep=',', index=False)
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=duration)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=duration)

            cgcn_model = CGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, bias=True, rnn_type='GRU', version='C', trans_version='L')
            cgcn_loss = SupervisedLoss()
            cgcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=duration, bias=True, trans_version='L')
            cgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder,
                                         embedding_folder=embedding_folder,
                                         node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                         loss=cgcn_loss, classifier=cgcn_classifier, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, label_list, single_output=False, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_c', classifier_file='cgcn_c_cls', embedding_type='connection', export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish CGCN_C embedding! cost time: ', t2 - t1, ' seconds!')
    return


def cgcn_structural_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path =  os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/CGCN_S')
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
    print('start CGCN_S embedding on ' + dataset)
    if learning_type == 'unsupervise':
        # time_list = []
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=duration)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            cgcn_model = CGCN(input_dim=max_degree, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, bias=True, rnn_type='GRU', version='S', trans_version='N')
            cgcn_loss = UnsupervisedLoss()
            cgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                              loss=cgcn_loss, max_time_num=max_time_num)
            # ta = time.time()
            cgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=2, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='cgcn_s', embedding_type='structure', export=True)
            if idx == 1:
                break
        #     tb = time.time()
        #     time_list.append(tb - ta)
        # df_output = pd.DataFrame({'time': time_list})
        # df_output.to_csv(dataset +'_cgcn_s_time.csv', sep=',', index=False)
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=duration)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration)

            cgcn_model = CGCN(input_dim=max_degree, hidden_dim=500, output_dim=128, trans_num=3,  diffusion_num=1, bias=True, rnn_type='GRU', version='S', trans_version='N')
            cgcn_loss = SupervisedLoss()
            cgcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=duration, bias=True, trans_version='L')
            cgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=cgcn_model,
                                            loss=cgcn_loss, classifier=cgcn_classifier, max_time_num=max_time_num)
            cgcn.learn_embedding(adj_list, x_list, label_list, single_output=False, epoch=20, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='cgcn_s', classifier_file='cgcn_s_cls', embedding_type='structure', export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish CGCN_S embedding! cost time: ', t2 - t1, ' seconds!')
    return


def ctgcn_connective_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    embedding_folder = os.path.join('..', '2.embedding/CTGCN_C')
    core_folder = 'ctgcn_cores'
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
    print('start CTGCN_C embedding on ' + dataset)
    if learning_type == 'unsupervise':
        walk_pair_folder = 'ctgcn_walk_pairs'
        node_freq_folder = 'ctgcn_node_freq'
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))

        embedding_folder = os.path.join('..', '2.embedding/CTGCN_C')
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)
            node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_num)
            neg_freq_list = data_loader.get_neg_freq_list(node_freq_base_path, start_idx=idx, duration=time_num)

            ctgcn_model = CTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, duration=time_num, bias=True, rnn_type='GRU', version='C', trans_version='L')
            ctgcn_loss = UnsupervisedLoss(neg_num=150, Q=10, node_pair_list=node_pair_list, neg_freq_list=neg_freq_list)
            ctgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=ctgcn_model,
                                              loss=ctgcn_loss, max_time_num=max_time_num)
            ctgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=5, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='ctgcn_c', embedding_type='connection', export=True)
            break
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list = data_loader.get_feature_list(None, start_idx=idx, duration=time_num)

            ctgcn_model = CTGCN(input_dim=node_num, hidden_dim=500, output_dim=128, trans_num=1, diffusion_num=2, duration=time_num, bias=True, rnn_type='GRU', version='C', trans_version='L')
            ctgcn_loss = SupervisedLoss()
            ctgcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=time_num, bias=True, trans_version='L')
            ctgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=ctgcn_model,
                                            loss=ctgcn_loss, classifier=ctgcn_classifier, max_time_num=max_time_num)
            ctgcn.learn_embedding(adj_list, x_list, label_list, single_output=False, epoch=50, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='ctgcn_c', classifier_file='ctgcn_c_cls', embedding_type='connection',  export=True)
    else:
        raise AttributeError('Unsupported learning type!')
    t2 = time.time()
    print('finish CTGCN_C embedding! cost time: ', t2 - t1, ' seconds!')
    return

def ctgcn_structural_embedding(dataset, learning_type='unsupervise'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '../data/' + dataset + '/CTGCN'))
    origin_folder = os.path.join('..', '1.format')
    origin_base_path =  os.path.abspath(os.path.join(base_path, origin_folder))
    embedding_folder = os.path.join('..', '2.embedding/CTGCN_S')
    core_folder = 'ctgcn_cores'
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
    print('start CTGCN_S embedding on ' + dataset)
    if learning_type == 'unsupervise':
        embedding_folder = os.path.join('..', '2.embedding/CTGCN_S')
        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=time_num)

            ctgcn_model = CTGCN(input_dim=max_degree, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, duration=time_num, bias=True, rnn_type='GRU', version='S', trans_version='N')
            ctgcn_loss = UnsupervisedLoss()
            ctgcn = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                              node_list=nodes_set['node'].tolist(), model=ctgcn_model,
                                              loss=ctgcn_loss, max_time_num=max_time_num)
            ctgcn.learn_embedding(adj_list, x_list, single_output=False, epoch=2, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                   weight_decay=5e-4, model_file='ctgcn_s', embedding_type='structure', export=True)
            break
    elif learning_type == 'supervise':
        label_file = os.path.join('..', 'nodes_set/trans_label.csv')
        label_path = os.path.abspath(os.path.join(base_path, label_file))
        df_label = pd.read_csv(label_path, sep='\t')
        label_list = df_label['label'].values

        for idx in range(0, max_time_num, duration):
            print('idx = ', idx)
            time_num = min(duration, max_time_num - idx)
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_num)
            x_list, max_degree, _ = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=time_num)

            ctgcn_model = CTGCN(input_dim=max_degree, hidden_dim=500, output_dim=128, trans_num=3, diffusion_num=1, duration=time_num, bias=True, rnn_type='GRU', version='S', trans_version='N')
            ctgcn_loss = SupervisedLoss()
            ctgcn_classifier = MLPClassifier(128, 64, label_list.max() + 1, layer_num=1, duration=time_num, bias=True, trans_version='L')
            ctgcn = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder,
                                            node_list=nodes_set['node'].tolist(), model=ctgcn_model,
                                            loss=ctgcn_loss, classifier=ctgcn_classifier, max_time_num=max_time_num)
            ctgcn.learn_embedding(adj_list, x_list, label_list, single_output=False, epoch=20, batch_size=4096 * 8, lr=0.001, start_idx=idx,
                                      weight_decay=5e-4, model_file='ctgcn_s', classifier_file='ctgcn_s_cls', embedding_type='structure', export=True)
    else:
        raise AttributeError('Unsupported learning type!')

    t2 = time.time()
    print('finish CTGCN_S embedding! cost time: ', t2 - t1, ' seconds!')
    return

if __name__ == '__main__':
    dataset = 'facebook'
    #gcn_embedding(dataset=dataset)
    # gat_embedding(dataset=dataset)
    evolvegcn_embedding(dataset=dataset)
    cgcn_connective_embedding(dataset=dataset)
    ctgcn_connective_embedding(dataset=dataset)
    cgcn_structural_embedding(dataset=dataset)
    ctgcn_structural_embedding(dataset=dataset)