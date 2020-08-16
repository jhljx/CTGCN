# coding: utf-8
import pandas as pd
import os
import time
import torch
from embedding import SupervisedEmbedding, UnsupervisedEmbedding
from helper import DataLoader
from metrics import NegativeSamplingLoss, ReconstructionLoss, VAELoss, VAEClassificationLoss
from metrics import ClassificationLoss, StructureClassificationLoss
from models import MLPClassifier, EdgeClassifier, InnerProduct
from utils import get_supported_gnn_methods, get_core_based_methods


def get_data_loader(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    core_folder = args.get('core_folder', None)
    nfeature_folder = args.get('nfeature_folder', None)
    node_file = args['node_file']
    has_cuda = args['has_cuda']

    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_list = nodes_set['node'].tolist()
    node_num = nodes_set.shape[0]

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder)) if origin_folder else None
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder)) if core_folder else None
    node_feature_path = os.path.abspath(os.path.join(base_path, nfeature_folder)) if nfeature_folder else None
    max_time_num = len(os.listdir(origin_base_path)) if origin_base_path else len(os.listdir(core_base_path))
    assert max_time_num > 0

    data_loader = DataLoader(node_list, max_time_num, has_cuda=has_cuda)
    args['origin_base_path'] = origin_base_path
    args['core_base_path'] = core_base_path
    args['nfeature_path'] = node_feature_path
    args['node_num'] = node_num
    return data_loader


def get_input_data(method, idx, data_loader, args):
    assert method in get_supported_gnn_methods()

    origin_base_path = args['origin_base_path']
    core_base_path = args['core_base_path']
    node_feature_path = args['nfeature_path']  # all the data sets we use don't have node features, so this path is None
    file_sep = args['file_sep']
    duration = args['duration']

    core_adj_list = []
    if method in get_core_based_methods():  # CGCN-C, CGCN-S, CTGCN-C, CTGCN-S
        max_core = args['max_core']
        core_adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=duration, max_core=max_core)
    if method in ['GCN', 'GAT']:
        normalize, row_norm, add_eye = True, True, True
    elif method in ['EvolveGCN']:  # normalization is quite important for the performance improvement of EvolveGCN
        normalize, row_norm, add_eye = True, False, True
    else:  # SAGE, GIN, TgGCN, TgGAT, TgSAGE, TgGIN, PGNN, GCRN, VGRNN, core_based_methods
        normalize, row_norm, add_eye = False, False, False

    adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx, duration=duration, sep=file_sep, normalize=normalize, row_norm=row_norm, add_eye=add_eye, data_type='tensor')
    # all gnn methods need edge_list when learning_type='S-link'
    edge_list = [adj._indices() for adj in adj_list]  # edge_indices: [2, edge_num]

    if method in get_core_based_methods():  # CGCN-C, CGCN-S, CTGCN-C, CTGCN-S
        adj_list = core_adj_list
    elif method in ['TgGCN', 'TgGAT', 'TgSAGE', 'TgGIN', 'PGNN', 'GCRN']:  # VGRNN uses GAE architecture, so adj_list is needed!
        adj_list = None

    if method in ['EvolveGCN', 'CGCN-S', 'CTGCN-S'] and node_feature_path is None:
        init_type = args['init_type']
        std = args.get('std', 1e-4)
        x_list, input_dim = data_loader.get_degree_feature_list(origin_base_path, start_idx=idx, duration=duration, sep=file_sep, init_type=init_type, std=std)
        # print('input_dim: ', input_dim)
    else:   # GCN, TgGCN, GAT, TgGAT, SAGE, TgSAGE, GIN, TgGIN, PGNN, GCRN, VGRNN, CGCN-C, CTGCN-C
        x_list, input_dim = data_loader.get_feature_list(node_feature_path, start_idx=idx, duration=duration, shuffle=False)
        if method == 'VGRNN':
            x_list = torch.stack(x_list)

    node_dist_list = None
    if method == 'PGNN':
        from baseline.pgnn import precompute_dist_data
        node_num = args['node_num']  # not hyper-parameter
        approximate = args['approximate']
        node_dist_list = precompute_dist_data(edge_list, node_num, approximate=approximate)
    # print('input_dim: ', input_dim, ', adj_list:', adj_list, ', x_list: ', x_list[0].shape, ', edge_list: ', edge_list[0].shape, ', node dist list: ', node_dist_list)
    return input_dim, adj_list, x_list, edge_list, node_dist_list


def get_gnn_model(method, args):
    assert method in get_supported_gnn_methods()

    from baseline.gcn import GCN, TgGCN
    from baseline.gat import GAT, TgGAT
    from baseline.sage import SAGE, TgSAGE
    from baseline.gin import GIN, TgGIN
    from baseline.pgnn import PGNN
    from baseline.gcrn import GCRN
    from baseline.egcn import EvolveGCN
    from baseline.vgrnn import VGRNN
    from models import CGCN, CTGCN

    input_dim = args['input_dim']
    hidden_dim = args['hid_dim']
    embed_dim = args['embed_dim']
    dropout = args.get('dropout', None)
    bias = args.get('bias', None)
    duration = args.get('duration', None)

    if method == 'GCN':
        return GCN(input_dim, hidden_dim, embed_dim, dropout=dropout, bias=bias)
    elif method == 'GAT':
        alpha = args['alpha']
        head_num = args['head_num']
        return GAT(input_dim, hidden_dim, embed_dim, dropout=dropout, alpha=alpha, head_num=head_num)
    elif method == "SAGE":
        num_sample = args['num_sample']
        pooling_type = args['pooling_type']
        return SAGE(input_dim, hidden_dim, embed_dim, num_sample=num_sample, pooling_type=pooling_type, gcn=False, dropout=dropout, bias=bias)
    elif method == 'GIN':
        layer_num = args['layer_num']
        mlp_layer_num = args['mlp_layer_num']
        learn_eps = args['learn_eps']
        neighbor_pooling_type = args['pooling_type']
        return GIN(input_dim, hidden_dim, embed_dim, layer_num, mlp_layer_num, learn_eps, neighbor_pooling_type, dropout=dropout, bias=bias)
    elif method in ['TgGCN', 'TgGAT', 'TgSAGE', 'TgGIN', 'PGNN', 'GCRN']:
        feature_pre = args['feature_pre']
        feature_dim = args['feature_dim']
        layer_num = args['layer_num']
        if method == 'TgGCN':
            return TgGCN(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias)
        elif method == 'TgGAT':
            return TgGAT(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias)
        elif method == 'TgSAGE':
            return TgSAGE(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias)
        elif method == 'TgGIN':
            return TgGIN(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias)
        elif method == 'PGNN':
            return PGNN(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias)
        elif method == 'GCRN':
            rnn_type = args['rnn_type']
            return GCRN(input_dim, feature_dim, hidden_dim, embed_dim, feature_pre=feature_pre, layer_num=layer_num, dropout=dropout, bias=bias,
                        duration=duration, rnn_type=rnn_type)
    elif method == 'VGRNN':
        rnn_layer_num = args['rnn_layer_num']
        conv_type = args['conv_type']
        return  VGRNN(input_dim, hidden_dim, embed_dim, rnn_layer_num=rnn_layer_num, conv_type=conv_type, bias=bias)
    elif method == 'EvolveGCN':
        egcn_type = args['model_type']
        return EvolveGCN(input_dim, hidden_dim, embed_dim, duration=duration, egcn_type=egcn_type)
    else:  # core-based gcn methods(both static and temporal core-based gcn)
        trans_num = args['trans_layer_num']
        diffusion_num = args['diffusion_layer_num']
        hidden_dim = args['hid_dim']
        model_type = args['model_type']
        rnn_type = args['rnn_type']
        trans_activate_type = args['trans_activate_type']
        if method in ['CGCN-C', 'CGCN-S']:
            return CGCN(input_dim, hidden_dim, embed_dim, trans_num=trans_num, diffusion_num=diffusion_num, bias=bias, rnn_type=rnn_type, model_type=model_type,
                        trans_activate_type=trans_activate_type)
        else:
            return CTGCN(input_dim, hidden_dim, embed_dim, trans_num=trans_num, diffusion_num=diffusion_num, duration=duration, bias=bias, rnn_type=rnn_type,
                         model_type=model_type, trans_activate_type=trans_activate_type)


def get_loss(method, idx, data_loader, args):
    learning_type = args['learning_type']
    assert learning_type in ['U-neg', 'U-own', 'S-node', 'S-edge', 'S-link']
    base_path = args['base_path']
    file_sep = args['file_sep']
    duration = args['duration']

    if learning_type == 'U-neg':
        walk_pair_folder = args['walk_pair_folder']
        node_freq_folder = args['node_freq_folder']
        neg_num = args['neg_num']
        Q = args['Q']
        walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=duration)
        neg_freq_list = data_loader.get_node_freq_list(node_freq_base_path, start_idx=idx, duration=duration)
        loss = NegativeSamplingLoss(node_pair_list, neg_freq_list, neg_num=neg_num, Q=Q)
        return loss
    elif learning_type == 'U-own':
        if method == 'VGRNN':
            eps = args['eps']
            loss = VAELoss(eps=eps)
        elif method in ['CGCN-S', 'CTGCN-S']:
            loss = ReconstructionLoss()
        else:
            raise NotImplementedError('No implementation of ' + method + '\'s unsupervised learning loss!')
        return loss
    else:   # supervised learning_type ['S-node', 'S-edge', 'S-link']:
        embed_dim = args['embed_dim']
        cls_hidden_dim = args.get('cls_hid_dim', None)
        cls_layer_num = args.get('cls_layer_num', None)
        cls_bias = args.get('cls_bias', None)
        cls_activate_type = args.get('cls_activate_type', None)

        node_label_list, edge_label_list = None, None
        if learning_type == 'S-node':
            nlabel_folder = args['nlabel_folder']
            nlabel_base_path = os.path.abspath(os.path.join(base_path, nlabel_folder))
            node_label_list, output_dim = data_loader.get_node_label_list(nlabel_base_path, start_idx=idx, duration=duration, sep=file_sep)
            classifier = MLPClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=duration, bias=cls_bias, activate_type=cls_activate_type)
        elif learning_type == 'S-edge':
            elabel_folder = args['elabel_folder']
            elabel_base_path = os.path.abspath(os.path.join(base_path, elabel_folder))
            edge_label_list, output_dim = data_loader.get_edge_label_list(elabel_base_path, start_idx=idx, duration=duration, sep=file_sep)
            classifier = EdgeClassifier(embed_dim, cls_hidden_dim, output_dim, layer_num=cls_layer_num, duration=duration, bias=cls_bias, activate_type=cls_activate_type)
        else:  # S-link
            classifier = InnerProduct()
            output_dim = 2  # postive link & negative link
        # loss
        if method == 'VGRNN':
            eps = args['eps']
            loss = VAEClassificationLoss(output_dim, eps=eps)
        elif method in ['CGCN-S', 'CTGCN-S']:
            loss = StructureClassificationLoss(output_dim)
        else:
            loss = ClassificationLoss(output_dim)
        return loss, classifier, node_label_list, edge_label_list


def gnn_embedding(method, args):
    # common params
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    model_folder = args['model_folder']
    model_file = args['model_file']
    node_file = args['node_file']
    # file_sep = args['file_sep']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    duration = args['duration']
    has_cuda = args['has_cuda']
    learning_type = args['learning_type']
    # hidden_dim = args['hid_dim']
    # embed_dim = args['embed_dim']
    epoch = args['epoch']
    lr = args['lr']
    batch_size = args['batch_size']
    load_model = args['load_model']
    shuffle = args['shuffle']
    export = args['export']
    record_time = args['record_time']
    weight_decay = args['weight_decay']

    data_loader = get_data_loader(args)
    max_time_num = data_loader.max_time_num
    node_list = data_loader.full_node_list

    if start_idx < 0:
        start_idx = max_time_num + start_idx
    if end_idx < 0:  # original time range is [start_idx, end_idx] containing start_idx and end_idx
        end_idx = max_time_num + end_idx + 1
    step = duration
    if learning_type == 'S-link':
        assert duration >= 2 and end_idx - start_idx >= 1
        end_idx = end_idx - 1
        step = duration - 1  # -1 is to make step and end_idx adapt to the dynamic link prediction setting

    t1 = time.time()
    time_list = []
    print('start_idx = ', start_idx, ', end_idx = ', end_idx, ', duration = ', duration)
    print('start ' + method + ' embedding!')
    for idx in range(start_idx, end_idx, step):
        print('idx = ', idx, ', duration = ', duration)
        input_dim, adj_list, x_list, edge_list, node_dist_list = get_input_data(method, idx, data_loader, args)
        args['input_dim'] = input_dim
        model = get_gnn_model(method, args)

        if learning_type in ['U-neg', 'U-own']:
            loss = get_loss(method, idx, data_loader, args)
            trainer = UnsupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=node_list,
                                            model=model, loss=loss, model_folder=model_folder, has_cuda=has_cuda)
            cost_time = trainer.learn_embedding(adj_list, x_list, edge_list, node_dist_list, epoch=epoch, batch_size=batch_size, lr=lr, start_idx=idx, weight_decay=weight_decay,
                                                model_file=model_file, load_model=load_model, shuffle=shuffle, export=export)
            time_list.append(cost_time)

        else:  # supervised learning
            cls_file = args.get('cls_file', None)
            train_ratio = args['train_ratio']
            val_ratio = args['val_ratio']
            test_ratio = args['test_ratio']
            loss, classifier, node_labels, edge_labels = get_loss(method, idx, data_loader, args)
            trainer = SupervisedEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=node_list, model=model,
                                          loss=loss, classifier=classifier, model_folder=model_folder, has_cuda=has_cuda)
            cost_time = trainer.learn_embedding(adj_list, x_list, node_labels, edge_labels, edge_list, node_dist_list, learning_type=learning_type, epoch=epoch, batch_size=batch_size,
                                                lr=lr, start_idx=idx, weight_decay=weight_decay, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                                                model_file=model_file, classifier_file=cls_file, load_model=load_model, shuffle=shuffle, export=export)
            time_list.append(cost_time)

    # record time cost of the model
    if record_time:
        df_output = pd.DataFrame({'time': time_list})
        df_output.to_csv(os.path.join(base_path, method + '_time.csv'), sep=',', index=False)
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')
