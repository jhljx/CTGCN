# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from embedding import BaseEmbedding
from helper import DataLoader

# dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning. For more information, please refer to https://arxiv.org/abs/1809.02657
# We refer to the dyngraph2vec tensorflow source code https://github.com/palash1992/DynamicGEM, and implement a pytorch version of dyngraph2vec
# Author: jhljx
# Email: jhljx8918@gmail.com


# DynAE model and its components
# Multi-linear perceptron class
class MLP(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, n_units[0], bias=bias))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.Linear(n_units[i - 1], n_units[i], bias=bias))
        self.layer_list.append(nn.Linear(n_units[-1], output_dim, bias=bias))
        self.layer_num = layer_num + 1

    def forward(self, x):
        for i in range(self.layer_num):
            x = F.relu(self.layer_list[i](x))
        return x


# DynAE class
class DynAE(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, output_dim, look_back=3, n_units=None, bias=True, **kwargs):
        super(DynAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'DynAE'

        self.encoder = MLP(input_dim * look_back, output_dim, n_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred


# L1 and L2 regularization loss
class RegularizationLoss(nn.Module):
    nu1: float
    nu2: float

    def __init__(self, nu1, nu2):
        super(RegularizationLoss, self).__init__()
        self.nu1 = nu1
        self.nu2 = nu2

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True).cuda() if torch.cuda.is_available() else Variable(torch.FloatTensor([0.]), requires_grad=True)
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0. and self.nu2 == 0.:
            return loss
        # calculate L1-regularization loss and L2-regularization loss
        weight_list = self.get_weight(model)
        weight_num = len(weight_list)
        # print('weight num', weight_num)
        l1_reg_loss, l2_reg_loss = 0, 0
        for name, weight in weight_list:
            if self.nu1 > 0:
                l1_reg = torch.norm(weight, p=1)
                l1_reg_loss = l1_reg_loss + l1_reg
            if self.nu2 > 0:
                l2_reg = torch.norm(weight, p=2)
                l2_reg_loss = l2_reg_loss + l2_reg
        l1_loss = self.nu1 * l1_reg_loss / weight_num
        l2_loss = self.nu2 * l2_reg_loss / weight_num
        return l1_loss + l2_loss


# Loss used for DynAE, DynRNN, DynAERNN
class DynGraph2VecLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, beta, nu1, nu2):
        super(DynGraph2VecLoss, self).__init__()
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        x_reconstruct, x_real, y_penalty = input_list[0], input_list[1], input_list[2]
        assert len(input_list) == 3
        reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss


# Batch generator used for DynAE, DynRNN and DynAERNN
class BatchGenerator:
    node_list: list
    node_num: int
    batch_size: int
    look_back: int
    beta: float
    shuffle: bool
    has_cuda: bool

    def __init__(self, node_list, batch_size, look_back, beta, shuffle=True, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.look_back = look_back
        self.beta = beta
        self.shuffle = shuffle
        self.has_cuda = has_cuda

    def generate(self, graph_list):
        graph_num = len(graph_list)
        train_size = graph_num - self.look_back
        assert train_size > 0
        all_node_num = self.node_num * train_size
        batch_num = all_node_num // self.batch_size
        if all_node_num % self.batch_size != 0:
            batch_num += 1
        node_indices = np.arange(all_node_num)

        if self.shuffle:
            np.random.shuffle(node_indices)
        counter = 0
        while True:
            batch_indices = node_indices[self.batch_size * counter: min(all_node_num, self.batch_size * (counter + 1))]
            x_pre_batch = torch.zeros((self.batch_size, self.look_back, self.node_num))
            x_pre_batch = x_pre_batch.cuda() if self.has_cuda else x_pre_batch
            x_cur_batch = torch.zeros((self.batch_size, self.node_num), device=x_pre_batch.device)
            y_batch = torch.ones(x_cur_batch.shape, device=x_pre_batch.device)  # penalty tensor for x_cur_batch

            for idx, record_id in enumerate(batch_indices):
                graph_idx = record_id // self.node_num
                node_idx = record_id % self.node_num
                for step in range(self.look_back):
                    # graph is a scipy.sparse.lil_matrix
                    pre_tensor = torch.tensor(graph_list[graph_idx + step][node_idx, :].toarray(), device=x_pre_batch.device)
                    x_pre_batch[idx, step, :] = pre_tensor
                # graph is a scipy.sparse.lil_matrix
                cur_tensor = torch.tensor(graph_list[graph_idx + self.look_back][node_idx, :].toarray(), device=x_pre_batch.device)
                x_cur_batch[idx] = cur_tensor

            y_batch[x_cur_batch != 0] = self.beta
            counter += 1
            yield x_pre_batch, x_cur_batch, y_batch

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(node_indices)
                counter = 0


# Batch Predictor used for DynAE, DynRNN and DynAERNN
class BatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(self, graph_list, model, batch_indices, counter, look_back, embedding_mat, x_pred):
        batch_size = len(batch_indices)
        x_pre_batches = torch.zeros((batch_size, look_back, self.node_num))
        x_pre_batches = x_pre_batches.cuda() if self.has_cuda else x_pre_batches
        
        for idx, node_idx in enumerate(batch_indices):
            for step in range(look_back):
                # graph is a scipy.sparse.lil_matrix
                pre_tensor = torch.tensor(graph_list[step][node_idx, :].toarray(), device=x_pre_batches.device)
                x_pre_batches[idx, step, :] = pre_tensor
        # DynAE uses 2D tensor as its input
        if model.method_name == 'DynAE':
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        embedding_mat_batch, x_pred_batch = model(x_pre_batches)
        if counter:
            embedding_mat = torch.cat((embedding_mat, embedding_mat_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = embedding_mat_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph_list):
        look_back = len(graph_list)
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(self.batch_size * counter, self.batch_size * (counter + 1))
            embedding_mat, x_pred = self.get_predict_res(graph_list, model, batch_indices, counter, look_back, embedding_mat, x_pred)
            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(graph_list, model, remain_indices, counter, look_back, embedding_mat, x_pred)
        return embedding_mat, x_pred


# Dynamic Embedding for DynGEM, DynAE, DynRNN, DynAERNN
class DynamicEmbedding(BaseEmbedding):

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, batch_generator, batch_predictor, model_folder="model", has_cuda=False):
        super(DynamicEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)
        self.batch_generator = batch_generator
        self.batch_predictor = batch_predictor

        assert batch_generator.batch_size == batch_predictor.batch_size
        assert batch_generator.node_num == batch_predictor.node_num

    def get_batch_info(self, adj_list, model):
        graph_num = len(adj_list)
        batch_size = self.batch_generator.batch_size
        if model.method_name == 'DynGEM':
            rows, cols, values = sp.find(adj_list[0])
            train_size = 0
            element_num = rows.shape[0]
        else:
            train_size = graph_num - self.batch_generator.look_back
            element_num = self.node_num * train_size
        batch_num = element_num // batch_size
        if element_num % batch_size != 0:
            batch_num += 1
        return batch_size, batch_num, train_size

    def get_model_res(self, model, generator):
        batch_size = self.batch_generator.batch_size
        if model.method_name == 'DynGEM':
            [xi_batch, xj_batch], [yi_batch, yj_batch, value_batch] = next(generator)
            hx_i, xi_pred = model(xi_batch)
            hx_j, xj_pred = model(xj_batch)
            loss_input_list = [xi_pred, xi_batch, yi_batch, xj_pred, xj_batch, yj_batch, hx_i, hx_j, value_batch]
        else:
            x_pre_batches, x_cur_batch, y_batch = next(generator)
            # DynAE uses 2D tensor as its input
            if model.method_name == 'DynAE':
                x_pre_batches = x_pre_batches.reshape(batch_size, -1)
            _, x_pred_batch = model(x_pre_batches)
            loss_input_list = [x_pred_batch, x_cur_batch, y_batch]
        return loss_input_list

    def learn_embedding(self, adj_list, epoch=50, lr=1e-3, idx=0, weight_decay=0., model_file='dynAE', load_model=False, export=True):
        print('start learning embedding!')
        model, loss_model, optimizer, _ = self.prepare(load_model, model_file, classifier_file=None, lr=lr, weight_decay=weight_decay)
        batch_size, batch_num, train_size = self.get_batch_info(adj_list, model)

        print('start training!')
        st = time.time()
        for i in range(epoch):
            for j in range(batch_num):
                t1 = time.time()
                generator = self.batch_generator.generate(adj_list)
                loss_input_list = self.get_model_res(model, generator)
                loss = loss_model(model, loss_input_list)
                loss.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                t2 = time.time()
                print("epoch", i + 1, ', batch num = ', j + 1, ", loss:", loss.item(), ', cost time: ', t2 - t1, ' seconds!')
        print('finish training!')
        print('start predicting!')
        # This needs the last look_back number of graphs to make prediction
        embedding_mat, next_adj = self.batch_predictor.predict(model, adj_list[train_size:])
        print('end predicting!')
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(embedding_mat, idx)
        if model_file:
            torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        del adj_list, embedding_mat, model
        self.clear_cache()
        print('learning embedding total time: ', cost_time, ' seconds!')
        return cost_time


def dyngem_embedding(method, args):
    assert method in ['DynGEM', 'DynAE', 'DynRNN', 'DynAERNN']
    from baseline.dynRNN import DynRNN
    from baseline.dynAERNN import DynAERNN
    from baseline.dynGEM import DynGEM, DynGEMLoss, DynGEMBatchGenerator, DynGEMBatchPredictor
    model_dict = {'DynGEM': DynGEM, 'DynAE': DynAE, 'DynRNN': DynRNN, 'DynAERNN': DynAERNN}

    # DynGEM, DynAE, DynRNN, DynAERNN common params
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    embedding_folder = args['embed_folder']
    model_folder = args['model_folder']
    model_file = args['model_file']
    node_file = args['node_file']
    file_sep = args['file_sep']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    duration = args['duration']
    embed_dim = args['embed_dim']
    has_cuda = args['has_cuda']
    epoch = args['epoch']
    lr = args['lr']
    batch_size = args['batch_size']
    load_model = args['load_model']
    shuffle = args['shuffle']
    export = args['export']
    record_time = args['record_time']

    # DynGEM, DynAE, DynRNN, DynAERNN model params
    n_units, ae_units, rnn_units = [], [], []
    look_back, alpha = 0, 0
    if method in ['DynGEM', 'DynAE', 'DynRNN']:
        n_units = args['n_units']
    else:  # DynAERNN
        ae_units = args['ae_units']
        rnn_units = args['rnn_units']
    if method in ['DynAE', 'DynRNN', 'DynAERNN']:
        look_back = args['look_back']
        assert look_back > 0
    else:  # DynGEM
        alpha = args['alpha']
    beta = args['beta']
    nu1 = args['nu1']
    nu2 = args['nu2']
    bias = args['bias']

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
    max_time_num = len(os.listdir(origin_base_path))
    node_path = os.path.abspath(os.path.join(base_path, node_file))
    nodes_set = pd.read_csv(node_path, names=['node'])
    node_num = nodes_set.shape[0]
    node_list = nodes_set['node'].tolist()
    data_loader = DataLoader(node_list, max_time_num, has_cuda=has_cuda)

    if start_idx < 0:
        start_idx = max_time_num + start_idx
    if end_idx < 0:  # original time range is [start_idx, end_idx] containing start_idx and end_idx
        end_idx = max_time_num + end_idx + 1
    else:
        end_idx = end_idx + 1

    if method == 'DynGEM':
        assert duration == 1
    assert start_idx + 1 - duration >= 0
    assert duration > look_back

    t1 = time.time()
    time_list = []

    print('start ' + method + ' embedding!')
    for idx in range(start_idx, end_idx):
        print('idx = ', idx)
        # As DynGEM, DynAE, DynRNN, DynAERNN use original adjacent matrices as their input, so normalization is not necessary(normalization=Fals, add_eye=False) !
        adj_list = data_loader.get_date_adj_list(origin_base_path, start_idx=idx - duration + 1, duration=duration, sep=file_sep, normalize=False, add_eye=False, data_type='matrix')
        adj_list = [adj.tolil() for adj in adj_list]
        model = model_dict[method](input_dim=node_num, output_dim=embed_dim, look_back=look_back, n_units=n_units, ae_units=ae_units, rnn_units=rnn_units, bias=bias)
        if method == 'DynGEM':
            loss = DynGEMLoss(alpha=alpha, beta=beta, nu1=nu1, nu2=nu2)
            batch_generator = DynGEMBatchGenerator(node_list=node_list, batch_size=batch_size, beta=beta, shuffle=shuffle, has_cuda=has_cuda)
            batch_predictor = DynGEMBatchPredictor(node_list=node_list, batch_size=batch_size, has_cuda=has_cuda)
        else:
            loss = DynGraph2VecLoss(beta=beta, nu1=nu1, nu2=nu2)
            batch_generator = BatchGenerator(node_list=node_list, batch_size=batch_size, look_back=look_back, beta=beta, shuffle=shuffle, has_cuda=has_cuda)
            batch_predictor = BatchPredictor(node_list=node_list, batch_size=batch_size, has_cuda=has_cuda)
        trainer = DynamicEmbedding(base_path=base_path, origin_folder=origin_folder, embedding_folder=embedding_folder, node_list=nodes_set['node'].tolist(), model=model, loss=loss,
                                             batch_generator=batch_generator, batch_predictor=batch_predictor, model_folder=model_folder, has_cuda=has_cuda)
        cost_time = trainer.learn_embedding(adj_list, epoch=epoch, lr=lr, idx=idx, model_file=model_file, load_model=load_model, export=export)
        time_list.append(cost_time)

    # record time cost of DynGEM, DynAE, DynRNN, DynAERNN
    if record_time:
        df_output = pd.DataFrame({'time': time_list})
        df_output.to_csv(os.path.join(base_path, method + '_time.csv'), sep=',', index=False)
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')