import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys
import gc
import time
import json
import torch
from CTGCN.utils import check_and_make_path, get_normalize_adj, get_sp_adj_mat, sparse_mx_to_torch_sparse_tensor
sys.path.append('..')


# A class which is designed for loading various kinds of data
class DataLoader:
    full_node_list: list
    node_num: int
    max_time_num: int

    def __init__(self, node_list, max_time_num):
        self.max_time_num = max_time_num
        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)
        return

    # get adjacent matrices for a graph list, this function supports Tensor type-based adj and sparse.coo type-based adj.
    def get_date_adj_list(self, origin_base_path, start_idx, duration, data_type='tensor'):
        date_dir_list = sorted(os.listdir(origin_base_path))
        date_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            original_graph_path = os.path.join(origin_base_path, date_dir_list[i])
            spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep='\t')
            # spmat = sp.coo_matrix((np.exp(alpha * spmat.data), (spmat.row, spmat.col)), shape=(self.node_num, self.node_num))
            spmat = get_normalize_adj(spmat)
            if data_type == 'tensor':
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                date_adj_list.append(sptensor.cuda() if torch.cuda.is_available() else sptensor)
            elif data_type == 'matrix':
                spmat = get_normalize_adj(spmat)
                date_adj_list.append(spmat)
            else:
                raise AttributeError('Unsupported adjacent matrix data type!')
        # print(len(date_adj_list))
        return date_adj_list

    # get k-core sub-graph adjacent matrices for a graph list, it is a 2-layer nested list, outer layer for graph,
    # inner layer for k-cores.
    def get_core_adj_list(self, core_base_path, start_idx, duration):
        date_dir_list = sorted(os.listdir(core_base_path))
        time_stamp_num = len(date_dir_list)
        assert start_idx < time_stamp_num
        core_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            date_dir_path = os.path.join(core_base_path, date_dir_list[i])
            f_list = sorted(os.listdir(date_dir_path))
            tmp_adj_list = []
            for i, f_name in enumerate(f_list):
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                spmat = get_normalize_adj(spmat)
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                tmp_adj_list.append(sptensor.cuda() if torch.cuda.is_available() else sptensor)
            core_adj_list.append(tmp_adj_list)
        # print('core_adj_list len: ', len(core_adj_list))
        return core_adj_list

    # get node co-occurrence pairs of random walk for a graph list, the node pair list is used for negative sampling
    def get_node_pair_list(self, walk_pair_base_path, start_idx, duration):
        walk_file_list = sorted(os.listdir(walk_pair_base_path))
        node_pair_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            walk_file_path = os.path.join(walk_pair_base_path, walk_file_list[i])
            walk_spadj = sp.load_npz(walk_file_path)
            neighbor_list = walk_spadj.tolil().rows
            node_pair_list.append(neighbor_list)
        return node_pair_list

    # get node frequencies of random walk for a graph list, the node frequency list is used for negative sampling
    def get_neg_freq_list(self, node_freq_base_path, start_idx, duration):
        freq_file_list = sorted(os.listdir(node_freq_base_path))
        node_freq_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            freq_file_path = os.path.join(node_freq_base_path, freq_file_list[i])
            with open(freq_file_path, 'r') as fp:
                node_freq_list.append(json.load(fp))
        return node_freq_list

    # load node features, use degree related features
    def get_degree_feature_list(self, origin_base_path, start_idx, duration, init='gaussian'):
        x_list = []
        max_degree = 0
        adj_list = []
        degree_list = []
        ret_degree_list = []
        date_dir_list = sorted(os.listdir(origin_base_path))
        # find the maximal degree for a list of graphs
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            original_graph_path = os.path.join(origin_base_path, date_dir_list[i])
            adj = get_sp_adj_mat(original_graph_path, self.full_node_list, sep='\t')
            adj_list.append(adj)
            degrees = adj.sum(axis=1).astype(np.int)
            max_degree = max(max_degree, degrees.max())
            degree_list.append(degrees)
            ret_degree_list.append(torch.from_numpy(degrees).cuda() if torch.cuda.is_available() else degrees)
        # generate degree_based features
        ret_shape = 0
        for i, degrees in enumerate(degree_list):
            # other structural feature initialization techniques can also be tried to improve performance
            if init == 'gaussian':
                fea_list = []
                for degree in degrees:
                    fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
                fea_arr = np.array(fea_list)
                ret_shape = fea_arr.shape[1]
                fea_tensor = torch.from_numpy(fea_arr)
                x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
                
            elif init == 'combine':
                fea_list = []
                for degree in degrees:
                    fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
                fea_arr = np.array(fea_list)
                ###################
                # here combine the adjacent matrix feature could improve strcutral role classification performance,
                # but if the graph is large, turning a sparse adjacent matrix into a dense adjacent feature matrix will be memory consuming!
                fea_arr = np.hstack((fea_arr, adj_list[i].toarray()))
                ###################
                ret_shape = fea_arr.shape[1]
                fea_tensor = torch.from_numpy(fea_arr)
                x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
                
            elif init == 'one-hot':  # one-hot degree feature
                data = np.ones(degrees.shape[0], dtype=np.int)
                row = np.arange(degrees.shape[0])
                col = degrees.flatten().A[0]
                spmat = sp.csr_matrix((data, (row, col)), shape=(degrees.shape[0], max_degree + 1))
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                x_list.append(sptensor.cuda() if torch.cuda.is_available() else sptensor)
                print('max degree: ', max_degree + 1)
                ret_shape = max_degree + 1
                
            else:
                raise AttributeError('Unsupported feature initialization type!')
        return x_list, ret_shape, ret_degree_list

    # load node features from file, or create one-hot node feature
    def get_feature_list(self, feature_base_path, start_idx, duration):
        if feature_base_path is None:
            x_list = []
            for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
                sptensor = sparse_mx_to_torch_sparse_tensor(sp.eye(self.node_num))
                x_list.append(sptensor.cuda() if torch.cuda.is_available() else sptensor)
            print('len x_list: ', len(x_list))
            return x_list
        else:
            feature_file_list = sorted(os.listdir(feature_base_path))
            x_list = []
            for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
                feature_file_path = os.path.join(feature_base_path, feature_file_list[i])
                df_feature = pd.read_csv(feature_file_path, sep='\t', header=0)
                feature_tensor = torch.from_numpy(df_feature.values)
                x_list.append(feature_tensor.cuda() if torch.cuda.is_available() else feature_tensor)
            print('len x_list: ', len(x_list))
            return x_list


# The base class of embedding
class BaseEmbedding:
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    model_base_path: str

    full_node_list: list
    node_num: int
    timestamp_list: list
    device: torch.device

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model'):
        # file paths
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        self.model_base_path = os.path.abspath(os.path.join(base_path, model_folder))

        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)  # node num
        self.timestamp_list = sorted(os.listdir(self.origin_base_path))

        # cpu gpu
        if torch.cuda.is_available():
            print('GPU')
            device = torch.device('cuda: 0')
        else:
            print('CPU')
            device = torch.device('cpu')
            self.set_thread()
        self.device = device

        self.model = model
        self.loss = loss

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    @staticmethod
    def set_thread(thread_num=None):
        if thread_num is None:
            thread_num = os.cpu_count() - 4
        torch.set_num_threads(thread_num)

    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()

    def save_embedding(self, output_list, start_idx):
        if isinstance(output_list, torch.Tensor) and len(output_list.size()) == 2:  # static embedding
            embedding = output_list
            output_list = [embedding]
        # output_list supports two type: list and torch.Tensor(2d or 3d tensor)
        for i in range(len(output_list)):
            embedding = output_list[i]
            timestamp = self.timestamp_list[start_idx + i].split('.')[0]
            df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
            embedding_path = os.path.join(self.embedding_base_path, timestamp + '.csv')
            df_export.to_csv(embedding_path, sep='\t', header=True, index=True)


# Supervised embedding class
class SupervisedEmbedding(BaseEmbedding):
    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, classifier, model_folder='model'):
        super(SupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder)
        self.classifier = classifier

    def learn_embedding(self, adj_list, x_list, label_list, single_output=True, epoch=50, batch_size=10240, lr=1e-3, start_idx=0, weight_decay=0.,
                        model_file='ctgcn', classifier_file='ctgcn_cls', embedding_type='connection', load_model=False, export=True):
        print('start learning embedding!')
        st = time.time()
        model = self.model
        loss_model = self.loss
        classifier = self.classifier
        if load_model:
            model_path = os.path.join(self.model_base_path, model_file)
            classifier_path = os.path.join(self.model_base_path, classifier_file)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
                model.eval()
            if os.path.exists(classifier_path):
                classifier.load_state_dict(torch.load(classifier_path))
                classifier.eval()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_model = loss_model.to(self.device)
            classifier = classifier.to(self.device)
            torch.cuda.empty_cache()
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()

        embedding_list, structure_list = [], []
        shuffled_node_idxs = np.random.permutation(np.arange(self.node_num))
        train_num = int(np.floor(self.node_num * 0.5))
        val_num = int(np.floor(self.node_num * 0.3))
        idx_train = shuffled_node_idxs[: train_num]
        label_train = label_list[idx_train]
        idx_val = shuffled_node_idxs[train_num: train_num + val_num]
        label_val = label_list[idx_val]
        idx_test = shuffled_node_idxs[train_num + val_num:]
        label_test = label_list[idx_test]

        batch_num = train_num // batch_size
        if train_num % batch_size != 0:
            batch_num += 1
        best_acc = 0
        print('start training!')
        for i in range(epoch):
            node_idx_list = np.random.permutation(np.arange(train_num))
            for j in range(batch_num):
                train_node_idxes = node_idx_list[j * batch_size: min(train_num, (j + 1) * batch_size)]
                batch_node_idxes = idx_train[train_node_idxes]
                batch_labels = label_train[train_node_idxes]
                t1 = time.time()
                if single_output:
                    embedding_list = model(x_list, adj_list)
                    cls_list = classifier(embedding_list)
                    loss_train, acc_train = loss_model(cls_list, batch_node_idxes, batch_labels, loss_type=embedding_type)
                else:
                    embedding_list, structure_list = model(x_list, adj_list)
                    cls_list = classifier(embedding_list)
                    loss_train, acc_train = loss_model(cls_list, batch_node_idxes, batch_labels, loss_type=embedding_type, structure_list=structure_list,
                                                       emb_list=embedding_list)
                loss_train.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                    loss_val, acc_val = loss_model(cls_list, idx_val, label_val, loss_type=embedding_type, structure_list=structure_list,
                                                   emb_list=embedding_list)
                    print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()),
                          'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()), 'cost time: {:.4f}s'.format(time.time() - t1))
                    if acc_val > best_acc:
                        best_acc = acc_val
                        torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
                        torch.save(classifier.state_dict(), os.path.join(self.model_base_path, classifier_file))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # t2 = time.time()
                # print('epoch', i + 1, ', batch num = ', j + 1, ', loss train:', loss_train.item(), ', cost time: ', t2 - t1, ' seconds!')
        print('finish training!')

        # load embedding model and classifier model
        model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
        model.eval()
        classifier.load_state_dict(torch.load(os.path.join(self.model_base_path, classifier_file)))
        classifier.eval()

        print('start model evaluation!')
        if single_output:
            embedding_list = model(x_list, adj_list)
            cls_list = classifier(embedding_list)
            loss_test, acc_test = loss_model(cls_list, idx_test, label_test, loss_type=embedding_type)
        else:
            embedding_list, structure_list = model(x_list, adj_list)
            cls_list = classifier(embedding_list)
            loss_test, acc_test = loss_model(cls_list, idx_test, label_test, loss_type=embedding_type, structure_list=structure_list, emb_list=embedding_list)
        print('Test set results:', 'loss= {:.4f}'.format(loss_test.item()), 'accuracy= {:.4f}'.format(acc_test.item()))
        print('finish evaluation!')

        if export:
            if embedding_type == 'connection':
                output_list = embedding_list
            elif embedding_type == 'structure':
                output_list = structure_list
            else:
                raise AttributeError('Unsupported embedding type!')
            self.save_embedding(output_list, start_idx)

        del adj_list, x_list, embedding_list, model
        self.clear_cache()
        en = time.time()
        print('training total time: ', en - st, ' seconds!')


# Unsupervised embedding class
class UnsupervisedEmbedding(BaseEmbedding):
    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model'):
        super(UnsupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss,
                                                    model_folder=model_folder)

    def learn_embedding(self, adj_list, x_list, single_output=True, epoch=50, batch_size=10240, lr=1e-3,
                        start_idx=0, weight_decay=0., model_file='ctgcn', embedding_type='connection', load_model=False,
                        export=True):
        print('start learning embedding!')
        st = time.time()
        model = self.model
        loss_model = self.loss
        if load_model:
            model_path = os.path.join(self.model_base_path, model_file)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
                model.eval()

        if torch.cuda.is_available():
            print('cuda available!')
            model = model.to(self.device)
            loss_model = loss_model.to(self.device)
            torch.cuda.empty_cache()
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()

        embedding_list, structure_list = [], []
        batch_num = self.node_num // batch_size
        if self.node_num % batch_size != 0:
            batch_num += 1

        print('start training!')
        for i in range(epoch):
            node_idx_arr = np.random.permutation(np.arange(self.node_num))
            for j in range(batch_num):
                batch_node_idxes = node_idx_arr[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                t1 = time.time()
                if single_output:
                    embedding_list = model(x_list, adj_list)
                    loss = loss_model(embedding_list, batch_node_idxes, loss_type=embedding_type)
                else:
                    embedding_list, structure_list = model(x_list, adj_list)
                    loss = loss_model(embedding_list, batch_node_idxes, loss_type=embedding_type, structure_list=structure_list)
                loss.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t2 = time.time()
                print('epoch', i + 1, ', batch num = ', j + 1, ', loss:', loss.item(), ', cost time: ', t2 - t1, ' seconds!')
        print('end training!')

        if export:
            if embedding_type == 'connection':
                output_list = embedding_list
            elif embedding_type == 'structure':
                output_list = structure_list
            else:
                raise AttributeError('Unsupported embedding type!')
            self.save_embedding(output_list, start_idx)

        torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        del adj_list, x_list, embedding_list, model
        self.clear_cache()
        en = time.time()
        print('training total time: ', en - st, ' seconds!')