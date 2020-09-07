# coding: utf-8
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import accuracy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


#####################
# Loss modules for supervised learning


# Unsupervised loss classes
class NegativeSamplingLoss(nn.Module):
    node_pair_list: list
    node_freq_list: list
    neg_sample_num: int
    Q: float

    def __init__(self, node_pair_list, neg_freq_list, neg_num=20, Q=10):
        super(NegativeSamplingLoss, self).__init__()
        self.node_pair_list = node_pair_list
        self.neg_freq_list = neg_freq_list
        self.neg_sample_num = neg_num
        self.Q = Q

    def forward(self, input_list):
        assert len(input_list) == 2
        node_embedding, batch_indices = input_list[0], input_list[1]
        node_embedding = [node_embedding] if not isinstance(node_embedding, list) and len(node_embedding.size()) == 2 else node_embedding
        return self.__negative_sampling_loss(node_embedding, batch_indices)

    # Negative sampling loss used for unsupervised learning to preserve local connective proximity
    def __negative_sampling_loss(self, node_embedding, batch_indices):
        bce_loss = nn.BCEWithLogitsLoss()
        neighbor_loss = Variable(torch.tensor([0.], device=batch_indices.device), requires_grad=True)
        timestamp_num = len(node_embedding)
        # print('timestamp num: ', timestamp_num)
        for i in range(timestamp_num):
            embedding_mat = node_embedding[i]   # tensor
            node_pairs = self.node_pair_list[i]  # list
            # print('node pairs: ', len(node_pairs[0]))
            node_freqs = self.neg_freq_list[i]  # tensor
            sample_num, node_indices, pos_indices, neg_indices = self.__get_node_indices(batch_indices, node_pairs, node_freqs)
            if sample_num == 0:
                continue
            # For this calculation block, we refer to some implementation details in https://github.com/aravindsankar28/DySAT/blob/master/models/DySAT/models.py
            # or https://github.com/kefirski/pytorch_NEG_loss/blob/master/NEG_loss/neg.py, or https://github.com/williamleif/GraphSAGE/blob/master/graphsage/models.py
            # or https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py.
            # Here when calculating the neg_score, we use the 'matmul' operation. We can also use the 'mul' operation to calculate neg_score just like calculating pos_score
            # and using the 'mul' operation can reduce the computation complexity of neg_score calculation.
            pos_score = torch.sum(embedding_mat[node_indices].mul(embedding_mat[pos_indices]), dim=1)
            neg_score = torch.sum(embedding_mat[node_indices].matmul(torch.transpose(embedding_mat[neg_indices], 1, 0)), dim=1)
            # print('pos score: ', pos_score.mean().item(), 'pos max: ', pos_score.max().item(), 'pos min: ', pos_score.min().item())
            # print('neg score: ', neg_score.mean().item(), 'neg max: ', neg_score.max().item(), 'neg min: ', neg_score.min().item())
            pos_loss = bce_loss(pos_score, torch.ones_like(pos_score))
            neg_loss = bce_loss(neg_score, torch.zeros_like(neg_score))
            loss_val = pos_loss + self.Q * neg_loss
            neighbor_loss = neighbor_loss + loss_val
            # print('neighbor loss: ', neighbor_loss.item())
            ######################
        return neighbor_loss

    def __get_node_indices(self, batch_indices, node_pairs: np.ndarray, node_freqs: np.ndarray):
        device = batch_indices.device
        dtype = batch_indices.dtype
        node_indices, pos_indices, neg_indices = [], [], []
        random.seed()

        sample_num = 0
        for node_idx in batch_indices:
            # print('node pair type: ', type(node_pairs))
            neighbor_num = len(node_pairs[node_idx])
            if neighbor_num <= self.neg_sample_num:
                pos_indices += node_pairs[node_idx]
                real_num = neighbor_num
            else:
                pos_indices += random.sample(node_pairs[node_idx], self.neg_sample_num)
                real_num = self.neg_sample_num
            node_indices += [node_idx] * real_num
            sample_num += real_num
        if sample_num == 0:
            return sample_num, None, None, None
        neg_indices += random.sample(node_freqs, self.neg_sample_num)

        node_indices = torch.tensor(node_indices, dtype=dtype, device=device)
        pos_indices = torch.tensor(pos_indices, dtype=dtype, device=device)
        neg_indices = torch.tensor(neg_indices, dtype=dtype, device=device)
        return sample_num, node_indices, pos_indices, neg_indices


# Reconstruction loss used for k-core based structure preserving methods(CGCN-S and CTGCN-S)
class ReconstructionLoss(nn.Module):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, input_list):
        assert len(input_list) == 3
        node_embedding, structure_embedding, batch_indices = input_list[0], input_list[1], input_list[2]
        node_embedding = [node_embedding] if not isinstance(node_embedding, list) and len(node_embedding.size()) == 2 else node_embedding
        structure_embedding = [structure_embedding] if not isinstance(structure_embedding, list) and len(structure_embedding.size()) == 2 else structure_embedding
        return self.__reconstruction_loss(node_embedding, structure_embedding, batch_indices)

    # Reconstruction loss used for unsupervised learning to preserve local connective proximity
    @staticmethod
    def __reconstruction_loss(node_embedding, structure_embedding, batch_indices=None):
        mse_loss = nn.MSELoss()
        structure_loss = 0
        timestamp_num = len(node_embedding)
        for i in range(timestamp_num):
            embedding_mat = node_embedding[i]
            structure_mat = structure_embedding[i]

            if batch_indices is not None:
                structure_loss = structure_loss + mse_loss(structure_mat[batch_indices], embedding_mat[batch_indices])
            else:
                structure_loss = structure_loss + mse_loss(structure_mat, embedding_mat)
        return structure_loss


# Variational autoencoder loss function used for VGRNN method
class VAELoss(nn.Module):
    eps: float

    def __init__(self, eps=1e-10):
        super(VAELoss, self).__init__()
        self.eps = eps

    def forward(self, input_list):
        enc_mean_list, enc_std_list, prior_mean_list, prior_std_list, dec_list, adj_list = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5]
        assert len(input_list) == 6
        timestamp_num = len(adj_list)

        kld_loss = 0
        nll_loss = 0

        for time in range(timestamp_num):
            kld_loss += self.__kld_gauss(enc_mean_list[time], enc_std_list[time], prior_mean_list[time], prior_std_list[time])
            nll_loss += self.__nll_bernoulli(dec_list[time], adj_list[time].to_dense())
        main_loss = kld_loss + nll_loss
        return main_loss

    def __kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) + (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) / torch.pow(std_2 + self.eps, 2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    @staticmethod
    def __nll_bernoulli(logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits, target=target_adj_dense, pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss


#####################
# Loss modules for supervised learning


# Supervised classification loss
class ClassificationLoss(nn.Module):
    n_class: int

    def __init__(self, n_class):
        super(ClassificationLoss, self).__init__()
        self.n_class = n_class

    def forward(self, input_list, batch_labels):
        cls_res = input_list
        cls_res = [cls_res] if not isinstance(cls_res, list) and len(cls_res.size()) == 2 else cls_res
        return self.__classification_loss(cls_res, batch_labels)

    def __classification_loss(self, cls_res, batch_labels):
        # log_softmax = nn.LogSoftmax(dim=1)
        # nll_loss = nn.NLLLoss()
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        total_loss, total_acc, total_auc = 0, 0, 0
        timestamp_num = len(cls_res)
        for i in range(timestamp_num):
            preds = cls_res[i]
            labels = batch_labels[i]
            if len(preds.size()) == 1:
                loss_val = bce_loss(preds, labels)
                preds = preds.unsqueeze(1)
                preds = torch.cat((1 - torch.sigmoid(preds), torch.sigmoid(preds)), dim=1)
                auc_val = roc_auc_score(labels.cpu().detach().numpy(), torch.sigmoid(preds[:, 1]).cpu().detach().numpy())
            else:
                assert preds.shape[1] == self.n_class
                loss_val = ce_loss(preds, labels)
                import warnings
                warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=71)
                auc_val = roc_auc_score(label_binarize(labels.cpu().detach().numpy(), np.arange(self.n_class)), torch.softmax(preds, dim=1).cpu().detach().numpy(), multi_class="ovr", average='micro')
            acc_val = accuracy(preds, labels)
            total_loss = total_loss + loss_val
            total_acc = total_acc + acc_val
            total_auc = total_auc + auc_val
        total_acc /= timestamp_num
        total_auc /= timestamp_num
        return total_loss, total_acc, total_auc


# Supervised structure preserving binary classification loss. It combines the reconstruction loss and the negative likelihood loss.
# This loss is only used for CGCN-S and CTGCN-S when supervised learning is needed.
class StructureClassificationLoss(nn.Module):

    def __init__(self, n_class):
        super(StructureClassificationLoss, self).__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.classification_loss = ClassificationLoss(n_class)

    def forward(self, input_list, batch_labels):
        assert len(input_list) == 3
        cls_res, node_embedding, structure_embedding = input_list[0], input_list[1], input_list[2]
        structure_input_list = [node_embedding, structure_embedding, None]
        structure_loss = self.reconstruction_loss(structure_input_list)
        cls_loss, total_acc, total_auc = self.classification_loss(cls_res, batch_labels)
        total_loss = structure_loss + cls_loss
        # print('structure loss: ', structure_loss.item(), 'cls loss: ', cls_loss.item())
        return total_loss, total_acc, total_auc


# Variational autoencoder loss function used for VGRNN method
class VAEClassificationLoss(nn.Module):

    def __init__(self, n_class, eps=1e-10):
        super(VAEClassificationLoss, self).__init__()
        self.vae_loss = VAELoss(eps=eps)
        self.classification_loss = ClassificationLoss(n_class)

    def forward(self, input_list, batch_labels):
        assert len(input_list) == 7
        vae_data_list = input_list[:-1]
        cls_list = input_list[-1]
        vae_loss = self.vae_loss(vae_data_list)
        classification_loss, total_acc, total_auc = self.classification_loss(cls_list, batch_labels)
        total_loss = vae_loss + classification_loss
        return total_loss, total_acc, total_auc
