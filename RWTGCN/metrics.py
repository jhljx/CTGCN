import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MainLoss(nn.Module):
    node_pair_list: list
    node_freq_list: list
    neg_sample_num: int
    Q: int

    def __init__(self, neg_num=20, Q=10):
        super(MainLoss, self).__init__()
        self.neg_sample_num = neg_num
        self.Q = Q

    def set_node_info(self, node_pair_list, neg_freq_list):
        self.node_pair_list = node_pair_list
        self.neg_freq_list = neg_freq_list

    def forward(self, embedding_list, batch_node_idxs):
        timestamp_num = len(embedding_list)
        assert timestamp_num == len(self.node_pair_list)
        bce_loss = nn.BCEWithLogitsLoss()
        if torch.cuda.is_available():
            loss_val_sum = torch.tensor([0.]).cuda()
        else:
            loss_val_sum = torch.tensor([0.])
        for i in range(timestamp_num):
            embedding_mat = embedding_list[i]
            node_pair_dict = self.node_pair_list[i]
            node_freq = self.neg_freq_list[i]
            node_idxs, pos_idxs, neg_idxs  = [], [], []

            for node_idx in batch_node_idxs:
                # print('nid = ', node_idx, ', type = ')
                neighbor_num = len(node_pair_dict[node_idx])
                if neighbor_num <= self.neg_sample_num:
                    pos_idxs += node_pair_dict[node_idx]
                    node_idxs += [node_idx] * len(node_pair_dict[node_idx])
                else:
                    pos_idxs += random.sample(node_pair_dict[node_idx], self.neg_sample_num)
                    node_idxs += [node_idx] * self.neg_sample_num
            assert len(node_idxs) <= len(batch_node_idxs) * self.neg_sample_num
            neg_idxs += random.sample(node_freq, self.neg_sample_num)

            pos_score = torch.sum(embedding_mat[node_idxs].mul(embedding_mat[pos_idxs]), dim=1)
            neg_score = -1.0 * torch.sum(embedding_mat[node_idxs].matmul(torch.transpose(embedding_mat[neg_idxs], 1, 0)), dim=1)

            loss_val = torch.mean(bce_loss(pos_score, torch.ones_like(pos_score))) + \
                       self.Q * torch.mean(bce_loss(neg_score, torch.zeros_like(neg_score)))
            loss_val_sum += loss_val
        return loss_val_sum