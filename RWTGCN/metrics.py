import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MainLoss(nn.Module):
    node_pair_list: list
    node_freq_list: list
    node2idx_dict: dict
    node_num: int
    neg_sample_num: int

    def __init__(self, neg_num=20, Q=10):
        super(MainLoss, self).__init__()
        self.neg_sample_num = neg_num
        self.Q = Q

    def set_node_info(self, node_pair_list, neg_freq_list):
        self.node_pair_list = node_pair_list
        self.neg_freq_list = neg_freq_list
        self.node2idx_dict = dict(zip(self.full_node_list, np.arange(self.node_num).tolist()))
        for i in range(len(node_pair_list)):
            for node in self.full_node_list:
                node_pair_dict = self.node_pair_list[i]
                node_pair_dict[node] = [self.node2idx_dict[neighbor] for neighbor in node_pair_dict[node]]

    def forward(self, embedding_list, batch_nodes):
        timestamp_num = len(embedding_list)
        assert timestamp_num == len(self.node_pair_list)
        bce_loss = nn.BCEWithLogitsLoss()
        loss_val_sum = torch.tensor([0])
        for i in range(timestamp_num):
            embedding_mat = embedding_list[i]
            node_pair_dict = self.node_pair_list[i]
            node_freq = self.neg_freq_list[i]
            node_idxs1, pos_idxs, node_idxs2, neg_idxs  = [], [], [], []

            for node in batch_nodes:
                # print('nid = ', nid)
                nid = self.node2idx_dict[node]
                pos_idxs += node_pair_dict[node]
                node_idxs1 += [nid] * len(node_pair_dict[node])
                neg_idxs += random.sample(node_freq, self.neg_sample_num)
                node_idxs2 += [nid] * len(node_pair_dict[node])

            pos_score = torch.sum(embedding_mat[node_idxs1].multiply(embedding_mat[pos_idxs]), dim=1)
            neg_score = -1.0 * torch.sum(embedding_mat[node_idxs1].multiply(embedding_mat[pos_idxs]), dim=1)

            loss_val = torch.mean(bce_loss(pos_score, torch.ones_like(pos_score))) + \
                       self.Q * torch.mean(bce_loss(neg_score, torch.zeros_like(neg_score)))
            loss_val_sum += loss_val
        return loss_val_sum