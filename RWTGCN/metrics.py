import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MainLoss(nn.Module):
    node_pair_list: list
    node_freq_list: list
    full_node_list: list
    node_num: int
    neg_sample_num: int

    def __init__(self, node_list, neg_num=20, Q=10):
        super(MainLoss, self).__init__()

        self.full_node_list = node_list
        self.node_num = len(node_list)
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
            self.neg_freq_list[i] = [self.node2idx_dict[node] for node in self.node_pair_list[i]]

    def forward(self, embedding_list):
        timestamp_num = len(embedding_list)
        assert timestamp_num == len(self.node_pair_list)
        main_loss = 0
        for i in range(timestamp_num):
            embedding_mat = embedding_list[i]
            node_pair_dict = self.node_pair_list[i]
            node_freq = self.neg_freq_list[i]
            node_loss_list = []
            for nid, node in enumerate(self.full_node_list):
                # print('nid = ', nid)
                pos_idxs = node_pair_dict[node]
                neg_idxs = random.sample(node_freq, self.neg_sample_num)
                node_idxs = [nid] * len(node_pair_dict[node])
                pos_score = F.cosine_similarity(embedding_mat[node_idxs], embedding_mat[pos_idxs])
                pos_score = torch.mean(torch.log(torch.sigmoid(pos_score)))
                # print(pos_score.shape)

                node_idxs = [nid] * len(neg_idxs)
                neg_score = F.cosine_similarity(embedding_mat[node_idxs], embedding_mat[neg_idxs])
                neg_score = self.Q * torch.mean(torch.log(1.0 - torch.sigmoid(neg_score)))
                # print(neg_score.shape)
                node_loss_list.append((-pos_score - neg_score).view(1,-1))
            if main_loss == 0:
                # print(node_loss_list)
                main_loss = torch.mean(torch.cat(node_loss_list, 0))
                # print(main_loss)
            else:
                main_loss += torch.mean(torch.cat(node_loss_list))
        return main_loss / timestamp_num