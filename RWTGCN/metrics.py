import torch
import torch.nn as nn
from RWTGCN.layers import Infomax


class MainLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, score_list):
        bce_loss = nn.BCEWithLogitsLoss()
        main_loss = 0

        for i in range(len(score_list)):
            pos_score, neg_score = score_list[i]
            score_num = pos_score.size()[0]
            embedding_loss = bce_loss(pos_score, torch.ones(score_num)) + bce_loss(neg_score,
                                                                                   torch.zeros(score_num))
            if i == 0:
                main_loss = embedding_loss
            else:
                main_loss += embedding_loss
        return main_loss
