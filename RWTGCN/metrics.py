import torch
import torch.nn as nn


class MainLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding_pos_score, embedding_neg_score, origin_matrix):
        bce_loss = nn.BCEWithLogitsLoss()
        score_num = embedding_pos_score.size()[0]
        embedding_loss = bce_loss(embedding_pos_score, torch.ones(score_num)) + bce_loss(embedding_neg_score,
                                                                                         torch.zeros(score_num))
        return embedding_loss
