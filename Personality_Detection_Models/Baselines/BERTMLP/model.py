# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import BertConfig, BertModel


class BERTMLPModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(BERTMLPModel, self).__init__()
        self.args = args
        self.BERT_config = BertConfig.from_pretrained(self.args.bert_path, local_files_only=True)
        self.BERT_config.output_hidden_states = True
        self.BERT = BertModel.from_pretrained(self.args.bert_path, config=self.BERT_config, local_files_only=True)
        self.linear1 = nn.Linear(self.args.hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifiers = ModuleList([nn.Linear(128, 2)
                                       for _ in range(self.args.num_labels)])
        self.pad_id = 0

    def forward(self, post_ids=None, posts_mask=None):
        attn_mask = (post_ids != self.pad_id).float()  # (B, N, L)
        node_ids = post_ids.view(-1, self.args.max_length)
        attention_mask = attn_mask.view(-1, self.args.max_length)
        cls_token = self.BERT(input_ids=node_ids, attention_mask=attention_mask).last_hidden_state[:, :1]
        cls_token = torch.where(torch.isnan(cls_token), torch.full_like(cls_token, 1e-6), cls_token)
        last_semantic_reply = cls_token.view(-1, self.args.max_posts, self.args.hidden_size)
        mean_pooling = last_semantic_reply.masked_fill(
            (1 - posts_mask)[:, :, None].expand_as(last_semantic_reply).bool(), 0). \
                           sum(dim=-2) / (posts_mask.sum(dim=-1)[:, None].expand(-1, self.args.hidden_size) + 1e-8)
        x = self.linear1(mean_pooling)
        x = self.relu(x)
        x = self.dropout(x)
        pred_prob = torch.stack([classifier(x) for classifier in self.classifiers],
                                dim=1).view(-1, 2)

        return pred_prob
