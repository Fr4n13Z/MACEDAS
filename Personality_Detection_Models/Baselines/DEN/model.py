import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN
from transformers import BertModel


class DENModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(DENModel, self).__init__()
        self.args = args
        self.post_encoder = BertModel.from_pretrained(args.ptm_path, local_files_only=True)
        self.long_term_gcn = GCN(in_channels=args.glove_embed_dim,
                                 hidden_channels=args.hidden_dim,
                                 out_channels=args.hidden_dim,
                                 num_layers=args.gcn_layers)
        self.interact_gcn = GCN(in_channels=args.hidden_dim,
                                out_channels=args.hidden_dim,
                                hidden_channels=args.hidden_dim,
                                num_layers=1)
        self.alpha = torch.full((self.args.batch_size, 1), self.args.alpha)
        self.gated_fusion_layer = nn.Linear(args.hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.classifiers = nn.ModuleList([nn.Linear(args.hidden_dim, 2)
                                       for _ in range(self.args.num_labels)])
        self.pad_id = 0

        # Freeze BERT layers if specified in config
        if args.freeze_bert is True:
            for param in self.post_encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _average_masked_word_nodes(word_reps: torch.Tensor,
                                   word_mask: torch.Tensor) -> torch.Tensor:
        """Compute context node as masked average of CLS tokens"""
        masked_tokens = word_reps * word_mask.unsqueeze(-1)
        sum_nodes = masked_tokens.sum(dim=1)
        post_counts = word_mask.sum(dim=1, keepdim=True)
        return (sum_nodes / post_counts).unsqueeze(1)

    def forward(self, post_ids, post_mask, word_nodes, word_mask, word2word_index, post2word_index):
        self.alpha = self.alpha.to(post_ids.device)
        attn_mask = (post_ids != self.pad_id).float()
        post_rep = self.post_encoder(post_ids, attn_mask)[0][:, 0].view(-1, self.args.max_posts, self.args.hidden_dim)
        word_rep = self.long_term_gcn(word_nodes, word2word_index).view(-1, self.args.max_word_num, self.args.hidden_dim)

        inter_rep = torch.concatenate([post_rep, word_rep], dim=1).view(-1, self.args.hidden_dim)
        inter_rep = self.interact_gcn(inter_rep, post2word_index).view(-1,  self.args.max_posts + self.args.max_word_num, self.args.hidden_dim)

        post_rep = inter_rep[:, :self.args.max_posts, :]
        word_rep = inter_rep[:, self.args.max_posts:, :]

        u_t = self._average_masked_word_nodes(word_rep, word_mask.view(-1, self.args.max_word_num)).view(-1, self.args.hidden_dim)
        u_s = self._average_masked_word_nodes(post_rep, post_mask.view(-1, self.args.max_posts)).view(-1, self.args.hidden_dim)
        combined_rep = torch.cat([u_t, u_s], dim=1)
        self.alpha = self.sigmoid(self.gated_fusion_layer(combined_rep))
        fused_rep = self.alpha * u_t + (1 - self.alpha) * u_s

        pred_prob = torch.stack([classifier(fused_rep) for classifier in self.classifiers],
                                dim=1).view(-1, 2)
        return pred_prob