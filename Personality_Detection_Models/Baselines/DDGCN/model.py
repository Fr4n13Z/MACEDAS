# -*- coding: utf-8 -*-
import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Linear
from transformers import BertConfig, BertModel

from .utils import HardConcrete


class DDGCN(nn.Module):
    def __init__(self, args: Any, device: torch.device):
        super(DDGCN, self).__init__()
        self.args = args
        self.device = device
        self.pad_id = 0

        # Initialize BERT model
        config = BertConfig.from_pretrained(
            args.ptm_path,
            output_hidden_states=True,
            local_files_only=True
        )
        self.embed_model = BertModel.from_pretrained(
            args.ptm_path,
            config=config,
            from_tf=False,
            local_files_only=True
        )

        # Model components
        self.dropout = nn.Dropout(args.dropout)
        self.args.embedding_dim = args.hidden_size
        self.deepGCN = MultiDGCN(args).to(device)

    def _get_cls_embeddings(self, post_token_ids: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embeddings from BERT"""
        batch_size, num_posts, seq_len = post_token_ids.shape
        flat_ids = post_token_ids.view(-1, seq_len)
        attn_mask = (flat_ids != self.pad_id).float()

        # Get [CLS] token embeddings
        outputs = self.embed_model(
            input_ids=flat_ids,
            attention_mask=attn_mask
        )
        cls_tokens = outputs[0][:, 0]  # Take first token ([CLS])
        return cls_tokens.view(batch_size, num_posts, -1)

    @staticmethod
    def _compute_context_node(cls_token: torch.Tensor,
                              posts_mask: torch.Tensor) -> torch.Tensor:
        """Compute context node as masked average of CLS tokens"""
        masked_tokens = cls_token * posts_mask.unsqueeze(-1)
        sum_tokens = masked_tokens.sum(dim=1)
        post_counts = posts_mask.sum(dim=1, keepdim=True)
        return (sum_tokens / post_counts).unsqueeze(1)

    def forward(self, post_token_ids: torch.Tensor, posts_mask: torch.Tensor) -> tuple:
        # Validate input
        if self.args.num_labels not in {4, 5}:
            raise ValueError(f"Unsupported label_num: {self.args.num_labels}. Must be 4 or 5")

        # Get token embeddings
        cls_token = self._get_cls_embeddings(post_token_ids)

        # Compute context node
        c_node = self._compute_context_node(cls_token, posts_mask)

        # Forward through GCN
        gcn_outputs = self.deepGCN(posts_mask, cls_token, c_node)

        # Dynamically handle output based on label_num
        return (*gcn_outputs[:-2],  # logits
                gcn_outputs[-2],  # l0_attr
                gcn_outputs[-1])  # retain_scores


class MultiDGCN(nn.Module):
    def __init__(self, args: argparse.Namespace):
        # hidden_dim: the dimension fo hidden vector
        super(MultiDGCN, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout)
        self.in_dim = self.args.gcn_hidden_size
        self.dgcns = nn.ModuleList([
            DynamicDeeperGCN(self.args, in_dim=args.embedding_dim,
                             hidden_size=args.gcn_hidden_size,
                             num_layers=args.gnn_num_layers)
            for _ in range(self.args.num_labels)
        ])
        mlp_layers = [
            nn.Linear(self.in_dim, args.final_hidden_size),
            nn.ReLU()
        ]
        for _ in range(self.args.mlp_num - 1):
            mlp_layers += [
                nn.Linear(self.args.final_hidden_size, self.args.final_hidden_size),
                nn.ReLU()
            ]
        self.fcs_layers = nn.ModuleList([
            nn.Sequential(*mlp_layers.copy())
            for _ in range(self.args.num_labels)
        ])
        self.fc_finals = nn.ModuleList([
            nn.Linear(self.args.final_hidden_size, self.args.num_classes)
            for _ in range(self.args.num_labels)
        ])

    def forward(self, pmask, feature, c_node):
        B = pmask.size(0)
        device = pmask.device
        # Process mask
        n_pmask = torch.cat((pmask, torch.ones(B, 1).to(device)), 1) \
            if not self.args.no_special_node else pmask
        # Process features
        extended_feature = torch.cat((feature, c_node), 1) \
            if not self.args.no_special_node else feature
        logits = []
        l0_attrs = []
        retain_scores = []
        rs_out = None
        for i in range(self.args.num_labels):
            gcn_out, rs, _, l0_attr = self.dgcns[i](n_pmask, extended_feature)
            out = gcn_out[:, -1]
            if rs is not None:
                rs_out = rs[:, -1]
            # Classifier
            x = self.dropout(out)
            x = self.fcs_layers[i](x)
            logit = self.fc_finals[i](x)
            # Collect outputs
            logits.append(logit)
            l0_attrs.append(l0_attr)
            if rs is not None:
                retain_scores.append(rs_out)
        # Prepare return values
        return_values = tuple(logits) + (
            tuple(l0_attrs),
            tuple(retain_scores) if retain_scores else None
        )

        return return_values


class DynamicDeeperGCN(nn.Module):
    """
    GCN module operated on L2C
    """

    def __init__(self, args: argparse.Namespace, in_dim, hidden_size, num_layers):
        super(DynamicDeeperGCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # self.layer_norm = LayerNormalization(in_dim)
        # gcn layer
        self.A = nn.ModuleList()

        self.proj = nn.Linear(self.in_dim, 1)

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.A.append(L2C(input_dim, input_dim, "train"))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def sparse_statistics(self, adj, pmask):
        p_sum = pmask.sum(-1)  # (B)
        c_adj = adj[:, -1, :-1]  # (B,N)
        g_adj = adj[:, :-1, :-1]  # (B, N, N)
        g_edges = g_adj.sum(-1).sum(-1)  # (B)
        c_node_sparsity = c_adj.sum(-1) / p_sum
        g_edge_sparsity = g_edges / p_sum ** 2
        return c_node_sparsity, g_edge_sparsity

    def forward(self, pmask, feature):
        B = pmask.size(0)
        p_mask = pmask.unsqueeze(-1)  # (B,N+1,1)
        full_adj = p_mask.bmm(p_mask.transpose(2, 1))  # (B,N+1,N+1)

        preds = []
        preds.append(feature)
        adjs = []
        c_sparsity, g_sparsity = [], []
        total_l0loss = 0

        for l in range(self.num_layers):
            residual = feature
            # if l == 0: #single-hop
            adj, l0_loss = self.A[l](feature, full_adj)
            # adj = adj.detach()
            total_l0loss += l0_loss
            c_spar, g_spar = self.sparse_statistics(adj, pmask)
            adjs.append(adj)
            c_sparsity.append(c_spar)
            g_sparsity.append(g_spar)

            denom = torch.diag_embed(adj.sum(2))  # B, N, N
            deg_inv_sqrt = denom.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt = deg_inv_sqrt.detach()
            adj_ = deg_inv_sqrt.bmm(adj)
            adj_ = adj_.bmm(deg_inv_sqrt)

            feature = adj_.transpose(-1, -2).bmm(feature)
            preds.append(feature)
        #
        c_sparsity = torch.stack(c_sparsity, dim=-1)  # (B, L)
        g_sparsity = torch.stack(g_sparsity, dim=-1)  # (B, L)
        total_l0loss /= self.num_layers
        l0_attr = (total_l0loss, c_sparsity, g_sparsity)
        pps = torch.stack(preds, dim=2)  # (B, N, L+1, D)
        retain_score = self.proj(pps)  # (B, N, L+1, 1)
        retain_score0 = torch.sigmoid(retain_score).view(-1, self.num_layers + 1, 1)  # (B*N, L+1, 1)
        # retain_score0 = torch.softmax(retain_score, dim=-2).view(-1, self.num_layers+1, 1) # (B*N, L+1, 1)
        retain_score = retain_score0.transpose(-1, -2)  # (B* N+1, 1, L+1)
        out = retain_score.bmm(
            pps.view(-1, self.num_layers + 1, self.in_dim))  # (B*N, 1, L+1) * (B*N, L+1, D) = (B* N+1, 1, D)
        out = out.squeeze(1).view(B, -1, self.in_dim)  # (B, N+1, D)

        return out, retain_score0.view(B, -1, self.num_layers + 1), torch.stack(adjs, dim=-1), l0_attr


class L2C(nn.Module):
    device = None

    def __init__(self, v_dim, h_dim, train):
        super(L2C, self).__init__()

        training = train.find("train") != -1
        self.hard_gates = HardConcrete(train=training)

        self.transforms = torch.nn.Sequential(
            Linear(v_dim, h_dim, False),
            ReLU(),
            Linear(h_dim, v_dim, False),
        )

    def forward(self, nodes_emb, adj_mat):
        srcs = self.transforms(nodes_emb)

        squeezed_a = torch.bmm(srcs, nodes_emb.transpose(-1, -2))

        """
        if False:  # undirected graph for ablation experiments
            tril_a = torch.tril(squeezed_a)
            tril_no_diag = torch.tril(squeezed_a, diagonal=-1)
            squeezed_a = tril_a + tril_no_diag.transpose(-1, -2)
        """
        gate, penalty = self.hard_gates(squeezed_a, summarize_penalty=False)

        gate = gate * (adj_mat > 0).float()
        penalty_norm = (adj_mat > 0).sum().float()
        penalty = (penalty * (adj_mat > 0).float()).sum() / (penalty_norm + 1e-8)

        return gate, penalty


class GCN(nn.Module):
    """
    GCN module operated on graphs
    """

    def __init__(self, args: argparse.Namespace, in_dim, hidden_size, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.W.append(nn.Linear(input_dim, hidden_size))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, feature):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.num_layers):
            Ax = adj.bmm(feature)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](feature)  # self loop
            AxW /= denom

            gAxW = F.relu(AxW)
            # gAxW = AxW
            feature = self.dropout(gAxW) if l < self.num_layers - 1 else gAxW
        return feature, mask


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class DynamicGCN(nn.Module):
    """
    GCN module operated on L2C
    """

    def __init__(self, args: argparse.Namespace, in_dim, hidden_size, num_layers):
        super(DynamicGCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.layer_norm = LayerNormalization(in_dim)
        # gcn layer
        self.W = nn.ModuleList()
        self.A = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.W.append(nn.Linear(input_dim, hidden_size))
            self.A.append(L2C(input_dim, input_dim, "train"))  # L2C

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def sparse_statistics(self, adj, pmask):
        p_sum = pmask.sum(-1)  # (B)
        c_adj = adj[:, -1, :-1]  # (B,N)
        g_adj = adj[:, :-1, :-1]  # (B, N, N)
        g_edges = g_adj.sum(-1).sum(-1)  # (B)
        c_node_sparsity = c_adj.sum(-1) / p_sum
        g_edge_sparsity = g_edges / p_sum ** 2
        return c_node_sparsity, g_edge_sparsity

    def forward(self, pmask, feature):
        # gcn layer
        p_mask = pmask.unsqueeze(-1)  # (B,N+1,1)
        full_adj = p_mask.bmm(p_mask.transpose(2, 1))  # (B,N+1,N+1)
        c_sparsity, g_sparsity = [], []
        total_l0loss = 0

        for l in range(self.num_layers):
            residual = feature
            adj, l0_loss = self.A[l](feature, full_adj)
            # adj = adj.detach()
            total_l0loss += l0_loss
            c_spar, g_spar = self.sparse_statistics(adj, pmask)
            c_sparsity.append(c_spar)
            g_sparsity.append(g_spar)
            denom = torch.diag_embed(adj.sum(2))  # B, N, N
            deg_inv_sqrt = denom.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt = deg_inv_sqrt.detach()
            adj_ = deg_inv_sqrt.bmm(adj)
            adj_ = adj_.bmm(deg_inv_sqrt)

            Ax = adj_.transpose(-1, -2).bmm(feature)
            AxW = self.W[l](Ax)
            gAxW = F.relu(AxW)

            feature = self.dropout(gAxW) + residual
        c_sparsity = torch.stack(c_sparsity, dim=-1)  # (B, L)
        g_sparsity = torch.stack(g_sparsity, dim=-1)  # (B, L)
        total_l0loss /= self.num_layers
        l0_attr = (total_l0loss, c_sparsity, g_sparsity)
        return feature, l0_attr
