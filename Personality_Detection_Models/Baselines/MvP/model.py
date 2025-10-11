import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math
from typing import Tuple, Optional, Any


class WordAttention(nn.Module):
    """Word-level attention mechanism to focus on personality-revealing words."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))
        # self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.context_vector, mean=0.0, std=0.02)
        # nn.init.constant_(self.bias, 0.0)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, max_posts, max_tokens, hidden_dim]
            mask: [batch_size, max_posts, max_tokens] (1 for real tokens, 0 for padding)
        Returns:
            Weighted post representations [batch_size, max_posts, hidden_dim]
        """
        # Project hidden states
        projected = torch.tanh(self.projection(hidden_states))  # [batch_size, max_posts, max_tokens, hidden_dim]

        # Compute attention scores
        scores = torch.einsum('bpth,h->bpt', projected, self.context_vector)  # [batch_size, max_posts, max_tokens]

        # Apply mask (set padding tokens to -inf)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, max_posts, max_tokens]

        # Weighted sum of hidden states
        weighted_reps = torch.einsum('bpt,bpth->bph', weights, hidden_states)

        return weighted_reps


class PostEncoder(nn.Module):
    """BERT-based post encoder with word-level attention."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.ptm_path, local_files_only=True)
        self.word_attention = WordAttention(args.hidden_dim)

        # Freeze BERT layers if specified in config
        if args.freeze_bert is False:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tokenized posts [batch_size, max_posts, max_tokens]
            attention_mask: Attention mask [batch_size, max_posts, max_tokens]
        Returns:
            Post representations [batch_size, max_posts, hidden_dim]
        """
        batch_size, max_posts, max_tokens = input_ids.shape

        # Flatten for BERT processing
        input_ids = input_ids.view(-1, max_tokens)  # [batch_size * max_posts, max_tokens]
        attention_mask = attention_mask.view(-1, max_tokens)

        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size * max_posts, max_tokens, hidden_dim]

        # Reshape and apply word attention
        hidden_states = hidden_states.view(batch_size, max_posts, max_tokens, -1)
        post_reps = self.word_attention(hidden_states, attention_mask.view(batch_size, max_posts, max_tokens))

        return post_reps  # [batch_size, max_posts, hidden_dim]


class ParameterWhitening(nn.Module):
    """Parameter Whitening layer for expert networks."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter whitening: (x - b) · W"""
        return torch.matmul(x - self.bias, self.weight)


class MultiViewMoE(nn.Module):
    """Multi-view Mixture-of-Experts network."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.num_experts = args.num_experts
        self.hidden_dim = args.hidden_dim
        self.noise_scale = 0.1  # Empirical value for gating noise

        # Initialize experts (each with parameter whitening)
        self.experts = nn.ModuleList([
            ParameterWhitening(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_experts)
        ])

        # Gating router parameters
        self.gate_proj = nn.Linear(self.hidden_dim, self.num_experts)
        self.noise_proj = nn.Linear(self.hidden_dim, self.num_experts)

        # Initialize gate parameters
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.noise_proj.weight)
        nn.init.constant_(self.noise_proj.bias, 0.0)

    def forward(self, post_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            post_reps: Post representations [batch_size, max_posts, hidden_dim]
        Returns:
            user_reps: User representations [batch_size, hidden_dim]
            gate_load: Expert utilization metrics [num_experts]
        """
        batch_size, max_posts, _ = post_reps.shape

        # Compute mean post representation for gating
        mean_post = post_reps.mean(dim=1)  # [batch_size, hidden_dim]

        # Compute gating weights with noise
        gate_logits = self.gate_proj(mean_post)  # [batch_size, num_experts]
        noise = torch.randn_like(gate_logits) * self.noise_scale
        noise = F.softplus(self.noise_proj(mean_post)) * noise  # [batch_size, num_experts]

        gate_weights = F.softmax(gate_logits + noise, dim=-1)  # [batch_size, num_experts]

        # Process each expert
        expert_outputs = []
        for expert in self.experts:
            # Apply expert to all posts
            expert_post_reps = expert(post_reps)  # [batch_size, max_posts, hidden_dim]

            # Average posts to get user representation
            expert_user_rep = expert_post_reps.mean(dim=1)  # [batch_size, hidden_dim]
            expert_outputs.append(expert_user_rep)

        # Stack and weight expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        user_reps = torch.einsum('be,beh->bh', gate_weights, expert_outputs)  # [batch_size, hidden_dim]

        # Compute expert utilization for load balancing
        gate_load = gate_weights.mean(dim=0)  # [num_experts]

        return user_reps, gate_load


class UserConsistencyRegularization(nn.Module):
    """User Consistency Regularization module."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.dropout_rate = args.dropout_rate
        self.num_traits = args.num_labels

        # Prediction head
        self.pred_head = nn.ModuleList([nn.Linear(self.hidden_dim, 2)
                                       for _ in range(self.num_traits)])

        # Initialize parameters
        for i in range(self.num_traits):
            nn.init.xavier_uniform_(self.pred_head[i].weight)
            nn.init.constant_(self.pred_head[i].bias, 0.0)

        # Dropout for augmentation
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def forward(self, user_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_reps: User representations [batch_size, hidden_dim]
        Returns:
            predictions: Personality predictions [batch_size, num_traits]
            ucr_loss: Consistency regularization loss [1]
        """
        # First forward pass
        user_reps_1 = self.dropout1(user_reps)
        user_reps_2 = self.dropout2(user_reps)
        pred_1 = torch.stack([classifier(user_reps_1) for classifier in self.pred_head],
                                dim=1).view(-1, 2)
        pred_2 = torch.stack([classifier(user_reps_2) for classifier in self.pred_head],
                                dim=1).view(-1, 2)
        # Second forward pass with different dropout mask

        # Compute bidirectional KL divergence
        kl_loss = (F.kl_div(F.log_softmax(pred_1, dim=-1), F.softmax(pred_2, dim=-1), reduction='batchmean') +
                   F.kl_div(F.log_softmax(pred_2, dim=-1), F.softmax(pred_1, dim=-1), reduction='batchmean')) / 2

        # Return mean predictions and UCR loss
        return pred_1, kl_loss


class MVP(nn.Module):
    """Main MvP model integrating all components."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.post_encoder = PostEncoder(args)
        self.multi_view_moe = MultiViewMoE(args)
        self.ucr = UserConsistencyRegularization(args)
        # self.lambda_ucr = args.lambda_ucr
        self.pad_token = args.pad_token
        # Loss function
        self.detection_loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor) -> tuple[Any, Any, Any]:
        """
        Args:
            input_ids: Tokenized posts [batch_size, max_posts, max_tokens]
            attention_mask: Attention mask [batch_size, max_posts, max_tokens]
            labels: Ground truth MBTI traits [batch_size, 4]
        Returns:
            logits: Personality predictions [batch_size, 4]
            losses: Dictionary of losses
        """
        attention_mask = (input_ids != self.pad_token).float().to(input_ids.device)
        # Encode posts
        post_reps = self.post_encoder(input_ids, attention_mask)  # [batch_size, max_posts, hidden_dim]

        # Multi-view representation
        user_reps, gate_load = self.multi_view_moe(post_reps)  # [batch_size, hidden_dim]

        # Personality prediction with UCR
        logits, ucr_loss = self.ucr(user_reps)  # [batch_size, 4]

        return logits, ucr_loss, gate_load