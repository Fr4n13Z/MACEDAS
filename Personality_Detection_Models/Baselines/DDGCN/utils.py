# -*- coding: utf-8 -*-
import argparse

import numpy as np
import torch
from torch import sigmoid
from torch.nn.parameter import Parameter
from torch.optim import Adam


class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3, train=True):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma  # l in [l, r]
        self.zeta = zeta  # r in [l, r]
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias
        self.training = train

    def forward(self, input_element, summarize_penalty=True):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0 - 1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            # penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = self.clip(s)

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        if self.training == False:
            penalty = hard_concrete
        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


class LagrangianOptimization:
    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    gradient_accumulation_steps = None
    update_counter = 0

    def __init__(self, original_optimizer, device, init_alpha=5, min_alpha=-2, max_alpha=30, alpha_optimizer_lr=1e-2,
                 gradient_accumulation_steps=None, max_grad_norm=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, dtype=torch.float, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer
        self.max_grad_norm = max_grad_norm

    def update(self, f, g, model):
        """
        L(x, lambda) = f(x) + lambda g(x)

        :param f_function:
        :param g_function:
        :return:
        """

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        if isinstance(model, torch.nn.DataParallel):
            loss = loss.mean()
        try:
            # with torch.autograd.detect_anomaly():
            loss.backward()
        except RuntimeError:
            import pdb;
            pdb.set_trace()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps > 1:
            if self.update_counter % self.gradient_accumulation_steps == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                # print(self.alpha, self.alpha.grad)
                self.optimizer_alpha.step()
                self.original_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()

            self.update_counter += 1
        else:
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()
            self.original_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        if self.alpha.item() < self.min_alpha:
            self.alpha.data = torch.full_like(self.alpha.data, self.min_alpha)
        elif self.alpha.item() > self.max_alpha:
            self.alpha.data = torch.full_like(self.alpha.data, self.max_alpha)

        return self.alpha.item()

    def zero_grad(self):
        self.original_optimizer.zero_grad()
        self.optimizer_alpha.zero_grad()


def get_optimizer(args: argparse.Namespace, model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    bert_params = list(map(id, model.embed_model.parameters()))
    gm_params_id = []
    for l in range(args.gnn_num_layers):
        for label_dim in range(args.num_labels):
            gm_params_id += list(map(id, model.deepGCN.dgcns[label_dim].A[l].parameters()))
    other_params = filter(lambda p: id(p) not in bert_params + gm_params_id, model.parameters())
    gm_params = filter(lambda p: id(p) in gm_params_id, model.parameters())
    optimizer_grouped_parameters = [
        {'params': other_params, 'lr': args.other_lr},
        {'params': model.embed_model.parameters(), 'lr': args.pretrained_model_lr},
        {'params': gm_params, 'lr': args.gm_lr}
    ]

    model_optimizer = Adam(optimizer_grouped_parameters,
                           eps=args.adam_epsilon)
    optimizer = LagrangianOptimization(model_optimizer, args.device,
                                       gradient_accumulation_steps=args.gradient_accumulation_steps,
                                       alpha_optimizer_lr=args.alpha_lr, max_grad_norm=args.max_grad_norm,
                                       max_alpha=args.max_alpha) if args.l0 else model_optimizer  # L0

    return optimizer
