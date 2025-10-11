# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class BERTDataset(Dataset):
    def __init__(self, args, text_list: List[List[str]], labels: List[int]):
        self.categories = ['[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]',
                           '[DRIV]', '[RELA]', '[INFO]', '[WORK]', '[LEIS]',
                           '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        self.tokenizer.add_tokens(['<eos>'] + self.categories, special_tokens=True)
        self.pad_id = self.tokenizer.pad_token_id
        self.text_list = text_list
        self.labels = labels
        self._preprocess_data()

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _create_padding_tokens(max_length: int) -> Tuple[List[str], List[int]]:
        """Create padding tokens and corresponding attention mask."""
        context = ["[CLS]"] + ["[PAD]"] * (max_length - 2) + ["[SEP]"]
        attention_mask = [1] + [0] * (max_length - 2) + [1]
        return context, attention_mask

    def _preprocess_data(self):
        self.processed_data = []
        for posts, label in tqdm(zip(self.text_list, self.labels), desc="Processing data"):

            post_ids = []
            posts_mask = []
            for post in posts[:self.args.max_posts]:
                encoded = self.tokenizer(
                    post,
                    max_length=self.args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                post_ids.append(encoded['input_ids'].squeeze(0))
                posts_mask.append(1)

            if len(post_ids) < self.args.max_posts:
                pad_context, pad_mask = self._create_padding_tokens(self.args.max_length)
                pad_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(pad_context))
                # Add padding posts
                num_pads = self.args.max_posts - len(post_ids)
                post_ids.extend([pad_ids] * num_pads)
                posts_mask.extend([0] * num_pads)  # 0 indicates padding post

            post_ids = torch.stack(post_ids)  # (max_posts, max_length)
            posts_mask = torch.tensor(posts_mask)  # (max_posts,)

            self.processed_data.append((post_ids, posts_mask, label))

    def __getitem__(self, idx):
        return self.processed_data[idx]
