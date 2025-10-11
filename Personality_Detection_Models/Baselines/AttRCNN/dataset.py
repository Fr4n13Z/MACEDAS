import gc
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel


class AttRCNNDataset(Dataset):
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

    def _preprocess_data(self, sentence_embed=False):
        encoder = BertModel.from_pretrained(self.args.bert_path, local_files_only=True).to(self.args.device)
        self.processed_data = []
        for posts, label in tqdm(zip(self.text_list, self.labels), desc="Processing data"):

            post_ids = []
            for post in posts[:self.args.max_posts]:
                encoded = self.tokenizer(
                    post,
                    max_length=self.args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                post_ids.append(encoded['input_ids'].squeeze(0))

            if len(post_ids) < self.args.max_posts:
                pad_context, pad_mask = self._create_padding_tokens(self.args.max_length)
                pad_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(pad_context))
                # Add padding posts
                num_pads = self.args.max_posts - len(post_ids)
                post_ids.extend([pad_ids] * num_pads)

            post_ids = torch.stack(post_ids)  # (max_posts, max_length)
            encoder.eval()
            with torch.no_grad():
                tokens = torch.tensor(post_ids).long().to(self.args.device)
                attn_mask = (tokens != self.pad_id).float()
                if sentence_embed:
                    cls_token = encoder(input_ids=tokens, attention_mask=attn_mask)[0][:, 0].detach().cpu()
                else:
                    cls_token = encoder(input_ids=tokens, attention_mask=attn_mask).last_hidden_state.detach().cpu()
                del tokens
                with torch.cuda.device(device=self.args.device):
                    torch.cuda.empty_cache()
            self.processed_data.append((cls_token, label))
        del encoder
        with torch.cuda.device(device=self.args.device):
            gc.collect()
            torch.cuda.empty_cache()

    def __getitem__(self, idx):
        return self.processed_data[idx]
