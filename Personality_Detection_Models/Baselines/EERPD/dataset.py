import argparse
from typing import List

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EERPDDataset(Dataset):
    def __init__(self, args: argparse.Namespace, text_list: List[List[str]], labels: List[List]):
        self.args = args
        self.text_list = text_list
        self.labels = labels
        self.input_samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.ptm_path, local_files_only=True)

    def __len__(self):
        return len(self.text_list)


    def __getitem__(self, idx):
        return self.text_list[idx], self.labels[idx]