mport argparse
import copy
from typing import List, Dict, Any

import liwc
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import re
import emoji
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from transformers import BertTokenizer


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'word2word_index':
            return self.word_nodes.size(0)
        if key == 'post2word_index':
            return self.post_ids.size(0) + self.word_nodes.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class DENDataset(Dataset):
    def __init__(self, args: argparse.Namespace, text_list: List[List[str]], labels: Dict[int, List[int]]):
        """
        Initialize the DENDataset.

        Args:
            args (argparse.Namespace): Command line arguments.
            text_list (List[List[str]]): List of user posts, where each post is a list of strings.
            labels (Dict[int, List[int]]): Dictionary mapping user indices to their corresponding labels.
        """
        self.args = args
        self.texts = text_list
        self.labels = labels
        self.parse_func, category_names = liwc.load_token_parser(args.liwc_path)
        self.datas = []
        self.post_tokenizer = BertTokenizer.from_pretrained(args.ptm_path, local_files_only=True)
        self.glove_model = KeyedVectors.load_word2vec_format(args.glove_path, binary=False)
        self.vocab = set(self.glove_model.key_to_index.keys())

        # Preprocess and build graph for each user's posts
        self._preprocess_data()

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """
        Preprocess the input text by decoding emojis, removing punctuation, converting to lowercase,
        and splitting into words.

        Args:
            text (str): Input text string.

        Returns:
            List[str]: List of preprocessed words.
        """
        # Decode emojis
        text = emoji.demojize(text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Split into words
        words = text.split()
        return words

    def build_graph(self, texts: List[str]):
        """
        Build the graph representation for the given texts.

        Args:
            texts (List[str]): List of text strings.

        Returns:
            Tuple: A tuple containing input IDs, attention mask, word nodes, word-to-word index,
                   post-to-word index, and word start index.
        """
        word_start_idx = len(texts)
        post_nodes = copy.deepcopy(texts)
        post_mask = [1] * word_start_idx
        post_mask += [0] * (self.args.max_posts - word_start_idx)
        all_words = set(word for text in texts for word in self.preprocess_text(text)) & self.vocab
        word_nodes = list(all_words)[:self.args.max_word_num]
        post2word_index = []
        word2word_index = []
        word_mask = [1] * len(word_nodes)
        word_psycho_category_dict = {word: set(self.parse_func(word)) for word in word_nodes}

        for post_idx, text in enumerate(texts):
            words = self.preprocess_text(text)
            for word in words:
                if word in word_nodes:
                    word_idx = word_nodes.index(word) + word_start_idx
                    post2word_index.append([post_idx, word_idx])
                    post2word_index.append([word_idx, post_idx])

        for word1 in word_nodes:
            word1_idx = word_nodes.index(word1)
            for word2 in word_nodes:
                if word1 == word2:
                    continue
                word2_idx = word_nodes.index(word2)
                if len(word_psycho_category_dict[word1] & word_psycho_category_dict[word2]) > 0:
                    word2word_index.append([word1_idx, word2_idx])
                    word2word_index.append([word2_idx, word1_idx])

        word_nodes = torch.from_numpy(np.array([self.glove_model[word] for word in word_nodes]))
        if word_nodes.size(0) < self.args.max_word_num:
            padding = torch.zeros(self.args.max_word_num - word_nodes.size(0), self.args.glove_embed_dim)
            word_mask += [0] * (self.args.max_word_num - word_nodes.size(0))
            word_nodes = torch.cat([word_nodes, padding], dim=0)

        tokenized_dict = self.post_tokenizer(post_nodes,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=self.args.max_length,
                                             return_tensors='pt')
        return (
            tokenized_dict['input_ids'],
            post_mask,
            word_nodes,
            word2word_index,
            post2word_index,
            word_start_idx,
            word_mask
        )

    def _preprocess_data(self):
        """
        Preprocess all data and build graphs for each user's posts.
        """
        for user_id, user_posts in tqdm(enumerate(self.texts)):
            post_ids, post_mask, word_nodes, word2word_index, post2word_index, word_start_idx, word_mask = self.build_graph(
                user_posts)
            current_data = PairData(
                post_ids=post_ids,
                post_mask=torch.tensor(post_mask).long(),
                word_nodes=word_nodes,
                word_mask=torch.tensor(word_mask).long(),
                word2word_index=to_undirected(torch.tensor(word2word_index).t()),
                post2word_index=to_undirected(torch.tensor(post2word_index).t()),
                word_start_idx=word_start_idx,
                label=torch.tensor(self.labels[user_id]).float()
            )
            self.datas.append(current_data)

    def __len__(self) -> int:
        """
        Return the number of users in the dataset.

        Returns:
            int: Number of users.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> PairData:
        """
        Get the data for the specified user index.

        Args:
            idx (int): User index.

        Returns:
            PairData: Graph data for the specified user.
        """
        return self.datas[idx]

def main():
    # Example data
    text_list = [
        ["Hello world!", "This is a test.", "Another example."],
        ["Good morning!", "How are you?", "Have a great day!"],
        ["I love coding.", "Machine learning is fun.", "Deep learning is powerful."]
    ]

    labels = {
        0: [1, 0, 1],
        1: [0, 1, 0],
        2: [1, 1, 1]
    }

    # Argument parser
    parser = argparse.ArgumentParser(description='DENDataset Parameters')
    parser.add_argument('--liwc-path', type=str, required=True, help='Path to LIWC dictionary file')
    parser.add_argument('--ptm-path', type=str, required=True, help='Path to BERT pre-trained model')
    parser.add_argument('--glove-path', type=str, required=True, help='Path to GloVe vector file')
    parser.add_argument('--max-word-num', type=int, default=50, help='Maximum number of words per user')
    parser.add_argument('--max-seq-length', type=int, default=128, help='Maximum sequence length for BERT')
    parser.add_argument('--glove-embed-dim', type=int, default=50, help='Dimension of GloVe embeddings')

    args = parser.parse_args([
        '--liwc-path', 'path/to/liwc.dic',
        '--ptm-path', 'bert-base-uncased',
        '--glove-path', 'glove.6B.50d.txt'
    ])

    # Instantiate the dataset
    dataset = DENDataset(args, text_list, labels)

    # Test getting an item from the dataset
    print(dataset[0])


if __name__ == '__main__':
    main()