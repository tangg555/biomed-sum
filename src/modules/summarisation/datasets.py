"""
@Desc:
@Reference:
"""
import os
from collections import deque
import dgl
import json
import torch
import random
import jsonlines as jsonl
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

from src.utils.file_utils import load_json
from src.utils.summarisation.data_utils import generate_doc

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
SEP = '[SEP]'
ABS_SEP = '[SEP]'
INDEX = 0
DEFAULT_VOCAB = ['[PAD]', '[UNK]', '[MASK]', '[SEP]', '[CLS]']
PAD_IDX, UNK_IDX, MASK_IDX, SEP_IDX, CLS_IDX = 0, 100, 103, 102, 101

DATA_KEYS = ['train', 'test', 'val']
class BartDataset(Dataset):
    def __init__(self, data_dir,
                 data_key:str, # train, val, test
                 max_neighbor_num: int,
                 abstract_max_len: int,
                 doc_max_len: int,
                 tokenizer):
        self.data_dir = data_dir
        self.max_neighbor_num = max_neighbor_num
        self.abstract_max_len = abstract_max_len
        self.doc_max_len = doc_max_len
        self.dataset, self.paper_dict = self.load_data(self.data_dir)
        self.tokenizer = tokenizer
        self.datakey = data_key

    def load_data(self, data_dir):
        dataset = {}
        # paper_dict = {}
        for key in DATA_KEYS:
            f = jsonl.open(os.path.join(data_dir, key + '.jsonl'))
            dataset[key] = []
            for d in f:
                dataset[key].append(d)
                # paper_dict[d['paper_id']] = d
            dataset[key] = pd.DataFrame(dataset[key])

        paper_dict_path = Path(data_dir).joinpath("paper_dict.json")
        print(f"load paper_dict from {paper_dict_path}")
        paper_dict = load_json(paper_dict_path)
        return dataset, paper_dict

    def __len__(self):
        return len(self.dataset[self.datakey])

    def __getitem__(self, i):
        data = self.dataset[self.datakey].iloc[i]
        doc = generate_doc(data, doc_max_len=self.doc_max_len)
        abstract = data['abstract']

        return {'doc': doc, 'target': abstract}

    def collate_fn(self, batch):
        doc = self.tokenizer([s['doc'] for s in batch],
                             add_special_tokens=False,
                             truncation=True,
                             padding="longest",
                             max_length=self.doc_max_len, return_tensors="pt")

        target = self.tokenizer(
            [s['target'] for s in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.abstract_max_len,
            return_tensors="pt",
        ).data
        abs = [s['target'] for s in batch]
        batched_data = {
            'doc': doc,
            'target': target,
            'abs': abs
        }
        return batched_data