"""
@Desc:
@Reference:
"""
import os
from collections import deque
import dgl
import json
import torch
from typing import List, Dict
from pathlib import Path
import random

from src.modules.summarisation.datasets import BartDataset, ABS_SEP, PAD_IDX
from src.utils.summarisation.data_utils import generate_doc

from transformers import BartTokenizer

# ==================================
class RefBartDataset(BartDataset):
    def __init__(self, data_dir, data_key: str, max_neighbor_num: int, abstract_max_len: int, doc_max_len: int,
                 tokenizer):
        super().__init__(data_dir, data_key, max_neighbor_num, abstract_max_len, doc_max_len, tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                                               [ABS_SEP],
                                           })

    def limit_tokens(self, text, tokenizer: BartTokenizer, limit: int):
        tokens = tokenizer.tokenize(text)[:limit]
        cutted_text = tokenizer.convert_tokens_to_string(tokens)
        return cutted_text

    def get_refs(self, paper_id):
        references = self.paper_dict[paper_id]['references']
        if len(references) > self.max_neighbor_num:
            references = random.sample(references, self.max_neighbor_num)

        ref_list = [self.paper_dict[one]['abstract'] for one in references]
        return ref_list

    def __len__(self):
        return len(self.dataset[self.datakey])

    def __getitem__(self, i):
        data = self.dataset[self.datakey].iloc[i]
        abstract = data['abstract']
        doc = generate_doc(data, doc_max_len=self.doc_max_len)
        references = self.get_refs(data['paper_id'])
        return {'doc': doc, 'ref_list': references, 'target': abstract, 'ref_len': len(references)}

    def collate_fn(self, batch):
        inputs = []
        item_nums = []
        for one in batch:
            if one['ref_len'] == 0:
                item_nums.append(1)
                inputs.append(f"{one['doc']} {ABS_SEP} ")
            else:
                for ref in one['ref_list']:
                    # inputs.append(f"{one['doc'][:self.doc_max_len]} {ABS_SEP} {ref[:self.abstract_max_len]}")
                    doc_trunk = self.limit_tokens(one['doc'], tokenizer=self.tokenizer, limit=self.doc_max_len)
                    abs_trunk = self.limit_tokens(ref, tokenizer=self.tokenizer,
                                                  limit=self.abstract_max_len)
                    inputs.append(f"{doc_trunk} {ABS_SEP} {abs_trunk}")
                item_nums.append(one['ref_len'])

        batched_data = self.tokenizer(
            inputs,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.doc_max_len + self.abstract_max_len + 2,
            return_tensors="pt",
        ).data

        target = self.tokenizer(
            [x["target"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.abstract_max_len,
            return_tensors="pt",
        ).data
        batched_data["target"] = target
        batched_data["abs"] = [s['target'] for s in batch]
        batched_data["item_nums"] = item_nums
        return batched_data

class OneRefBartDataset(RefBartDataset):
    def __init__(self, data_dir, data_key: str, max_neighbor_num: int, abstract_max_len: int, doc_max_len: int,
                 tokenizer):
        super().__init__(data_dir, data_key, max_neighbor_num, abstract_max_len, doc_max_len, tokenizer)

    def collate_fn(self, batch):
        inputs = []
        for one in batch:
            if one['ref_len'] == 0:
                inputs.append(f"{one['doc']} {ABS_SEP} ")
            else:
                # inputs.append(f"{one['doc'][:self.doc_max_len]} {ABS_SEP} {one['ref_list'][0][:self.abstract_max_len]}")

                doc_trunk = self.limit_tokens(one['doc'], tokenizer=self.tokenizer, limit=self.doc_max_len)
                abs_trunk = self.limit_tokens(one['ref_list'][0], tokenizer=self.tokenizer, limit=self.abstract_max_len)
                inputs.append(f"{doc_trunk} {ABS_SEP} {abs_trunk}")
        doc = self.tokenizer(
            inputs,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.doc_max_len + self.abstract_max_len + 2,
            return_tensors="pt",
        ).data

        target = self.tokenizer(
            [x["target"] for x in batch],
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