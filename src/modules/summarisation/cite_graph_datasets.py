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
import jsonlines as jsonl
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from src.utils.summarisation.data_utils import generate_doc
from src.modules.summarisation.ref_bart_datasets import RefBartDataset, ABS_SEP

class CiteGraphDataset(RefBartDataset):
    def __init__(self, data_dir,
                 data_key:str, # train, val, test
                 max_neighbor_num: int,
                 abstract_max_len: int,
                 doc_max_len: int,
                 tokenizer):
        super().__init__(data_dir, data_key, max_neighbor_num, abstract_max_len, doc_max_len, tokenizer)

    def generate_graph_structs(self, paper_id):
        references = self.paper_dict[paper_id]['references']
        if len(references) > self.max_neighbor_num:
            references = random.sample(references, self.max_neighbor_num)

        sub_graph_dict = {paper_id: references}
        return sub_graph_dict

    def _k_hop_neighbor(self, paper_id, n_hop, max_neighbor):
        '''
        references = self.paper_dict[paper_id]['references']
        if len(references)>max_neighbor:
        references = random.sample(references,max_neighbor)
        '''
        sub_graph = [[] for _ in range(n_hop + 1)]
        level = 0
        visited = set()
        queue = deque()
        queue.append([paper_id, level])
        curr_node_num = 0
        while len(queue) != 0:
            paper_first = queue.popleft()
            paper_id_first, level_first = paper_first
            if level_first > n_hop:
                return sub_graph
            sub_graph[level_first].append(paper_id_first)
            curr_node_num += 1
            if curr_node_num > max_neighbor:
                return sub_graph
            visited.add(paper_id_first)
            for pid in self.paper_dict[paper_id_first]["references"]:
                if pid not in visited and pid in self.paper_dict:
                    queue.append([pid, level_first + 1])
                    visited.add(pid)
        return sub_graph

    def generate_dgl_graph(self, graph_struct, doc):
        nodeid2paperid = []
        edges = []
        for level in graph_struct:
            edges.append([len(nodeid2paperid), len(nodeid2paperid)])
            dest_node_id = len(nodeid2paperid)
            nodeid2paperid.append(level)
            for nbr in graph_struct[level]:
                src_node_id = len(nodeid2paperid)
                nodeid2paperid.append(nbr)
                edges.append([src_node_id, dest_node_id])

        abs_list = [self.paper_dict[paper_id]['abstract'] for paper_id in nodeid2paperid]
        max_len = max(list(map(len, abs_list)))
        # remove golden abs
        abs_list[0] = " ".join([self.tokenizer.pad_token] * max_len)

        # abstract_input = self.tokenizer(abs_list, add_special_tokens=False, truncation=True,
        #                                 padding="max_length",
        #                                 max_length=self.abstract_max_len,
        #                                 return_tensors="pt")
        # abstract_input['input_ids'][0] = torch.ones_like(abstract_input['input_ids'][0]) * self.tokenizer.pad_token_id
        # abstract_input['input_ids'] = torch.cat((torch.ones(
        #     (abstract_input['input_ids'].shape[0], 1)) * self.tokenizer.cls_token_id, abstract_input['input_ids']),
        #                                         dim=1)
        #
        # doc_input = self.tokenizer([doc for _ in range(len(nodeid2paperid))], add_special_tokens=False,
        #                            truncation=True,
        #                            padding="max_length",
        #                            max_length=self.doc_max_len,
        #                            return_tensors="pt")
        # graph_input_ids = torch.cat((abstract_input['input_ids'], doc_input['input_ids']), dim=1)
        # attention_mask = (graph_input_ids != self.tokenizer.pad_token_id)

        inputs = []
        for abs in abs_list:
            doc_trunk = self.limit_tokens(doc, tokenizer=self.tokenizer, limit=self.doc_max_len)
            abs_trunk = self.limit_tokens(abs, tokenizer=self.tokenizer,
                                          limit=self.abstract_max_len)
            inputs.append(f"{doc_trunk} {ABS_SEP} {abs_trunk}")

        batched_inputs = self.tokenizer(
            inputs,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=1023,
            return_tensors="pt",
        ).data
        graph_input_ids = batched_inputs["input_ids"]
        attention_mask = batched_inputs["attention_mask"]
        edges = torch.tensor(edges)
        g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(nodeid2paperid))
        g.ndata['h'] = graph_input_ids.long()
        g.ndata['attention_mask'] = attention_mask.long()
        return g

    def __getitem__(self, i):
        data = self.dataset[self.datakey].iloc[i]
        abstract = data['abstract']
        doc = generate_doc(data, doc_max_len=self.doc_max_len)
        graph_structure = self.generate_graph_structs(data['paper_id'])
        graph = self.generate_dgl_graph(graph_structure, doc)

        return {'graph': graph, 'target': abstract}

    def collate_fn(self, batch):
        graph = dgl.batch([s['graph'] for s in batch])  # (bsz,)

        target = self.tokenizer(
            [s['target'] for s in batch],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.abstract_max_len,
            return_tensors="pt",
        ).data
        batched_data = {
            'graph': graph,
            'tgt_ids': target['input_ids'],
            'tgt_masks': target['attention_mask']
        }
        return batched_data

