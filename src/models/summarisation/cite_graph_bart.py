"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
import dgl
import jsonlines
from tqdm import tqdm
import os

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.modules.summarisation.cite_graph_datasets import CiteGraphDataset
from src.modules.summarisation.cite_graph_bart_modules import CiteGNNBartForCG
from src.utils.summarisation import model_utils
from src.models.lightning_base import BaseTransformer
from src.models.summarisation import SumBart
from src.utils.file_utils import load_json, save_json

logger = logging.getLogger(__name__)


class CiteGraphBart(SumBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)
        # for the dataset
        self.paper_dict = self._load_paper_dict()
        print(f"paper_dict is loaded from {self.hparams.data_dir}, the size is {len(self.paper_dict)}")

    def _load_paper_dict(self):
        data_dir = Path(self.hparams.data_dir)

        dict_path = data_dir.joinpath("paper_dict.json")
        if dict_path.exists():
            return load_json(dict_path)

        covid_data_list = json.load(data_dir.joinpath("covid_data.json").open("r", encoding="utf-8"))
        covid_dict = {}
        for one in covid_data_list:
            covid_dict[one["paper_id"]] = one
        cleaned_covid_data_list = list(jsonlines.open(data_dir.joinpath("cleaned_covid_data.jsonl"), mode="r"))
        cleaned_paper_list = [one["paper_id"] for one in cleaned_covid_data_list]
        paper_dict = {}
        for file_name in os.listdir(data_dir):
            if file_name not in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                continue
            with jsonlines.open(data_dir.joinpath(file_name)) as fr:
                for json_dict in tqdm(list(fr), desc=f"make_paper_dict from {file_name}"):
                    paper_dict[json_dict['paper_id']] = json_dict
                    # reference
                    for ref_id in json_dict['references']:
                        target_obj = covid_dict[ref_id]
                        cleaned_refs = []
                        for ref_id_ in target_obj['references']:
                            if ref_id_ in cleaned_paper_list:
                                cleaned_refs.append(ref_id_)
                        paper_dict[ref_id] = cleaned_refs
        save_json(paper_dict, dict_path)
        print(f"create paper dict; size: {len(paper_dict)} ")
        return paper_dict


    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, CiteGNNBartForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = CiteGraphDataset
        # A decoder-only architecture is being used, please set `padding_side='left'` when initializing the tokenizer.
        self.tokenizer.padding_side = 'left'

    def _step(self, batch: dict):
        graph = batch['graph']
        tgt_ids = batch['tgt_ids']
        outputs = self(graph=graph,
                       labels=tgt_ids)

        loss = outputs["loss"]
        return loss

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        graph = batch['graph']
        tgt_ids = batch['tgt_ids']

        # transformers.generation_utils
        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty

        graphs = dgl.unbatch(graph)
        # [batch, seq_len]
        src_ids = torch.stack([g.ndata['h'][0] for g in graphs])
        generated_ids = self.model.generate(
            inputs=src_ids.long(),
            graph=graph,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.hparams.max_target_length,
            min_length=130,
            top_p=self.top_p,
            **extra_params
        )

        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(tgt_ids)
        loss = self._step(batch)

        base_metrics = {"loss": loss.item()}

        rouge_metrics: Dict = nlg_eval_utils.compute_metrics_for_summary(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def training_step(self, batch, batch_idx) -> Dict:
        outputs = self._step(batch)
        loss = outputs
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        logs["batch_size"] = batch['tgt_ids'].shape[0]
        return {"loss": loss, "log": logs}

    def get_dataset(self, data_key: str = "train") -> CiteGraphDataset:
        dataset = self.dataset_class(
            data_dir=self.hparams.data_dir,
            data_key=data_key,
            max_neighbor_num=self.hparams.max_neighbor_num,
            doc_max_len=self.hparams.doc_max_len,
            abstract_max_len=self.hparams.abstract_max_len,
            tokenizer=self.tokenizer,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def get_dataloader(self, data_key: str,  batch_size: int, shuffle: bool = False):
        dataset = self.get_dataset(data_key)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size,
                                           shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)
