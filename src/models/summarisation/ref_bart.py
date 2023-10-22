"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartConfig
from transformers import BartTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.modules.summarisation.ref_bart_datasets import (
    RefBartDataset
)
from src.modules.summarisation.ref_bart_modules import BartForRefSumCG
from src.utils.summarisation import model_utils
from src.models.lightning_base import BaseTransformer
from src.models.summarisation.mybart import SumBart

logger = logging.getLogger(__name__)


class SumRefBart(SumBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, BartForRefSumCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = RefBartDataset

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch: dict):
        src_ids, src_mask = batch['input_ids'], batch["attention_mask"]
        tgt_ids = batch["target"]["input_ids"]
        item_nums = batch['item_nums']
        extra_params = {}
        if self.hparams.use_r_drop:
            extra_params["use_r_drop"] = self.hparams.use_r_drop
            extra_params["r_drop_alpha"] = self.hparams.r_drop_alpha

        outputs = self(src_ids, attention_mask=src_mask, labels=tgt_ids, use_cache=False,
                       output_attentions=True, output_hidden_states=True, item_nums=item_nums, **extra_params)

        loss = outputs["loss"]
        return loss

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        src_ids, src_mask = batch['input_ids'], batch["attention_mask"]
        item_nums = batch['item_nums']
        encoder_outputs = self.model.get_encoder()(input_ids=src_ids, attention_mask=src_mask, item_nums=item_nums)
        # transformers.generation_utils
        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty
        generated_ids = self.model.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.hparams.max_target_length,
            min_length=130,
            top_p=self.top_p if self.use_top_p else None,
            item_nums=item_nums,
            **extra_params
        )
        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(batch["target"]["input_ids"])
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
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        logs["batch_size"] = batch['input_ids'].shape[0]
        return {"loss": loss, "log": logs}

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        generative_metrics = {
            name: np.array([x[name] for x in outputs]).mean() for name in self.metric_names
        }
        metric_val = (
            torch.tensor(generative_metrics[self.val_metric])
        )
        val_metrics = {f"{prefix}_{k}": x for k, x in generative_metrics.items()}
        val_metrics["step_count"] = float(self.step_count)
        self.current_val_metrics = val_metrics
        self.metrics[prefix].append(val_metrics)  # callback writes this to self.metrics_save_path.
        print(f"Evaluation result: {val_metrics}")
        preds = model_utils.flatten_list([x["preds"] for x in outputs])
        tgts = model_utils.flatten_list([x["targets"] for x in outputs])
        self.log_dict(self.current_val_metrics)
        return {
            "log": val_metrics,
            "preds": preds,
            "tgts": tgts,
            f"{prefix}_loss": generative_metrics["loss"],
            f"{prefix}_{self.val_metric}": metric_val,
        }

    def get_dataset(self, data_key: str = "train") -> RefBartDataset:
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

