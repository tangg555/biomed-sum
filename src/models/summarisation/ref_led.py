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

from transformers.models.led import modeling_led
from transformers.models.led.modeling_led import LEDConfig
from transformers import LEDTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.modules.summarisation.ref_bart_datasets import (
    RefBartDataset
)
from src.modules.summarisation.ref_led_modules import LEDForRefSumCG
from src.utils.summarisation import model_utils
from src.models.lightning_base import BaseTransformer
from src.models.summarisation.ref_bart import SumRefBart

logger = logging.getLogger(__name__)


class SumRefLED(SumRefBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: LEDConfig = LEDConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: LEDTokenizer = LEDTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, LEDForRefSumCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = RefBartDataset

    def _step(self, batch: dict):
        src_ids, src_mask = batch['input_ids'], batch["attention_mask"]
        tgt_ids = batch["target"]["input_ids"]
        item_nums = batch['item_nums']

        outputs = self(src_ids, attention_mask=src_mask, labels=tgt_ids, use_cache=False,
                       output_attentions=True, output_hidden_states=True, item_nums=item_nums)

        loss = outputs["loss"]
        return loss

