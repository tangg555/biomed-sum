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

from transformers.models.pegasus.modeling_pegasus import PegasusForConditionalGeneration, PegasusConfig
from transformers import PegasusTokenizer
from src.models.summarisation.mybart import SumBart
from src.modules.summarisation.datasets import BartDataset

logger = logging.getLogger(__name__)


class SumPegasus(SumBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: PegasusConfig = PegasusConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: PegasusTokenizer = PegasusTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, PegasusForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = BartDataset
