"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer


from src.modules.summarisation.ref_bart_datasets import (
    OneRefBartDataset,
)

from src.models.summarisation.mybart import SumBart


logger = logging.getLogger(__name__)


class OneRefBart(SumBart):
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
        self.model = self._load_model(self.hparams.model_name_or_path, BartForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = OneRefBartDataset
