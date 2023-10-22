"""
@Desc:
@Reference:
"""
import torch
import logging

from transformers.models.pegasus import modeling_pegasus
from transformers.models.pegasus.modeling_pegasus import PegasusForConditionalGeneration, PegasusConfig
from transformers import PegasusTokenizer

from src.modules.summarisation.ref_pegasus_modules import PegasusForRefSumCG
from src.modules.summarisation.ref_bart_datasets import (
    RefBartDataset,
)

logger = logging.getLogger(__name__)

from src.models.summarisation.ref_bart import SumRefBart


class SumRefPegasus(SumRefBart):
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
        self.model = self._load_model(self.hparams.model_name_or_path, PegasusForRefSumCG, self.config)
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