"""
@Desc:
@Reference:
@Notes:
"""
import logging
import copy
import torch.nn.functional as F

import torch
from torch import nn
from transformers.models.bart.modeling_bart import (
    BartModel, shift_tokens_right, Seq2SeqModelOutput, BartConfig, BartPretrainedModel, BartDecoder,
    CrossEntropyLoss, Seq2SeqLMOutput, BartAttention, BartEncoder, BartDecoderLayer, _expand_mask, BaseModelOutput,
    BartForConditionalGeneration, Optional, random
)

from src.modules.summarisation.ref_bart_modules import *

logger = logging.getLogger(__name__)


class AvgRefSumBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # =========== customized ============
        item_nums: list = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # aggregation ===============
        assert hidden_states.shape[0] == sum(item_nums)
        aggregated_encoder_last_hidden_state = []
        _slice=0
        for item_nums in item_nums:
            abs_hidden_states = hidden_states[_slice:_slice + item_nums] # [ref_len, seq_len, embed_size]
            attn_factor = 1 / item_nums  # [ref_len, seq_len, embed_size]
            abs_hidden_states = torch.sum(attn_factor * abs_hidden_states, dim=0)

            aggregated_encoder_last_hidden_state.append(abs_hidden_states)
            _slice += item_nums
        aggregated_encoder_last_hidden_state = torch.stack(aggregated_encoder_last_hidden_state)
        # =====================

        if not return_dict:
            return tuple(v for v in [aggregated_encoder_last_hidden_state] if v is not None)
        return BaseModelOutput(
            last_hidden_state=aggregated_encoder_last_hidden_state, hidden_states=None, attentions=None
        )



# replace BartModel
class AvgRefSumBartModel(RefSumBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.encoder = AvgRefSumBartEncoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

class AvgBartForRefSumCG(BartForRefSumCG):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = AvgRefSumBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

