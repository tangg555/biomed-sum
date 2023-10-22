import sys

import torch
import torch.nn as nn
import dgl
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.file_utils import ModelOutput
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    shift_tokens_right, BartModel, BaseModelOutput
from src.modules.summarisation.graph_attention import GATlayer


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class CiteGNNBartForCG(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.graph_model = GATlayer(config)
        self.model = CiteGNNBartModel(config)
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.pad_token_id = self.config.pad_token_id
        self.config.is_encoder_decoder = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, graph,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                decoder_head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[List[torch.FloatTensor]] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            graph=graph,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        graph=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        src_ids = graph.ndata['h'].long()
        attention_mask = graph.ndata['attention_mask']
        encoder_outputs = self.model.encoder(
            input_ids=src_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=True,
        )

        graph.ndata['h_features'] = encoder_outputs[0]
        graph = self.graph_model(graph)
        graphs = dgl.unbatch(graph)
        # [batch, seq_len, embed]
        encoder_outputs = BaseModelOutput(
            last_hidden_state=torch.stack([g.ndata['h_features'][0] for g in graphs])
        )
        encoder_attention_mask = torch.stack([g.ndata['attention_mask'][0] for g in graphs])
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "graph": graph,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

class CiteGNNBartModel(BartModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.graph_model = GATlayer(config)

        # self.config.is_encoder_decoder = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, graph,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                decoder_head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[List[torch.FloatTensor]] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs
                ):
        src_ids = graph.ndata['h'].long()
        attention_mask = graph.ndata['attention_mask']

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                "passed, `input_ids` cannot be `None`. Please pass either "
                "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=src_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            graph.ndata['h_features'] = encoder_outputs[0]
            graph = self.graph_model(graph)
            graphs = dgl.unbatch(graph)
            # [batch, seq_len, embed]
            encoder_hidden_states = torch.stack([g.ndata['h_features'][0] for g in graphs])
            encoder_attention_mask = torch.stack([g.ndata['attention_mask'][0] for g in graphs])
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            encoder_hidden_states = encoder_outputs[0]
            encoder_attention_mask = kwargs["encoder_attention_mask"] if "encoder_attention_mask" in kwargs else None
        else:
            encoder_hidden_states = encoder_outputs[0]
            encoder_attention_mask = kwargs["encoder_attention_mask"] if "encoder_attention_mask" in kwargs else None

        # for beam search ---------------------
        if encoder_hidden_states.shape[0] != decoder_input_ids.shape[0]:
            cat_emb = torch.cat([encoder_hidden_states] * int(decoder_input_ids.shape[0] / encoder_hidden_states.shape[0]), dim=1)
            encoder_hidden_states = cat_emb.view(decoder_input_ids.shape[0], encoder_hidden_states.shape[1], encoder_hidden_states.shape[2])

            if encoder_attention_mask:
                cat_mask = torch.cat([encoder_attention_mask] * int(decoder_input_ids.shape[0] / encoder_attention_mask.shape[0]), dim=1)
                encoder_attention_mask = cat_mask.view(decoder_input_ids.shape[0], encoder_attention_mask.shape[1])

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        graph=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        src_ids = graph.ndata['h'].long()
        attention_mask = graph.ndata['attention_mask']
        encoder_outputs = self.encoder(
            input_ids=src_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )

        graph.ndata['h_features'] = encoder_outputs[0]
        graph = self.graph_model(graph)
        graphs = dgl.unbatch(graph)
        # [batch, seq_len, embed]
        encoder_outputs = BaseModelOutput(
            last_hidden_state=torch.stack([g.ndata['h_features'][0] for g in graphs])
        )
        encoder_attention_mask = torch.stack([g.ndata['attention_mask'][0] for g in graphs])
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "graph": graph,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
