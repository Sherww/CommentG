from typing import List, Optional
import pytorch_lightning as pl
import torch
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ..utils import PadFunction
import copy

from torchmetrics import MeanMetric


class BartForMaskedLM(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        config = copy.deepcopy(config)
        super().__init__(config)

        self.d_model = 768 # 1024

        self.transformer = transformers.BartModel.from_pretrained('facebook/bart-base')
        self.lm_head = torch.nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

    def forward(self,
            input_ids: torch.LongTensor = None,
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
            return_dict: Optional[bool] = None
        ):

        # label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            decoder_hidden_states=transformer_outputs.decoder_hidden_states,
            decoder_attentions=transformer_outputs.decoder_attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            encoder_last_hidden_state=transformer_outputs.encoder_last_hidden_state,
            encoder_hidden_states=transformer_outputs.encoder_hidden_states,
            encoder_attentions=transformer_outputs.encoder_attentions,
        )

class BartForMaskedLMModel(pl.LightningModule):
    def __init__(self, bos_token_id: int, eos_token_id: int, pad_token_id: int, vocab_size: int):
        super().__init__()

        self.learning_rate = 1e-5

        config = PretrainedConfig()
        config.d_model = 768
        config.max_length = 1024
        config.bos_token_id = bos_token_id
        config.eos_token_id = eos_token_id
        config.pad_token_id = pad_token_id
        config.vocab_size = vocab_size
        config.model_type = "bart"

        self.model = BartForMaskedLM(config)
        # metrics
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
    
    def forward(self, source_tokens, target_tokens):
        inputs, labels = source_tokens, target_tokens

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=label_ids,
            decoder_attention_mask=label_mask,
            return_dict=True,
        )
        
        lm_logits = outputs.logits

        return lm_logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), shift_label_ids.view(-1))

        self.log('train_loss_step', loss)
        self.train_loss.update(loss.item())
        return loss
    
    def training_epoch_end(self, outputs):
        self.log("train_loss_epoch", self.train_loss.compute())
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size),
                        shift_label_ids.view(-1))
        self.log('valid_loss_step', loss)
        self.valid_loss.update(loss.item())
    
    def validation_epoch_end(self, outputs):
        self.log("valid_loss_epoch", self.valid_loss.compute())
        self.valid_loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
