#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Variable
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
import numpy as np, math
from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn import init

from transformers import BartPretrainedModel, BartConfig, BartModel, T5PreTrainedModel, T5Config, T5Model, BertModel
from transformers.models.bart.modeling_bart import shift_tokens_right

class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class BART4ERC(BartPretrainedModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        self.model = BartModel(config)
        self.init_weights()
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                            nn.Dropout(0.3),
                            nn.GELU(),
                            nn.Linear(400, config.num_labels)).to(args.device)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)# how many sequences contains in each dialogue session.

        outputs_cls = self.model(input_ids=input_ids[context_mask, :],
                            attention_mask=attention_mask[context_mask, :])
        hidden_states_cls = outputs_cls.last_hidden_state
        mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, hidden_states_cls.shape[-1]).bool()
        #hidden_states_dropout = hidden_states_cls.clone().detach()
        hidden_states = hidden_states_cls.masked_fill(~mask_for_fill, -1e8)
        #hidden_states_dropout = hidden_states_dropout.masked_fill(~mask_for_fill, -1e8)
        cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling
        #cls_tokens_dropout, _ = torch.max(hidden_states_dropout, dim=1)

        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            cls_tokens = torch.cat([cls_tokens[:index4insert], fullzeropad4insert, cls_tokens[index4insert:]],
                                   dim=0)
            #cls_tokens_dropout = torch.cat([cls_tokens_dropout[:index4insert], fullzeropad4insert, cls_tokens_dropout[index4insert:]], dim=0)
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        logits = self.ffn(cls_tokens)
        if labels is not None:
            loss = self.loss_fct(logits[context_mask, :], labels[context_mask])
            return Seq2SeqLMOutput(
                loss=loss,
                logits=None,
                last_hidden_states=None
            )
        else:# Evluate
            return Seq2SeqLMOutput(
                loss=None,
                logits=logits[context_mask, :],
                last_hidden_states=cls_tokens[context_mask, :]
            )


class BART4ERCG(BartPretrainedModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                                 nn.Dropout(0.3),
                                 nn.GELU(),
                                 nn.Linear(400, config.num_labels)).to(args.device)
        self.loss_fct = CrossEntropyLoss()
        self.proj1 = nn.Linear(2*config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, his_ids, his_attention_mask, next_input_ids, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)  # how many sequences contains in each dialogue session.

        # history memory
        output_his = self.model(his_ids[context_mask, :], attention_mask=his_attention_mask[context_mask, :])
        hidden_states_memory = output_his.last_hidden_state
        his_mask_fill = his_attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, hidden_states_memory.shape[-1]).bool()
        hidden_states_mem = hidden_states_memory.masked_fill(~his_mask_fill, -1e8)#bs seql dim

        # cls task
        outputs_cls = self.model(input_ids=input_ids[context_mask, :], attention_mask=attention_mask[context_mask, :])
        hidden_states_cls = outputs_cls.last_hidden_state

        mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, hidden_states_cls.shape[-1]).bool()
        # hidden_states_dropout = hidden_states_cls.clone().detach()
        hidden_states = hidden_states_cls.masked_fill(~mask_for_fill, -1e8)
        # hidden_states_dropout = hidden_states_dropout.masked_fill(~mask_for_fill, -1e8)

        #cancat
        hidden_states = self.proj1(torch.concat([hidden_states, hidden_states_mem], dim=-1))

        cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling
        # cls_tokens_dropout, _ = torch.max(hidden_states_dropout, dim=1)

        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            cls_tokens = torch.cat([cls_tokens[:index4insert], fullzeropad4insert, cls_tokens[index4insert:]], dim=0)
            # cls_tokens_dropout = torch.cat([cls_tokens_dropout[:index4insert], fullzeropad4insert, cls_tokens_dropout[index4insert:]], dim=0)
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        cls_logits = self.ffn(cls_tokens)

        # generation task
        decoder_input_ids = shift_tokens_right(
            next_input_ids[context_mask, :], self.config.pad_token_id, self.config.decoder_start_token_id
        )
        outputs_gen = self.model(input_ids=input_ids[context_mask, :],
                                 attention_mask=attention_mask[context_mask, :],
                                 decoder_input_ids=decoder_input_ids)
        hidden_states_gen = outputs_gen.last_hidden_state
        gen_logits = self.lm_head(hidden_states_gen) + self.final_logits_bias

        gen_loss = self.loss_fct(gen_logits.view(-1, self.config.vocab_size),
                                 next_input_ids[context_mask, :].view(-1))
        cls_loss = self.loss_fct(cls_logits[context_mask, :], labels[context_mask])
        loss = 0.5 * cls_loss + 0.5 * gen_loss
        return Seq2SeqLMOutput(
            loss=loss,
            logits=cls_logits[context_mask, :],
            last_hidden_states=cls_tokens[context_mask, :]
        )


class BART4ERCGM(BartPretrainedModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                                 nn.Dropout(0.3),
                                 nn.GELU(),
                                 nn.Linear(400, config.num_labels)).to(args.device)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, next_input_ids, attention_mask, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)  # how many sequences contains in each dialogue session.

        # cls task
        outputs_cls = self.model(input_ids=input_ids[context_mask, :],
                                 attention_mask=attention_mask[context_mask, :])
        hidden_states_cls = outputs_cls.last_hidden_state

        mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, hidden_states_cls.shape[-1]).bool()
        # hidden_states_dropout = hidden_states_cls.clone().detach()
        hidden_states = hidden_states_cls.masked_fill(~mask_for_fill, -1e8)
        # hidden_states_dropout = hidden_states_dropout.masked_fill(~mask_for_fill, -1e8)
        cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling
        # cls_tokens_dropout, _ = torch.max(hidden_states_dropout, dim=1)

        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            cls_tokens = torch.cat([cls_tokens[:index4insert], fullzeropad4insert, cls_tokens[index4insert:]],
                                   dim=0)
            # cls_tokens_dropout = torch.cat([cls_tokens_dropout[:index4insert], fullzeropad4insert, cls_tokens_dropout[index4insert:]], dim=0)
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        cls_logits = self.ffn(cls_tokens)

        # generation task
        decoder_input_ids = shift_tokens_right(
            next_input_ids[context_mask, :], self.config.pad_token_id, self.config.decoder_start_token_id
        )
        outputs_gen = self.model(input_ids=input_ids[context_mask, :],
                                 attention_mask=attention_mask[context_mask, :],
                                 decoder_input_ids=decoder_input_ids)
        hidden_states_gen = outputs_gen.last_hidden_state
        gen_logits = self.lm_head(hidden_states_gen) + self.final_logits_bias

        # evaluate
        if labels == None:
            return Seq2SeqLMOutput(
                loss=None,
                logits=cls_logits[context_mask, :],
                last_hidden_states=cls_tokens[context_mask, :]
            )
        # train
        else:
            gen_loss = self.loss_fct(gen_logits.view(-1, self.config.vocab_size),
                                     next_input_ids[context_mask, :].view(-1))
            cls_loss = self.loss_fct(cls_logits[context_mask, :], labels[context_mask])
            loss = 0.5 * cls_loss + 0.5 * gen_loss
            return Seq2SeqLMOutput(
                loss=loss,
                logits=None,
                last_hidden_states=None
            )


class T54ERC(T5PreTrainedModel):
    def __init__(self, config: T5Config, args):
        super().__init__(config)
        self.model = T5Model(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()

        self.gru = nn.GRU(input_size=768, hidden_size=768, dropout=0.5, num_layers=2, bidirectional=True, bias=False)
        self.proj1 = nn.Linear(in_features=768*3, out_features=768, bias=False)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                                 nn.Dropout(0.3),
                                 nn.GELU(),
                                 nn.Linear(400, config.num_labels)).to(args.device)
        self.loss_fct = CrossEntropyLoss()
        self.args = args

    def forward(self, input_ids, attention_mask, his_ids, his_attention_mask, next_input_ids, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)  # how many sequences contains in each dialogue session.

        decoder_input_ids = shift_tokens_right(next_input_ids[context_mask, :], self.config.pad_token_id,
                                               self.config.decoder_start_token_id)
        LM_outputs = self.model(input_ids=input_ids[context_mask, :],
                                attention_mask=attention_mask[context_mask, :],
                                decoder_input_ids=decoder_input_ids)
        LM_hidden_states = LM_outputs.last_hidden_state
        LM_mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1,
                                                                                LM_hidden_states.shape[-1]).bool()
        LM_hidden_states = LM_hidden_states.masked_fill(~LM_mask_for_fill, -1e8)
        gen_logits = self.lm_head(LM_hidden_states) + self.final_logits_bias
        l1 = self.loss_fct(gen_logits.view(-1, self.config.vocab_size), next_input_ids[context_mask, :].view(-1))

        UL_hidden_states = LM_hidden_states
        UL_hidden_states = UL_hidden_states.masked_fill(~LM_mask_for_fill, -1e8)
        UL_cls_tokens, _ = torch.max(UL_hidden_states, dim=1)
        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=UL_cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            UL_cls_tokens = torch.cat(
                [UL_cls_tokens[:index4insert], fullzeropad4insert, UL_cls_tokens[index4insert:]], dim=0)
        UL_cls_tokens = UL_cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        UL_logits = self.ffn(UL_cls_tokens)
        l2 = self.loss_fct(UL_logits[context_mask, :], labels[context_mask])

        # Memory History
        history_mask = torch.sum(his_attention_mask, dim=-1).gt(0)
        history_decoder_input_ids = shift_tokens_right(his_ids[history_mask, :], self.config.pad_token_id,
                                                       self.config.decoder_start_token_id)
        mem_hidden_state = self.model(input_ids=his_ids[history_mask, :],
                                      attention_mask=his_attention_mask[history_mask, :],
                                      decoder_input_ids=history_decoder_input_ids).last_hidden_state
        CLS_hidden_states, _ = self.gru(UL_hidden_states)
        concat_hidden_states = torch.cat([mem_hidden_state, CLS_hidden_states], dim=-1)
        CLS_tokens, _ = torch.max(self.proj1(concat_hidden_states), dim=1)
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=CLS_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            CLS_tokens = torch.cat([CLS_tokens[:index4insert], fullzeropad4insert, CLS_tokens[index4insert:]],
                                   dim=0)
        CLS_tokens = CLS_tokens.view(batch_size, max_seq_len_ex, we_dim)
        CLS_logits = self.ffn(CLS_tokens)
        l3 = self.loss_fct(CLS_logits[context_mask, :], labels[context_mask])

        if labels is not None:
            loss = l1 + l2 + l3
            return Seq2SeqLMOutput(
                loss=loss,
                logits=None,
                last_hidden_states=None
            )
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=CLS_logits[context_mask, :],
                last_hidden_states=CLS_tokens[context_mask, :]
            )


class BERT4ERC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = BertModel.from_pretrained(args.model)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.config.vocab_size)))
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        self.model.config.num_labels = 6
        #self.init_weights()

        self.gru = nn.GRU(input_size=self.model.config.hidden_size, hidden_size=self.model.config.hidden_size, dropout=0.5, bidirectional=True, num_layers=2,bias=False, batch_first=True)
        self.proj1 = nn.Linear(in_features=self.model.config.hidden_size*3, out_features=self.model.config.hidden_size, bias=False)
        self.ffn = nn.Sequential(nn.Linear(self.model.config.hidden_size, 500),

                                 nn.GELU(),
                                 nn.Linear(500, self.model.config.num_labels)).to(args.device)
        self.context_memory = ScaledDotProductAttention(d_model=768, d_k=768, d_v=768, h=8)
        self.loss_fct = CrossEntropyLoss()
        self.args = args

    def forward(self, input_ids, attention_mask, his_ids, his_attention_mask, next_input_ids, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)  # how many sequences contains in each dialogue session.

        LM_outputs = self.model(input_ids=input_ids[context_mask, :],
                                attention_mask=attention_mask[context_mask, :])
        LM_hidden_states = LM_outputs.last_hidden_state
        LM_mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, LM_hidden_states.shape[-1]).bool()
        LM_hidden_states = LM_hidden_states.masked_fill(~LM_mask_for_fill, -1e8)
        gen_logits = self.lm_head(LM_hidden_states) + self.final_logits_bias
        #gen_logits = self.lm_head(LM_hidden_states)
        l1 = self.loss_fct(gen_logits.view(-1, self.model.config.vocab_size), next_input_ids[context_mask, :].view(-1))

        UL_hidden_states = LM_hidden_states
        UL_hidden_states = UL_hidden_states.masked_fill(~LM_mask_for_fill, -1e8)
        UL_cls_tokens, _ = torch.max(UL_hidden_states, dim=1)
        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim], device=UL_cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            UL_cls_tokens = torch.cat(
                [UL_cls_tokens[:index4insert], fullzeropad4insert, UL_cls_tokens[index4insert:]], dim=0)
        UL_cls_tokens = UL_cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        UL_logits = self.ffn(UL_cls_tokens)
        l2 = self.loss_fct(UL_logits[context_mask, :], labels[context_mask])

        # Memory History
        his_mask = torch.sum(his_attention_mask, dim=-1).gt(0)

        mem_hidden_state = self.model(input_ids=his_ids[context_mask, :],
                                      attention_mask=his_attention_mask[context_mask, :]).last_hidden_state

        outputs_MEM = self.context_memory(mem_hidden_state, UL_hidden_states, UL_hidden_states)
        CLS_hidden_states, _ = self.gru(outputs_MEM)
        concat_hidden_states = torch.cat([mem_hidden_state, CLS_hidden_states], dim=-1)
        CLS_tokens, _ = torch.max(self.proj1(concat_hidden_states), dim=1)
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=CLS_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            CLS_tokens = torch.cat([CLS_tokens[:index4insert], fullzeropad4insert, CLS_tokens[index4insert:]],
                                   dim=0)
        CLS_tokens = CLS_tokens.view(batch_size, max_seq_len_ex, we_dim)
        CLS_logits = self.ffn(CLS_tokens)
        l3 = self.loss_fct(CLS_logits[context_mask, :], labels[context_mask])

        #loss = (self.args.alpha * l1) + (self.args.beta * l2) + (1 - self.args.alpha - self.args.beta) * l3
        loss = (self.args.beta * l2) + (1 - self.args.beta) * l3
        return Seq2SeqLMOutput(
            loss=loss,
            logits=CLS_logits,
            last_hidden_states=None
        )