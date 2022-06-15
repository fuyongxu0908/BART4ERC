#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from tqdm import tqdm, trange
import argparse
import logging

import sys
sys.path.append("../")
from train_MELD import set_seed, get_dataloaders, EarlyStopping, evaluate
from models import Seq2SeqLMOutput

from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, BertPreTrainedModel, BertConfig, BertModel

class bertERC(BertPreTrainedModel):
    def __init__(self, config: BertConfig, args):
        super().__init__(config)
        self.model = BertModel(config)
        self.init_weights()
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                                 nn.Dropout(0.3),
                                 nn.GELU(),
                                 nn.Linear(400, config.num_labels)).to(args.device)
        self.loss_fct = CrossEntropyLoss()
    def forward(self, input_ids, attention_mask, his_ids, his_attention_mask, next_input_ids, labels):
        context_mask = torch.sum(attention_mask, dim=-1).gt(0)
        batch_size, max_seq_len_ex, max_text_seq_len = input_ids.shape
        seqlens = torch.sum(context_mask, dim=-1)  # how many sequences contains in each dialogue session.

        outputs_cls = self.model(input_ids=input_ids[context_mask, :], attention_mask=attention_mask[context_mask, :])
        hidden_states_cls = outputs_cls.last_hidden_state

        mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, hidden_states_cls.shape[-1]).bool()
        # hidden_states_dropout = hidden_states_cls.clone().detach()
        hidden_states = hidden_states_cls.masked_fill(~mask_for_fill, -1e8)
        # hidden_states_dropout = hidden_states_dropout.masked_fill(~mask_for_fill, -1e8)

        cls_tokens, _ = torch.max(hidden_states, dim=1)
        we_dim = self.model.config.hidden_size
        for ibatch in range(batch_size):
            fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], we_dim],
                                             device=cls_tokens.device)  # max_seq_len_ex: num of max dialog turn
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            cls_tokens = torch.cat([cls_tokens[:index4insert], fullzeropad4insert, cls_tokens[index4insert:]], dim=0)
            # cls_tokens_dropout = torch.cat([cls_tokens_dropout[:index4insert], fullzeropad4insert, cls_tokens_dropout[index4insert:]], dim=0)
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        cls_logits = self.ffn(cls_tokens)
        cls_loss = self.loss_fct(cls_logits[context_mask, :], labels[context_mask])
        return Seq2SeqLMOutput(
            loss=cls_loss,
            logits=cls_logits[context_mask, :],
            last_hidden_states=cls_tokens[context_mask, :]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=False, default='MELD')
    parser.add_argument('--num_labels', type=int, required=False, default=7)
    parser.add_argument('--train_with_generation', type=int, default=1,
                        help="1: train with auxiliary generation task, 0: verse vice")
    parser.add_argument("--sd", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--learning_rate", default=1e-6)
    parser.add_argument("--adam_epsilon", default=1e-6)
    parser.add_argument("--warmup_ratio", default=0.1)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--temperature", default=1)
    parser.add_argument("--eval_batch_size", default=4)
    parser.add_argument("--train_batch_size", default=4)
    parser.add_argument("--logging_steps", default=50)
    parser.add_argument("--output_dir", default='./save')
    parser.add_argument("--patience", default=10)
    parser.add_argument("--alpha", default=0.4)
    parser.add_argument("--beta", default=0.3)
    parser.add_argument("--model", default='bert-base-uncased')
    parser.add_argument("--train_start", default=None)

    args = parser.parse_args()
    logging.info("The random seed is {}".format(args.sd))
    set_seed(sd=args.sd)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=None,
        use_fast=True)
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(tokenizer, args.device,
                                                                         dataset_name=args.task_name,
                                                                         train_batch_size=args.train_batch_size,
                                                                         eval_batch_size=args.eval_batch_size,
                                                                         max_seq_length=128, train_with_generation=1)
    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        cache_dir=None,
        revision=None,
        use_auth_token=None,
    )
    config.num_labels = 7
    config.use_cache = True

    model = bertERC.from_pretrained(
        args.model,
        config=config,
        args=args
    )
    model = model.to(args.device)

    best_score = 0
    steps_per_epoch = len(train_dataloader)

    num_train_steps = int(steps_per_epoch * args.epochs)
    t_total = num_train_steps

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio * t_total,
                                                num_training_steps=t_total)
    early_stopping = EarlyStopping(args.patience, verbose=True)
    global_step = 0

    if args.train_start == None:
        start_epoch = 0
    else:
        model_state = torch.load(args.train_start)
        start_epoch = model_state['epoch']
        model.load_state_dict(model_state['model'])
        optimizer.load_state_dict(model_state['optimizer'])
        scheduler.load_state_dict(model_state['scheduler'])

    for epoch in trange(start_epoch, int(args.epochs), desc="Epoch"):
        training_steps = 0
        model.zero_grad()
        for data in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            model.train()
            input_ids, attention_mask, labels, next_input_ids, speakers, his_ids, his_attention_mask = \
                data['input_ids'], data['attention_mask'], data['labels'], data['next_input_ids'], data['speakers'], \
                data['his_ids'], data['his_attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, his_ids=his_ids,
                            his_attention_mask=his_attention_mask, next_input_ids=next_input_ids, labels=labels)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, next_input_ids=next_input_ids, labels=labels)
            loss = outputs.loss
            #wandb.log({'Train_loss': loss})
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            training_steps += 1
            global_step += 1
            torch.cuda.empty_cache()

        results = evaluate(args, eval_dataloader, model, "evaluate")
        logging.info("val:{}".format(results))
        # logging.info("Val:{}".format(each_label_results))
        early_stopping(results['acc'], model, epoch, optimizer, scheduler, results)
        torch.cuda.empty_cache()

        results = evaluate(args, test_dataloader, model, "predict")
        logging.info("Test:{}".format(results))
        if early_stopping.early_stop:
            print("Early stopping")
            break
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
