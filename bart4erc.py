#!/usr/bin/python
# -*- coding:utf8 -*-
import argparse
import logging
import numpy as np
import json
import pickle
from typing import Optional, Tuple
from tqdm import tqdm, trange
import os
import sys
#import wandb
#wandb.init(project="bert4rec", entity="xfy")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import BartPretrainedModel, BartConfig, BartModel, BartTokenizer, BertConfig
from transformers.file_utils import ModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

from models import BART4ERC, BART4ERCG, T54ERC, BERT4ERC

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)



def set_seed(sd):
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    np.random.seed(sd)
    torch.backends.cudnn.deterministic = True


class ERCDataset(Dataset):

    def __init__(self, tokenizer, device, max_seq_length, split, dataset_name, train_with_generation):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)
        self.label_vocab = None
        self.speaker_vocab = None
        self.max_seq_length = max_seq_length
        self.train_with_generation = train_with_generation

    def __getitem__(self, idx):
        if self.train_with_generation:
            return self.data[idx]['utterances'], self.data[idx]['labels'], self.data[idx]['speakers'], \
                   self.data[idx]['next_sentences'], self.data[idx]['speakers'], self.data[idx]['history']
        else:
            return self.data[idx]['utterances'], self.data[idx]['labels'], self.data[idx]['speakers']

    def __len__(self):
        return self.len

    def get_bart_feature(self, sentence, tokenizer):
        inputs = tokenizer(sentence, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def read(self, dataset_name, split):
        with open('./data/%s/%s_data_generation.json' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
        # process dialogue
        dialogs = []
        for d in raw_data:
            history = []
            utterances = []
            labels = []
            speakers = []
            next_sentence = []
            for i, u in enumerate(d):
                # convert label from text to number
                label_id = self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -100
                utterances.append(u['text'])
                next_sentence.append(u['next_sentence'])
                labels.append(int(label_id))
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                if i == 0:
                    history.append("<pad>")
                    tmp = u['text']
                else:
                    history.append(history[-1] + " " + tmp)
                    tmp = u['text']

            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'next_sentences': next_sentence,
                'history': history
            })

        # random.shuffle(dialogs)
        return dialogs

    def collate_fn(self, datas):
        """

        :param datas:
        :return:
        inputs['input_ids'] (Batch_size, Utter_len, Sent_len)
        inputs['attention_mask'] (Batch_size, Utter_len, Sent_len)
        """

        inputs = {'input_ids': pad_sequence([self.get_bart_feature(data[0], self.tokenizer)['input_ids'] for data in datas], batch_first=True,
                                            padding_value=1),
                  'attention_mask': pad_sequence([self.get_bart_feature(data[0], self.tokenizer)['attention_mask'] for data in datas],
                                                 batch_first=True, padding_value=0)}

        inputs['his_ids'] = pad_sequence([self.get_bart_feature(data[5], self.tokenizer)['input_ids'] for data in datas], batch_first=True, padding_value=1)
        inputs['his_attention_mask'] = pad_sequence([self.get_bart_feature(data[5], self.tokenizer)['attention_mask'] for data in datas], batch_first=True, padding_value=1)


        inputs['labels'] = pad_sequence([torch.tensor(data[1], device=inputs['input_ids'].device) for data in datas], batch_first=True,
                                        padding_value=-100)
        inputs['speakers'] = pad_sequence([torch.tensor(data[2], device=inputs['input_ids'].device) for data in datas], batch_first=True,
                                          padding_value=-100)

        if self.train_with_generation:
            inputs['next_input_ids'] = pad_sequence(
                [self.get_bart_feature(data[3], self.tokenizer)['input_ids'] for data in datas], batch_first=True,
                padding_value=1)
            inputs['next_attention_mask'] = pad_sequence(
                [self.get_bart_feature(data[3], self.tokenizer)['attention_mask'] for data in datas],
                batch_first=True, padding_value=0)

        return inputs


def get_dataloaders(tokenizer, device, max_seq_length, dataset_name, train_batch_size=32, eval_batch_size=16, num_workers=0, pin_memory=False, train_with_generation=1):
    print('building datasets..')

    train_dataset = ERCDataset(tokenizer=tokenizer, device=device, max_seq_length=max_seq_length, split='train', dataset_name=dataset_name, train_with_generation=train_with_generation)
    dev_dataset = ERCDataset(tokenizer=tokenizer, device=device, max_seq_length=max_seq_length, split='dev', dataset_name=dataset_name, train_with_generation=train_with_generation)
    test_dataset = ERCDataset(tokenizer=tokenizer, device=device, max_seq_length=max_seq_length, split='test', dataset_name=dataset_name, train_with_generation=train_with_generation)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(dataset=dev_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              collate_fn=test_dataset.collate_fn)

    return train_loader, valid_loader, test_loader


def evaluate(args, eval_loader, model, eval_or_test):
    def compute_acc_for_categories(preds, labels):
        categories_count = {"label_%s" % i: 0 for i in range(6)}
        categories_right = {"label_%s" % i: 0 for i in range(6)}
        categories_acc = {}
        for pred, label in zip(preds, labels):
            categories_count["label_%s" % label] += 1
            if pred == label:
                categories_right["label_%s" % label] += 1
        for index, (key, value) in enumerate(categories_count.items()):
            categories_acc["label_%s" % index] = format(categories_right["label_%s" % index] / value, '.4f')
        print(categories_acc)
        return categories_acc

    def compute_metrics(preds_id, labels_id):
        results = {}

        # -------------- eval classification --------------
        accuracy = round(accuracy_score(labels_id, preds_id) * 100, 4)
        if args.task_name in ['MELD', 'EmoryNLP', 'IEMOCAP']:
            macro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(6)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(6)], average='micro')
        else:
            macro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(1, 6)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(1, 6)], average='micro')
        results['acc'] = accuracy
        results['macro_f1'] = round(macro_f1 * 100, 4)
        results['micro_f1'] = round(micro_f1 * 100, 4)

        return results

    results = {}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running %s *****" % eval_or_test)
    logger.info("  Num examples = %d", len(eval_loader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # eval_loss = 0.0

    all_preds, all_labels, loss = [], [], []

    for batch in tqdm(eval_loader, desc=eval_or_test):
        model.eval()
        with torch.no_grad():
            input_ids, attention_mask, labels, next_input_ids, speakers, his_ids, his_attention_mask = \
                batch['input_ids'], batch['attention_mask'], batch['labels'], batch['next_input_ids'], batch['speakers'], \
                batch['his_ids'], batch['his_attention_mask']
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, his_ids=his_ids, his_attention_mask=his_attention_mask, next_input_ids=next_input_ids, labels=labels)

            #labels = labels[labels.ne(-100)].cpu().numpy()

            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, next_input_ids=next_input_ids, labels=labels)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, his_ids=his_ids, his_attention_mask=his_attention_mask, next_input_ids=next_input_ids, labels=labels)

            labels = labels[labels.ne(-100)].cpu().numpy()
            preds = outputs.logits
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
            all_labels.append(labels)
            all_preds.append(preds)
            loss.append(outputs.loss.item())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    correct_num = np.sum(all_preds == all_labels)

    # eval_loss = eval_loss / nb_eval_steps
    result = compute_metrics(all_preds, all_labels)
    results.update(result)
    logger.info("***** %s results *****" % eval_or_test)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))
    logger.info("Correct / Total num = ({}/{})".format(correct_num, len(all_labels)))
    results.update({'loss': (sum(loss)/len(loss))})

    return results


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_acc = 0
        self.delta = delta

    def __call__(self, val_acc, model, epoch, optimizer, scheduler, results):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, epoch, optimizer, scheduler, results)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, epoch, optimizer, scheduler, results)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, epoch, optimizer, scheduler, results):
        '''Saves model when validation acc increased.'''
        if self.verbose:
            print(f'Validation acc increased ({self.val_loss_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        acc = results['acc']
        maf1 = results['macro_f1']
        mif1 = results['micro_f1']
        save_dir = "./save/bestres_{}_{}_{}.pth".format(acc, maf1, mif1)
        model_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch}

        torch.save(model_state, save_dir)
        self.val_acc_min = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=False, default='IEMOCAP')
    parser.add_argument('--num_labels', type=int, required=False, default=6)
    parser.add_argument('--train_with_generation', type=int, default=1, help="1: train with auxiliary generation task, 0: verse vice")
    parser.add_argument("--sd", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--learning_rate", default=1e-5)
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
    parser.add_argument("--model", default='facebook/bart-base', choices=['facebook/bart-base', 't5-base', 'bert-base-uncased'])
    parser.add_argument("--train_start", default=None)

    args = parser.parse_args()

    logging.info("The random seed is {}".format(args.sd))
    set_seed(sd=args.sd)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=None,
        use_fast=True)
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(tokenizer, args.device, dataset_name=args.task_name,
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
    config.use_cache = True

    if "bart" in args.model:
        model_class = BART4ERCG
    elif 't5' in args.model:
        model_class = T54ERC
    else:
        model_class = BERT4ERC

    #model = BERT4ERC(args=args)
    model = BART4ERCG.from_pretrained(
        args.model,
        config=config,
        args=args
    )
    model = model.to(args.device)

    best_score = sys.float_info.max
    steps_per_epoch = len(train_dataloader)

    # total number of training steps
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
            data['input_ids'], data['attention_mask'], data['labels'], data['next_input_ids'], data['speakers'], data['his_ids'], data['his_attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, his_ids=his_ids, his_attention_mask=his_attention_mask, next_input_ids=next_input_ids, labels=labels)
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, next_input_ids=next_input_ids, labels=labels)
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



