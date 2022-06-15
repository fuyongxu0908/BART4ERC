#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from numpy import random
import pickle
import argparse
from transformers import BertTokenizer

from DGCN import DialogueGCNModel
from Graph import DGCN

class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open('./IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, idx):
        vid = self.keys[idx]
        return self.videoSentence[vid], \
               torch.FloatTensor(np.array(self.videoText[vid])), \
               torch.FloatTensor(np.array([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]])), \
               torch.FloatTensor(np.array([1] * len(self.videoLabels[vid]))), \
               torch.LongTensor(np.array(self.videoLabels[vid]))

    def __len__(self):
        return self.len

    def collate_fn(self, batch, tokenizer):
        t5features = pad_sequence([tokenizer(b[0], max_length=128, padding='max_length', truncation=True, return_tensors='pt')['input_ids'] for b in batch], batch_first=True, padding_value=1)
        t5attention_mask = pad_sequence([tokenizer(b[0], max_length=128, padding='max_length', truncation=True, return_tensors='pt')['attention_mask'] for b in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([b[-1] for b in batch])
        speakers = pad_sequence([b[2] for b in batch])
        umaks = pad_sequence([b[3] for b in batch])
        return t5features, t5attention_mask, labels, speakers, umaks


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, tokenizer=None):
    train_set = IEMOCAPDataset()
    train_loader = DataLoader(dataset=train_set, batch_size=2, collate_fn=lambda x: train_set.collate_fn(x, tokenizer))
    valid_loader = DataLoader(dataset=train_set, batch_size=2, collate_fn=lambda x: train_set.collate_fn(x, tokenizer))
    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset, batch_size=16, collate_fn=testset.collate_fn)
    return train_loader, valid_loader, test_loader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False, args=None):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        t5features, t5attention_mask, labels, speakers, umaks = data
        t5features, t5attention_mask, labels, speakers, umaks = t5features.to("cuda"), t5attention_mask.to("cuda"), labels.to("cuda"), speakers.to("cuda"), umaks.to("cuda")
        lengths = [(umaks.permute(1,0)[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umaks.permute(1,0)))]
        model(t5features, t5attention_mask, speakers, umaks, lengths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base_model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph_model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal_attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec_dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=2, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active_listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--seed', type=int, default=1234)


    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    if args.graph_model:
        seed_everything(args.seed)
    loss_function = nn.NLLLoss()

    model = DGCN()
    if args.cuda:
        model = model.to("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(tokenizer=tokenizer)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    for e in range(n_epochs):
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, optimizer, True, args)

