#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import RGCNConv, GraphConv
from transformers import BertModel

import numpy as np, itertools, random, copy, math

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        # score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            # scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()#mask, bs, 110, 110
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()

            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()

            edge_ind_ = []
            for i, j in enumerate(edge_ind):# i:bs; j:node2node tuple
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()# np.array(edge_ind_), num of node2node,3(bs, startnode, endnode)
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy
            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            # if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            # if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()

    # if torch.cuda.is_available():
    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                            for s, l in zip(start.data.tolist(),
                                            input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions

def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer,
                           nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """

    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob

class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5,
                 no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()

        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size, att_type='general2')
        self.linear = nn.Linear(num_features + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type)
        #out = self.conv1(x, edge_index, edge_type, edge_norm)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout,
                                          self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob

def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces
    from any other dialogue instances in that mini-batch.
    """

    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    # scores are the edge weights, features:sl,bs,hd
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])

            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()

            if item1[0] < item1[1]:
                # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    # if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths

class DGCN(nn.Module):
    def __init__(self):
        super(DGCN, self).__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, dropout=0.5)
        n_relations = 2 * 2 ** 2
        self.window_past = 10
        self.window_future = 10

        self.proj1 = nn.Linear(in_features=768, out_features=100)

        self.att_model = MaskedEdgeAttention(2 * 100, 128, False)
        self.nodal_attention = True

        self.graph_net = GraphNetwork(2 * 100, 6, n_relations, 128, 100, 0.5,
                                      False)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        edge_type_mapping = {}
        for j in range(2):
            for k in range(2):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

    def forward(self, t5features, t5attention_maks, qmask, umask, seq_lenghts):
        """
        :param t5features: bs, sesslen, seqlen
        :param t5attention_maks: bs, sesslen, seqlen
        :param qmask: sesslen, bs, num_speakers
        :param umask: sess_num, bs
        :param seq_lenghts: list[l1, l2, ..., lbs]
        :return:
        """
        ptm_hidden = self.bert(input_ids=t5features.view(-1, 128), attention_mask=t5attention_maks.view(-1, 128))
        ptm_hidden = self.proj1(ptm_hidden.last_hidden_state)
        emotions, hidden = self.lstm(ptm_hidden)
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions.view(2,-1,emotions.shape[1], emotions.shape[2]).permute(1,0,2,3), qmask, seq_lenghts,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model, False)

        print(features)