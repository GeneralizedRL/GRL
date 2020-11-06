"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Fact scoring networks.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/model.py
"""

import torch.nn as nn
import torch.nn.functional as F
from util.utils import *


def create_model(args):
    args = DottableDict(args)
    print('Loading Embedding Model.')
    embmodel = args.embmodel
    if embmodel == 'conve':
        model1 = ConvE(args, args.num_entities)
        return model1
    elif embmodel == 'distmult':
        model1 = DistMult(args)
        return model1


class ConvE(nn.Module):
    def __init__(self, args, num_entities):
        super(ConvE, self).__init__()
        self.args = args
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations

        assert (args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert (args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)
        self.emb_dropout_rate = args.emb_dropout_rate
        self.define_modules()
        self.initialize_modules()

        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    def forward(self, x, openset=False):
        if openset:
            e1, e2, r = x.split(1, 1)
            e1 = e1.view(-1)
            r = r.view(-1)
            e2 = e2.view(-1)

            head = self.entity_embeddings(e1)
            relation = self.relation_embeddings(r)
            tail = self.get_entity_embeddings(e2)

            E1 = self.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
            R = self.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
            E2 = self.get_entity_embeddings(e2)

            stacked_inputs = torch.cat([E1, R], 2)
            stacked_inputs = self.bn0(stacked_inputs)

            X = self.conv1(stacked_inputs)
            # X = self.bn1(X)
            X = F.relu(X)
            X = self.FeatureDropout(X)
            X = X.view(-1, self.feat_dim)
            X = self.fc(X)
            X = self.HiddenDropout(X)
            X = self.bn2(X)
            X = F.relu(X)
            X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
            X += self.b[e2].unsqueeze(1)

            S = F.sigmoid(X)

            return head, tail, relation, S, self.get_all_relation_embeddings()

        else:
            e1, e2, r = x.split(1, 1)
            e1 = e1.view(-1)
            r = r.view(-1)
            e2 = e2.view(-1)

            E1 = self.entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
            R = self.relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
            E2 = self.get_entity_embeddings(e2)
            E2_all = self.get_all_entity_embeddings()

            stacked_inputs = torch.cat([E1, R], 2)
            stacked_inputs = self.bn0(stacked_inputs)

            X = self.conv1(stacked_inputs)
            # X = self.bn1(X)
            X = F.relu(X)
            X = self.FeatureDropout(X)
            X = X.view(-1, self.feat_dim)
            X = self.fc(X)
            X = self.HiddenDropout(X)
            X = self.bn2(X)
            X = F.relu(X)
            X_1 = X

            X_all = torch.mm(X_1, E2_all.transpose(1, 0))
            X_all += self.b.expand_as(X_all)

            S_all = F.sigmoid(X_all)
            S = S_all
            return S, S_all, E1.view(-1, self.entity_dim), self.entity_embeddings(e2), R.view(-1,
                                                                                              self.relation_dim), self.get_all_relation_embeddings()

    def define_modules(self):

        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)


class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.args = args
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.emb_dropout_rate = args.emb_dropout_rate
        self.define_modules()
        self.initialize_modules()

    def forward(self, x, openset=False):
        e1, e2, r = x.split(1, 1)
        e1 = e1.view(-1)
        r = r.view(-1)
        e2 = e2.view(-1)
        if openset:
            E1 = self.get_entity_embeddings(e1)
            R = self.get_relation_embeddings(r)
            E2 = self.get_entity_embeddings(e2)
            S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
            S = F.sigmoid(S)

            return E1, E2, R, S, self.get_all_relation_embeddings()
        else:
            E1 = self.get_entity_embeddings(e1)
            R = self.get_relation_embeddings(r)
            E2_all = self.get_all_entity_embeddings()
            S_all = torch.mm(E1 * R, E2_all.transpose(1, 0))
            S_all = F.sigmoid(S_all)

            return S_all, S_all, E1.view(-1, self.entity_dim), self.entity_embeddings(e2), R.view(-1,
                                                                                                  self.relation_dim), self.get_all_relation_embeddings()

    def define_modules(self):
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)
