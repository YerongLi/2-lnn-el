import sys
sys.path.append('../../src/meta_rule/')

from lnn_operators import and_lukasiewicz, or_lukasiewicz, negation
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(100)


class PureNameLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(PureNameLNN, self).__init__()
        self.threshold = 0.5

        self.sim_disjunction_or_ops = nn.ModuleList([or_lukasiewicz(alpha, arity, slack) for i in range(3)])
        self.predicate_and = and_lukasiewicz(alpha, arity, slack)
        self.batch_norm = nn.BatchNorm1d(6, affine=True)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        yhat = None
        x = self.batch_norm(x)
        ####### RULE 1: lookup predicate #######
        lookup_features = x[:, 4].view(-1, 1)
        #         print("lookup_features", lookup_features)

        ####### RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb)) #######
        feature_list = []
        # rule 2 (1) mention==label
        mentions = np.array([m.split(';')[0].strip().lower() for m in mention_labels])
        labels = np.array([m.split(';')[1].strip().lower() for m in mention_labels])
        exact_match_features = torch.from_numpy(np.array(mentions == labels).astype(float)).float().view(-1, 1)
        feature_list.append(exact_match_features)

        # rule 2 (2) Jacc(mention, label)
        jacc_features = x[:, 1].view(-1, 1)
        jacc_features_ = torch.where(jacc_features >= self.threshold, jacc_features, torch.zeros_like(jacc_features))
        feature_list.append(jacc_features_)

        # rule 2 (3) Lev(mention, label)
        lev_features = x[:, 2].view(-1, 1)
        lev_features_ = torch.where(lev_features >= self.threshold, lev_features, torch.zeros_like(lev_features))
        feature_list.append(lev_features_)

        # rule 2 (4) Jaro(mention, label)
        jaro_features = x[:, 0].view(-1, 1)
        jaro_features_ = torch.where(jaro_features >= self.threshold, jaro_features, torch.zeros_like(jaro_features))
        feature_list.append(jaro_features_)

        # disjunction of (1) to (4)
        disjunction_result = feature_list[0]
        for i in range(0, 3):
            disjunction_result = self.sim_disjunction_or_ops[i](torch.cat((disjunction_result, feature_list[i + 1]), 1))

        # RULE 1 + RULE 2
        yhat = self.predicate_and(torch.cat((lookup_features, disjunction_result), 1))
        return yhat


class ContextLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNN, self).__init__()
        self.threshold = 0.2
        self.context_threshold = 0.25
        self.sim_disjunction_or_ops = nn.ModuleList([or_lukasiewicz(alpha, arity, slack) for i in range(3)])
        self.predicate_and_ops = nn.ModuleList([and_lukasiewicz(alpha, arity, slack) for i in range(2)])
        self.batch_norm = nn.BatchNorm1d(6, affine=True)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        yhat = None
        x = self.batch_norm(x)
        ####### RULE 1: lookup predicate #######
        lookup_features = x[:, 4].view(-1, 1)
        #         print("lookup_features", lookup_features)

        ####### RULE 3: contains predicate #######
        context_features = x[:, 5].view(-1, 1)
        # context mask
        context_mask = context_features >= self.context_threshold

        ####### RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb)) #######
        # check: https://github.ibm.com/IBM-Research-AI/enhanced_amr/blob/5501e3af41794353ed9bb147320666622474171a/entity_linking.py#L295
        feature_list = []
        # rule 2 (1) mention==label
        mentions = np.array([m.split(';')[0].strip().lower() for m in mention_labels])
        labels = np.array([m.split(';')[1].strip().lower() for m in mention_labels])
        exact_match_features = torch.from_numpy(np.array(mentions == labels).astype(float)).float().view(-1, 1)
        feature_list.append(exact_match_features)

        # rule 2 (2) Jacc(mention, label)
        jacc_features = x[:, 1].view(-1, 1)
        jacc_features_ = torch.where(jacc_features >= self.threshold, jacc_features,
                                     torch.zeros_like(jacc_features))
        feature_list.append(jacc_features_ * context_mask)

        # rule 2 (3) Lev(mention, label)
        lev_features = x[:, 2].view(-1, 1)
        lev_features_ = torch.where(lev_features >= self.threshold, lev_features, torch.zeros_like(lev_features))
        feature_list.append(lev_features_ * context_mask)

        # rule 2 (4) Jaro(mention, label)
        jaro_features = x[:, 0].view(-1, 1)
        jaro_features_ = torch.where(jaro_features >= self.threshold, jaro_features,
                                     torch.zeros_like(jaro_features))
        feature_list.append(jaro_features_ * context_mask)

        # disjunction of (1) to (4)
        disjunction_result = feature_list[0]
        for i in range(0, 3):
            disjunction_result = self.sim_disjunction_or_ops[i](torch.cat((disjunction_result, feature_list[i + 1]), 1))

        # RULE 1 + RULE 2
        r1_r2_res = self.predicate_and_ops[0](torch.cat((lookup_features, disjunction_result), 1))
        yhat = self.predicate_and_ops[1](torch.cat((r1_r2_res, context_features), 1))
        return yhat


class ComplexRuleLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleLNN, self).__init__()

        self.pureNameRule = PureNameLNN(alpha, arity, None)
        self.contextRule = ContextLNN(alpha, arity, None)
        self.rule_or = or_lukasiewicz(alpha, arity, slack)
        self.batch_norm = nn.BatchNorm1d(2, affine=True)

    def forward(self, x, mention_labels=None):
        yhat = None
        pure_res = self.pureNameRule(x, mention_labels)
        context_res = self.contextRule(x, mention_labels)
        cat_features = torch.cat((pure_res, context_res), 1)
        cat_features = self.batch_norm(cat_features)
        pure_context_res = self.rule_or(cat_features)
        return pure_context_res