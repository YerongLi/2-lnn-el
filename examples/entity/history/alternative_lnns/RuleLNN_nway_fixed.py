"""
Original version of the LNN with fixed thresholds code
    - more self-explanatory, but less efficient
"""
import sys
sys.path.append('../../src/meta_rule/')

from lnn_operators import and_lukasiewicz, or_lukasiewicz, negation, or_max, and_product
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(100)


class LogitsRegression(nn.Module):
    def __init__(self):
        super(LogitsRegression, self).__init__()
        self.batch_norm = nn.BatchNorm1d(6, affine=True)
        self.linear = nn.Linear(6, 1)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        yhat = torch.sigmoid(self.linear(x))
        return yhat


class PureNameLNN(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNN, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.5]))

        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None, debug=False):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        if debug:
            print(x.shape)
        # RULE 1: lookup predicate
        lookup_features = x[:, 4].view(-1, 1)

        # RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb))
        feature_list = []

        # rule 2 (2) Jacc(mention, label)
        jacc_features = x[:, 1].view(-1, 1)
        jacc_features_ = torch.where(jacc_features > self.threshold, jacc_features, torch.zeros_like(jacc_features))
        feature_list.append(jacc_features_)

        # rule 2 (3) Lev(mention, label)
        lev_features = x[:, 2].view(-1, 1)
        lev_features_ = torch.where(lev_features > self.threshold, lev_features, torch.zeros_like(lev_features))
        feature_list.append(lev_features_)

        # rule 2 (4) Jaro(mention, label)
        jaro_features = x[:, 0].view(-1, 1)
        jaro_features_ = torch.where(jaro_features > self.threshold, jaro_features, torch.zeros_like(jaro_features))
        feature_list.append(jaro_features_)

        # rule 2 (5) spacy
        spacy_features = x[:, 3].view(-1, 1)
        spacy_features_ = torch.where(spacy_features > self.threshold, spacy_features, torch.zeros_like(spacy_features))
        feature_list.append(spacy_features_)

        if debug:
            print('>>>>>', torch.cat(feature_list, 1).shape)
            print('endl')

        # disjunction of (1) to (5)
        disjunction_result = self.sim_disjunction_or(torch.cat(feature_list, 1))

        #if debug:
        #    print(disjunction_result.shape)

        # RULE 1 + RULE 2
        lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat


class ContextLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNN, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2]))
        self.context_threshold = torch.nn.Parameter(torch.Tensor([.25]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        # RULE 1: lookup predicate
        lookup_features = x[:, 4].view(-1, 1)

        # RULE 3: contains predicate
        context_features = x[:, 5].view(-1, 1)
        # context mask
        context_mask = context_features >= self.context_threshold

        # RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb))
        feature_list = []

        # rule 2 (2) Jacc(mention, label)
        jacc_features = x[:, 1].view(-1, 1)
        jacc_features_ = torch.where(jacc_features > self.threshold, jacc_features, torch.zeros_like(jacc_features))
        feature_list.append(jacc_features_ * context_mask)

        # rule 2 (3) Lev(mention, label)
        lev_features = x[:, 2].view(-1, 1)
        lev_features_ = torch.where(lev_features > self.threshold, lev_features, torch.zeros_like(lev_features))
        feature_list.append(lev_features_ * context_mask)

        # rule 2 (4) Jaro(mention, label)
        jaro_features = x[:, 0].view(-1, 1)
        jaro_features_ = torch.where(jaro_features > self.threshold, jaro_features, torch.zeros_like(jaro_features))
        feature_list.append(jaro_features_ * context_mask)

        # rule 2 (5) spacy
        spacy_features = x[:, 3].view(-1, 1)
        spacy_features_ = torch.where(spacy_features > self.threshold, spacy_features, torch.zeros_like(spacy_features))
        feature_list.append(spacy_features_ * context_mask)

        # disjunction of (1) to (5)
        disjunction_result = self.sim_disjunction_or(torch.cat(feature_list, 1))

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, context_features], 1)
        yhat = self.predicate_and(cat_features)
        return yhat


class ComplexRuleLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleLNN, self).__init__()

        self.pureNameRule = PureNameLNN(alpha, -1, None)
        self.contextRule = ContextLNN(alpha, -1, None)
        self.rule_or = or_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None):
        pure_res = self.pureNameRule(x, mention_labels)
        context_res = self.contextRule(x, mention_labels)
        cat_features = torch.cat((pure_res, context_res), 1)
        pure_context_res = self.rule_or(cat_features)
        return pure_context_res


