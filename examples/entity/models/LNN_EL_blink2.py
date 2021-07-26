"""
Vectorized version of the LNN with learnable thresholds
    - containing BLINK features
    - with refcount

11 features in the data
[scores['jw'], scores['jacc'], scores['lev'], scores['spacy'], scores['in'], scores['pr'], scores['smith_waterman'] ,
normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx], has_class, scores['blink']]
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
        self.linear = nn.Linear(11, 1)

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
        self.threshold = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None, debug=False):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]

            [scores['jw'], scores['jacc'], scores['lev'], scores['spacy'], scores['in'],
            scores['pr'], scores['smith_waterman'] ,normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx], has_class]

        """
        if debug:
            print(x.shape)

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1: lookup predicate
        lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result, blink_features), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat


class ContextLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNN, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2]))
        self.context_threshold = torch.nn.Parameter(torch.Tensor([.25]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 4, slack)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """

        # RULE 1: lookup predicate
        lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        context_features = x[:, 8].view(-1, 1)
        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, context_features_, blink_features], 1)
        yhat = self.predicate_and(cat_features)
        return yhat


class TypeLNN(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(TypeLNN, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.5, .5, .5, .5, .5]))
        self.type_features_threshold = torch.nn.Parameter(torch.Tensor([.1]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 5, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """

        # RULE 1: lookup predicate
        lookup_features = x[:, 7].view(-1, 1)

        type_features = x[:, 9].view(-1, 1)
        type_features_ = type_features * nn.Sigmoid()(type_features - nn.Sigmoid()(self.type_features_threshold))

        sim_features = x[:, [1, 2, 0, 3, 10]].view(-1, 5)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, type_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat


class ComplexRuleWithTypeLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleWithTypeLNN, self).__init__()

        self.pureNameRule = PureNameLNN(alpha, -1, None)
        self.contextRule = ContextLNN(alpha, -1, None)
        self.typeRule = TypeLNN(alpha, -1, None)
        self.rule_or = or_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        pure_res = self.pureNameRule(x, mention_labels)
        context_res = self.contextRule(x, mention_labels)
        type_res = self.typeRule(x, mention_labels)
        cat_features = torch.cat((pure_res, context_res, type_res), 1)
        final_res = self.rule_or(cat_features)
        return final_res


class ComplexRuleWithoutTypeLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleWithoutTypeLNN, self).__init__()

        self.pureNameRule = PureNameLNN(alpha, -1, None)
        self.contextRule = ContextLNN(alpha, -1, None)
        self.rule_or = or_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None):
        pure_res = self.pureNameRule(x, mention_labels)
        context_res = self.contextRule(x, mention_labels)
        cat_features = torch.cat((pure_res, context_res), 1)
        final_res = self.rule_or(cat_features)
        return final_res
