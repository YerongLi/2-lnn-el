
import sys
from typing import Any

#sys.path.append('../../src/meta_rule/')

from lnn.src.meta_rule.lnn_operators import and_lukasiewicz, or_lukasiewicz, negation, or_max, and_product
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(100)

class PureNameLNNBlink(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNNBlink, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None, debug=False):
        """
            The feature details:
            0: [scores['jw']
            1: scores['jacc']
            2: scores['lev']
            3: scores['spacy']
            4: scores['in']
            5: scores['pr']
            6: scores['smith_waterman']
            7: normalized_ref_scores_db[ref_idx]
            8: normalized_ctx_scores_db[ctx_idx]
            9: has_class
            10: scores['blink']
            11: normalized_ref_scores_blink[ref_idx]
            12: normalized_ctx_scores_blink[ctx_idx]]
        """
        if debug:
            print(x.shape)

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        lookup_cat_disjunc_features = torch.cat((blink_features, disjunction_result), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat

class PureNameLNNBlinkLong(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNNBlinkLong, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None, debug=False):
        """
            The feature details:
            0: [scores['jw']
            1: scores['jacc']
            2: scores['lev']
            3: scores['spacy']
            4: scores['in']
            5: scores['pr']
            6: scores['smith_waterman']
            7: normalized_ref_scores_db[ref_idx]
            8: normalized_ctx_scores_db[ctx_idx]
            9: has_class
            10: scores['blink']
            11: normalized_ref_scores_blink[ref_idx]
            12: normalized_ctx_scores_blink[ctx_idx]]
        """
        if debug:
            print(x.shape)

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        lookup_cat_disjunc_features = torch.cat((blink_features, disjunction_result), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat


class ContextLNNBlink(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNNBlink, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2]))
        self.context_threshold = torch.nn.Parameter(torch.Tensor([.25]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        """
             The feature details:
            0: [scores['jw']
            1: scores['jacc']
            2: scores['lev']
            3: scores['spacy']
            4: scores['in']
            5: scores['pr']
            6: scores['smith_waterman']
            7: normalized_ref_scores_db[ref_idx]
            8: normalized_ctx_scores_db[ctx_idx]
            9: has_class
            10: scores['blink']
            11: normalized_ref_scores_blink[ref_idx]
            12: normalized_ctx_scores_blink[ctx_idx]]
        """

        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        # context_features = x[:, 17].view(-1, 1)
        context_features = x[:, 12].view(-1, 1)
        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        # context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([blink_features, disjunction_result, context_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat

class ContextLNNBlinkLong(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNNBlinkLong, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2]))
        self.context_threshold = torch.nn.Parameter(torch.Tensor([.25]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        """
             The feature details:
            0: [scores['jw']
            1: scores['jacc']
            2: scores['lev']
            3: scores['spacy']
            4: scores['in']
            5: scores['pr']
            6: scores['smith_waterman']
            7: normalized_ref_scores_db[ref_idx]
            8: normalized_ctx_scores_db[ctx_idx]
            9: has_class
            10: scores['blink']
            11: normalized_ref_scores_blink[ref_idx]
            12: normalized_ctx_scores_blink[ctx_idx]]
        """

        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        context_features = x[:, 17].view(-1, 1)
        # context_features = x[:, 12].view(-1, 1)
        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        # context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([blink_features, disjunction_result, context_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat

