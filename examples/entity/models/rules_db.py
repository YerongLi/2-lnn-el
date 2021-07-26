"""
Vectorized version of the LNN with learnable thresholds
    - containing BLINK features
    - with refcount

13 features in the data
[scores['jw'], scores['jacc'], scores['lev'], scores['spacy'], scores['in'], scores['pr'], scores['smith_waterman'] ,
normalized_ref_scores[ref_idx], normalized_ctx_scores_db[ctx_idx], has_class,
scores['blink'], normalized_ref_scores_blink[ref_idx], normalized_ctx_scores_blink[ctx_idx]]
"""
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


class PureNameLNNDB(nn.Module):
    '''
    Similarity Scores as disjunctions
    '''
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNNDB, self).__init__()
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
        lookup_features = x[:, 7].view(-1, 1)
        # lookup_features = x[:, 16].view(-1, 1)
        # print('lookup_features', lookup_features)

        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)
        
        lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat

class PureNameLNNDBLong(nn.Module):
    '''
    Similarity Scores as disjunctions
    '''
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNNDBLong, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        # self.sim_disjunction_or1 = or_lukasiewicz(alpha, 4, slack)
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
        # sim_features1 = x[:, [19, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)
        # disjunction_result2 = self.sim_disjunction_or(sim_features_1)
        # disjunction_result = self.sim_disjunction_or(sim_features_2)

        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)
        lookup_features = x[:, 16].view(-1, 1)
        # print('lookup_features', lookup_features)

        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)
        
        lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result), 1)
        yhat = self.predicate_and(lookup_cat_disjunc_features)

        return yhat



class ContextLNNDB(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNNDB, self).__init__()
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
        lookup_features = x[:, 7].view(-1, 1)
        # lookup_features = x[:, 16].view(-1, 1)


        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        # context_features = x[:, 17].view(-1, 1)
        # print('context_features', context_features)
        context_features = x[:, 8].view(-1, 1)
        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        # context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, context_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat

class ContextLNNDBLong(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNNDBLong, self).__init__()
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
        lookup_features = x[:, 16].view(-1, 1)


        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        # print('context_features', context_features)
        # context_features = x[:, 8].view(-1, 1)
        context_features = x[:, 17].view(-1, 1)

        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        # context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, context_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat

class ContextLNNDB1(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNNDB, self).__init__()
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
        lookup_features = x[:, 7].view(-1, 1)

        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)

        # RULE 3: contains predicate
        context_features = x[:, 8].view(-1, 1)
        context_features_ = context_features * nn.Sigmoid()(context_features - nn.Sigmoid()(self.context_threshold))
        # context_mask = context_features >= self.context_threshold

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, context_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat

class TypeLNNDB(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(TypeLNNDB, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2]))
        self.type_features_threshold = torch.nn.Parameter(torch.Tensor([.1]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 4, slack)
        self.predicate_and = and_lukasiewicz(alpha, 3, slack)

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """

        # RULE 1: lookup predicate
        lookup_features = x[:, 7].view(-1, 1)
        #lookup_features = torch.ones(lookup_features.shape)

        type_features = x[:, 9].view(-1, 1)
        type_features_ = type_features * nn.Sigmoid()(type_features - nn.Sigmoid()(self.type_features_threshold))

        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        disjunction_result = self.sim_disjunction_or(sim_features_)

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([lookup_features, disjunction_result, type_features_], 1)
        yhat = self.predicate_and(cat_features)
        return yhat