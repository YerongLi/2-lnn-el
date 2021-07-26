"""
Vectorized version of the LNN with learnable thresholds
    - containing BLINK features
    - with refcount

12 features in the data
[scores['jw'], scores['jacc'], scores['lev'], scores['spacy'], scores['in'], scores['pr'], scores['smith_waterman'] ,
normalized_ref_scores[ref_idx], normalized_ctx_scores_db[ctx_idx], has_class, scores['blink'], normalized_ctx_scores_blink[ctx_idx]]

"""
import sys
from abc import ABC
from typing import Any

from lnn.src.meta_rule.lnn_operators import and_lukasiewicz, or_lukasiewicz, negation, or_max, and_product
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(100)

from .rules_db import ContextLNNDBLong, PureNameLNNDB, ContextLNNDB, PureNameLNNDBLong, TypeLNNDB
from .rules_blink import ContextLNNBlinkLong, PureNameLNNBlink, ContextLNNBlink, PureNameLNNBlinkLong


class ComplexRuleDbpediaBLINK(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleDbpediaBLINK, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.rule_or_db = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None):
        pure_res = self.pureNameRule(x)
        context_res = self.contextRule(x)
        db_res = self.rule_or_db(torch.cat((pure_res, context_res), 1))

        pure_blink_res = self.pureNameBlinkRule(x)
        context_blink_res = self.contextBlinkRule(x)
        blink_res = self.rule_or_db(torch.cat((pure_blink_res, context_blink_res), 1))

        cat_features = torch.cat((db_res, blink_res), 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = db_res
        return final_res


class EnsembleRule(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleRule, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.rule_or_db = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res), 1))

        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class EnsembleTypeRule(nn.Module):
    """Ensemble of BLINK + LNN-EL"""
    def __init__(self, alpha, arity, slack=None, printout=False):
        super(EnsembleTypeRule, self).__init__()

        # self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameRule = PureNameLNNDBLong(alpha, -1, None)
        # self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlinkLong(alpha, -1, None)

        # self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextRule = ContextLNNDBLong(alpha, -1, None)
        # self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlinkLong(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)

        self.rule_or_db = or_lukasiewicz(alpha, 3, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)
        self.printout = printout

    def forward(self, x_db, x_blink, mention_labels=None):

        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res), 1))
            # print('pure_res', pure_res.shape)

        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        # if not (db_res.is_cuda and blink_res.is_cuda):
        #     print(x_db.is_cuda, 'x_db')
        #     print(x_blink.is_cuda, 'x_blink')
        #     print(db_res.is_cuda, 'db_res')
        #     print(blink_res.is_cuda, 'blink_res')
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        # print('cat_features.shape',cat_features.shape)
        # print('cat_features.shape',cat_features)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class ExactNameLNN(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(ExactNameLNN, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([0.3]).view(1, 1))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 1, slack)
        #self.predicate_and = and_lukasiewicz(alpha, 2, slack)

    def forward(self, x, mention_labels=None, debug=False):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]

            [scores['jw'], scores['jacc'], scores['lev'], scores['spacy'], scores['in'],
            scores['pr'], scores['smith_waterman'] ,normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx], has_class]

        """
        if debug:
            print(x.shape)

        sim_features = x[:, [5]].view(-1, 1)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        exact_res = self.sim_disjunction_or(sim_features_)

        # RULE 1: lookup predicate
        #lookup_features = x[:, 7].view(-1, 1)
        #lookup_features = torch.ones(lookup_features.shape)

        # BLINK score
        # blink_features = x[:, 10].view(-1, 1)

        #lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result), 1)
        #yhat = self.predicate_and(lookup_cat_disjunc_features)

        return exact_res


class EnsembleTypeRuleWebSQP(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleTypeRuleWebSQP, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)

        self.exactNameRule = ExactNameLNN(alpha, -1, None)
        self.rule_and_db = and_lukasiewicz(alpha, 2, slack)
        self.exactNameBlinkRule = ExactNameLNN(alpha, -1, None)
        self.rule_and_blink = and_lukasiewicz(alpha, 2, slack)

        self.rule_or_db = or_lukasiewicz(alpha, 3, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        # print(x_db.shape, x_blink.shape)
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res), 1))
            exact_res = self.exactNameRule(x_db)
            db_res = self.rule_and_db(torch.cat((db_res, exact_res), 1))
        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class BertRule(nn.Module):
    """LNN-EL + BLINK + BERT"""
    def __init__(self, alpha, arity=-1, slack=None):
        super(BertRule, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2, .2]))
        self.type_features_threshold = torch.nn.Parameter(torch.Tensor([.1]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 5, slack)
        self.predicate_and = and_lukasiewicz(alpha, 2, slack)
    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        # RULE 1: BERT score
        bert_features = x[:, 13].view(-1, 1)
        #lookup_features = torch.ones(lookup_features.shape)
        # type_features = x[:, 9].view(-1, 1)
        # type_features_ = type_features * nn.Sigmoid()(type_features - nn.Sigmoid()(self.type_features_threshold))
        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features = torch.cat([bert_features, sim_features], 1)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        yhat = self.sim_disjunction_or(sim_features_)
        #sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        #sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        #disjunction_result = self.sim_disjunction_or(sim_features_)
        # RULE 1 + RULE 2 + RULE 3
        #cat_features = torch.cat([bert_features, disjunction_result], 1)
        #yhat = self.predicate_and(cat_features)
        return yhat


class BoxRule(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(BoxRule, self).__init__()
        self.threshold = torch.nn.Parameter(torch.Tensor([.2, .2, .2, .2,.2]))
        self.type_features_threshold = torch.nn.Parameter(torch.Tensor([.1]))
        self.sim_disjunction_or = or_lukasiewicz(alpha, 5, slack)
        self.predicate_and = and_lukasiewicz(alpha, 2, slack)
    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        # RULE 1: BERT score (TODO: now the implementation is identical to BertRule,
        #  TODO: when we merge two datasets, box feature wil be posited to the last
        #box_features = torch.sigmoid(x[:, -1]).view(-1, 1)
        box_features = x[:, -1].view(-1, 1)
        #box_features = torch.ones(box_features.shape)
        # type_features = x[:, 9].view(-1, 1)
        # type_features_ = type_features * nn.Sigmoid()(type_features - nn.Sigmoid()(self.type_features_threshold))
        sim_features = x[:, [1, 2, 0, 3]].view(-1, 4)
        sim_features = torch.cat([box_features, sim_features], 1)
        sim_features_ = sim_features * nn.Sigmoid()(sim_features - nn.Sigmoid()(self.threshold))
        yhat = self.sim_disjunction_or(sim_features_)
        # RULE 1 + RULE 2 + RULE 3
        #cat_features = torch.cat([box_features, disjunction_result], 1)
        #print(cat_features[:5])
        #yhat = self.predicate_and(cat_features)
        return yhat


class EnsembleBLINKBoxRule(nn.Module):
    """Ensemble of LNN-EL + BLINK + BOX"""
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleBLINKBoxRule, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)
        self.boxRule = BoxRule(alpha, -1, None)

        self.rule_or_db = or_lukasiewicz(alpha, 4, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            box_res = self.boxRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res, box_res), 1))

        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class EnsembleBLINKBoxRuleWebQSP(nn.Module):
    """Ensemble of LNN-EL + BLINK + BOX"""
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleBLINKBoxRuleWebQSP, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)
        self.boxRule = BoxRule(alpha, -1, None)

        self.exactNameRule = ExactNameLNN(alpha, -1, None)
        self.rule_and_db = and_lukasiewicz(alpha, 2, slack)

        self.rule_or_db = or_lukasiewicz(alpha, 4, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            box_res = self.boxRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res, box_res), 1))
            exact_res = self.exactNameRule(x_db)
            db_res = self.rule_and_db(torch.cat((db_res, exact_res), 1))
        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class EnsembleBLINKBoxBertRule(nn.Module):
    """Ensemble of LNN-EL + BLINK + BOX"""
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleBLINKBoxBertRule, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)
        self.boxRule = BoxRule(alpha, -1, None)
        self.bertRule = BertRule(alpha, -1, None)

        self.rule_or_db = or_lukasiewicz(alpha, 5, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            box_res = self.boxRule(x_db)
            bert_res = self.bertRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res, box_res, bert_res), 1))

        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res


class EnsembleBLINKBoxBertRuleWebQSP(nn.Module):
    """Ensemble of LNN-EL + BLINK + BOX"""
    def __init__(self, alpha, arity, slack=None):
        super(EnsembleBLINKBoxBertRuleWebQSP, self).__init__()

        self.pureNameRule = PureNameLNNDB(alpha, -1, None)
        self.pureNameBlinkRule = PureNameLNNBlink(alpha, -1, None)

        self.contextRule = ContextLNNDB(alpha, -1, None)
        self.contextBlinkRule = ContextLNNBlink(alpha, -1, None)

        self.typeRule = TypeLNNDB(alpha, -1, None)
        self.boxRule = BoxRule(alpha, -1, None)
        self.bertRule = BertRule(alpha, -1, None)

        self.exactNameRule = ExactNameLNN(alpha, -1, None)
        self.rule_and_db = and_lukasiewicz(alpha, 2, slack)

        self.rule_or_db = or_lukasiewicz(alpha, 5, slack)
        self.rule_or_blink = or_lukasiewicz(alpha, 2, slack)
        self.rule_or_merge = or_lukasiewicz(alpha, 2, slack)

        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x_db, x_blink, mention_labels=None):
        if len(x_db) == 0:
            db_res = x_db
        else:
            pure_res = self.pureNameRule(x_db)
            context_res = self.contextRule(x_db)
            type_res = self.typeRule(x_db)
            box_res = self.boxRule(x_db)
            bert_res = self.bertRule(x_db)
            db_res = self.rule_or_db(torch.cat((pure_res, context_res, type_res, box_res, bert_res), 1))
            exact_res = self.exactNameRule(x_db)
            db_res = self.rule_and_db(torch.cat((db_res, exact_res), 1))
        if len(x_blink) == 0:
            blink_res = x_blink
        else:
            pure_blink_res = self.pureNameBlinkRule(x_blink)
            context_blink_res = self.contextBlinkRule(x_blink)
            blink_res = self.rule_or_blink(torch.cat((pure_blink_res, context_blink_res), 1))

        # if len(db_res) == 0 and len(blink_res) == 0:
        #     print("db_res", db_res, db_res.shape)
        #     print("blink_res", blink_res, blink_res.shape)
        db_res_extend = torch.cat([db_res, blink_res], 0)
        db_res_extend[len(db_res): , :] = 0
        blink_res_extend = torch.cat([db_res, blink_res], 0)
        blink_res_extend[:len(db_res), :] = 0

        # print("db_res_extend", db_res_extend)
        # print("blink_res_extend", blink_res_extend)

        cat_features = torch.cat((db_res_extend, blink_res_extend), 1)
        cat_features = self.batch_norm(cat_features)
        cat_features = torch.softmax(cat_features, 1)
        final_res = self.rule_or_merge(cat_features)
        # final_res = or_max()(cat_features)
        final_res = torch.softmax(final_res, 0)
        # print(final_res)
        return final_res
