import sys
sys.path.append('../../src/meta_rule/')

from lnn_operators import and_lukasiewicz, or_lukasiewicz, negation, or_max, and_product, or_sum
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(100)


class PureNameLNN(nn.Module):
    def __init__(self, alpha, arity=-1, slack=None):
        super(PureNameLNN, self).__init__()
        self.threshold = 0.5

        self.sim_disjunction_or = or_lukasiewicz(alpha, 5, slack)  # dummy weight to pass optimizer

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        # x = self.batch_norm(x)
        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb))
        feature_list = []
        # rule 2 (1) mention==label
        # mentions = np.array([m.split(';')[0].strip().lower() for m in mention_labels])
        # labels = np.array([m.split(';')[1].strip().lower() for m in mention_labels])
        # exact_match_features = torch.from_numpy(np.array(mentions == labels).astype(float)).float().view(-1, 1)
        # feature_list.append(exact_match_features)

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

        # rule 2 (6) blink
        blink_features = x[:, 10].view(-1, 1)
        blink_features_ = torch.where(blink_features > self.threshold, blink_features, torch.zeros_like(blink_features))
        feature_list.append(blink_features_)

        # disjunction of (1) to (5)
        disjunction_result = or_sum()(torch.cat(feature_list, 1))

        # RULE 1 + RULE 2
        # lookup_cat_disjunc_features = torch.cat((lookup_features, disjunction_result), 1)
        # yhat = and_product()(disjunction_result)  # and_product
        return disjunction_result * 0.5


class ContextLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ContextLNN, self).__init__()
        self.threshold = 0.2
        self.context_threshold = 0.25
        self.sim_disjunction_or = or_lukasiewicz(alpha, 5, slack)  # dummy weight to pass optimizer

    def forward(self, x, mention_labels=None):
        """
            x: scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
               normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx]
        """
        # x = self.batch_norm(x)
        # RULE 1: lookup predicate
        # lookup_features = x[:, 7].view(-1, 1)

        # RULE 3: contains predicate
        context_features = x[:, 8].view(-1, 1)
        # context mask
        context_mask = context_features >= self.context_threshold

        # RULE 2: similarity predicate(mention==label AND Jacc(m, lb) AND Lev(m, lb) AND Jaro(m, lb))
        feature_list = []
        # rule 2 (1) mention==label
        # mentions = np.array([m.split(';')[0].strip().lower() for m in mention_labels])
        # labels = np.array([m.split(';')[1].strip().lower() for m in mention_labels])
        # exact_match_features = torch.from_numpy(np.array(mentions == labels).astype(float)).float().view(-1, 1)
        # feature_list.append(exact_match_features)

        # rule 2 (2) Jacc(mention, label)
        jacc_features = x[:, 1].view(-1, 1)
        jacc_features_ = torch.where(jacc_features > self.threshold, jacc_features,
                                     torch.zeros_like(jacc_features))
        #print(jacc_features_)
        #feature_list.append(jacc_features_ * context_mask)

        # rule 2 (3) Lev(mention, label)
        lev_features = x[:, 2].view(-1, 1)
        lev_features_ = torch.where(lev_features > self.threshold, lev_features, torch.zeros_like(lev_features))
        feature_list.append(lev_features_ * context_mask)

        # rule 2 (4) Jaro(mention, label)
        jaro_features = x[:, 0].view(-1, 1)
        jaro_features_ = torch.where(jaro_features > self.threshold, jaro_features,
                                     torch.zeros_like(jaro_features))
        feature_list.append(jaro_features_ * context_mask)

        # rule 2 (5) spacy
        spacy_features = x[:, 3].view(-1, 1)
        spacy_features_ = torch.where(spacy_features > self.threshold, spacy_features,
                                     torch.zeros_like(spacy_features))
        feature_list.append(spacy_features_ * context_mask)

        # rule 2 (6) blink
        blink_features = x[:, 10].view(-1, 1)
        blink_features_ = torch.where(blink_features > self.threshold, blink_features, torch.zeros_like(blink_features))
        feature_list.append(blink_features_ * context_mask)

        # disjunction of (1) to (5)
        disjunction_result = or_sum()(torch.cat(feature_list, 1)) # or_sum

        # RULE 1 + RULE 2 + RULE 3
        cat_features = torch.cat([disjunction_result, context_features], 1)
        yhat = and_product()(cat_features)  # and_product
        return yhat


class ComplexRuleLNN(nn.Module):
    def __init__(self, alpha, arity, slack=None):
        super(ComplexRuleLNN, self).__init__()

        self.pureNameRule = PureNameLNN(alpha, -1, None)
        self.contextRule = ContextLNN(alpha, -1, None)

    def forward(self, x, mention_labels=None):
        pure_res = self.pureNameRule(x, mention_labels)
        context_res = self.contextRule(x, mention_labels)
        cat_features = torch.cat((pure_res, context_res), 1)
        print(cat_features.max(), cat_features.min())
        pure_context_res = or_sum()(cat_features)  # or_sum
        return pure_context_res


