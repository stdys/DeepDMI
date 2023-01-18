from sklearn import metrics
from sklearn import metrics as metricser
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import torch
from scipy import stats
import rs_metrics 
import pandas as pd
    
def evaluate_mrr(db, tax, label, pred, top):
    re = pd.DataFrame()
    re['tax'] = tax
    re['db'] = db
    re['label'] = label
    re['pred'] = pred
    re.sort_values(by='pred', ascending=False, inplace=True)
    re_true = re[re['label']==1]
    mrr_ = rs_metrics.mrr(re_true, re, k=top, user_col='tax', item_col='db')
    return mrr_

def evaluate_mrr_p(db, tax, uni, label, pred, m1, m2, top):
    re = pd.DataFrame()
    re['tax'] = tax
    re['db'] = db
    re['uni'] = uni
    re['label'] = label
    re['pred'] = pred
    re['mask1'] = m1
    re['mask2'] = m2
    t_d = re['tax'].map(str) + '_' + re['db'].map(str)
    re['t_d'] = t_d
    re = re[(re['mask1']==1) & (re['mask2']==1)]
    re.sort_values(by='pred', ascending=False, inplace=True)
    re_true = re[re['label']==1]
    mrr_ = rs_metrics.mrr(re_true, re, k=top, user_col='t_d', item_col='uni')
    return mrr_

