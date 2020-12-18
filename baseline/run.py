import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import pandas as pd
import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re 


if __name__ == '__main__':
    """hyperparams"""
    SEED = 2020
    # Adult, Amazon, Click prediction, KDD appetency, KDD churn, KDD internet, KDD upselling, KDD 98, Kick prediction
    DATASET = 'Adult'  
    DATASET_PATH = "/home/v-tyan/NN_for_tabular/datasets/"
    MODEL = 'lgb'  # xgb, lgb, cat, mlp, deepfm, xdeepfm, tabnet

    if MODEL == 'cat':
        from models import CATModel
        Model = CATModel
    elif MODEL == 'lgb':
        from models import LGBModel
        Model = LGBModel
    elif MODEL == 'xgb':
        from models import XGBModel
        Model = XGBModel
    elif MODEL == 'mlp':
        from models import MLPModel
        Model = MLPModel
    elif MODEL == 'deepfm':
        from models import DeepFMModel
        Model = DeepFMModel
    elif MODEL == 'xdeepfm':
        from models import XDeepFMModel
        Model = XDeepFMModel
    elif MODEL == 'tabnet':
        from models import TabNetModel
        Model = TabNetModel
    
    
    





# seed
np.random.seed(SEED)


# load data
train_X, train_Y, test_X, test_Y, cat_cols, num_cols = data_loader(DATASET, DATASET_PATH)


# training & testing
if MODEL == 'cat':
    model_default = cat.CatBoostClassifier(verbose=10)
    model_default.fit(train_X, train_Y, cat_features=cat_cols)
    test_Y_hat = model_default.predict_proba(test_X)
    print(f'Default Logloss values: {log_loss(test_Y, test_Y_hat)}')
elif MODEL == 'lgb':
    encoder = LabelEncoder()
    for cat_col in cat_cols:
        encoder.fit(list(train_X[cat_col]) + ['UNK'])
        test_X[cat_col], train_X[cat_col] = encoder.transform(test_X[cat_col]), encoder.transform(train_X[cat_col])
    model_default = lgb.LGBMClassifier()
    model_default.fit(train_X, train_Y, categorical_feature=cat_cols)
    test_Y_hat = model_default.predict_proba(test_X)
    print(f'Default Logloss values: {log_loss(test_Y, test_Y_hat)}')
elif MODEL == 'xgb':
    encoder = OneHotEncoder(sparse=False)
    train_X_cat = encoder.fit(train_X[cat_cols].append(pd.DataFrame([['UNK'] * len(cat_cols)], columns=cat_cols)))
    test_X_cat, train_X_cat = encoder.transform(test_X[cat_cols]), encoder.transform(train_X[cat_cols])
    model_default = xgb.XGBClassifier()
    model_default.fit(np.concatenate((train_X[num_cols], train_X_cat), axis=1), train_Y)
    test_Y_hat = model_default.predict_proba(np.concatenate((test_X[num_cols], test_X_cat), axis=1))
    print(f'Default Logloss values: {log_loss(test_Y, test_Y_hat)}')
