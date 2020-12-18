import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re 
from utils import *


### TODO: add parameter tuning by hyperopt

class BaseModel():
    def __init__(self, dataset, dataset_path):
        self.model_name = None
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.model = None

    def fit(self, train_X, train_Y, cat_cols):
        raise NotImplementedError('Method train is not implemented.')

    def test(self, test_X, test_Y, cat_cols):
        test_Y_hat = self.model.predict_proba(test_X)
        print(f'{self.model_name} Default Logloss values: {log_loss(test_Y, test_Y_hat)}')

    def preprocessing(self, train_X, test_X, cat_cols):
        return train_X, test_X

    def run(self):
        print('loading data...')
        train_X, train_Y, test_X, test_Y, cat_cols = data_loader(self.dataset, self.dataset_path)

        print('preprocessing data...')
        train_X, test_X = self.preprocessing(train_X, test_X, cat_cols)

        ### TODO: add parameter tuning by hyperopt

        print('training...')
        self.fit(train_X, train_Y, cat_cols)

        print('testing...')
        self.test(test_X, test_Y, cat_cols)


class CATModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)
        self.model_name = 'cat'

    def fit(self, train_X, train_Y, cat_cols):
        self.model = cat.CatBoostClassifier(verbose=50)
        self.model.fit(train_X, train_Y, cat_features=cat_cols)


class LGBModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)
        self.model_name = 'lgb'

    def preprocessing(self, train_X, test_X, cat_cols):
        train_X, test_X = remove_unseen_category(train_X, test_X, cat_cols)
        train_X, test_X = label_encoding(train_X, test_X, cat_cols)
        return train_X, test_X

    def fit(self, train_X, train_Y, cat_cols):
        self.model = lgb.LGBMClassifier()
        self.model.fit(train_X, train_Y, categorical_feature=cat_cols)


class XGBModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)
        self.model_name = 'xgb'

    def preprocessing(self, train_X, test_X, cat_cols):
        train_X, test_X = remove_unseen_category(train_X, test_X, cat_cols)
        train_X, test_X = one_hot_encoding(train_X, test_X, cat_cols)
        return train_X, test_X

    def fit(self, train_X, train_Y, cat_cols):
        self.model = xgb.XGBClassifier()
        self.model.fit(train_X, train_Y)


class MLPModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)


class DeepFMModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)


class XDeepFMModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)


class TabNetModel(BaseModel):
    def __init__(self, dataset, dataset_path):
        BaseModel.__init__(self, dataset, dataset_path)