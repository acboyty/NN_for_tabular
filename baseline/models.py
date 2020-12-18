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
from utils import data_loader


class BaseModel():
    def __init__(self):
        pass

    def preprocessing(self, train_X, test_X, type, cat_cols):
        """
        type
        1 - remove category in test set unseen in train set, if we need to encode category
        2 - 
        """
        if type == 1:
            for cat_col in cat_cols:
                unseen_set = list(set(test_X[cat_col]) - set(train_X[cat_col]))
                test_X[cat_col].replace(unseen_set, 'UNK', inplace=True)
        elif type == 2:
            pass

    def fit(self):
        pass

    def test(self):
        pass

    def run(self):
        pass


class CATModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)


class LGBModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

    def run(self):
        pass


class XGBModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)


class MLPModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)


class DeepFMModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)


class XDeepFMModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)


class TabNetModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)