import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re
from utils import *
from time import time
import torch
from torch.optim import Adam


# TODO: add parameter tuning by hyperopt

class BaseModel():
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        # params related to model and control
        self.device = 'cpu'

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.dataset_type = None
        self.timeseries = False

        self.sample = sample
        self.sample_num = None
        if self.sample:
            self.sample_num = sample_num
        self.split_method = split_method  # TODO: take more splitting methods into consideration
        self.use_category = use_category

        self.model_name = None
        self.model = None

    def fit(self, train_X, train_Y, val_X, val_Y, cat_cols):
        raise NotImplementedError('Method train is not implemented.')

    def test(self, train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols):
        if self.dataset_type in {'2-class', 'm-class'}:
            train_Y_proba, train_Y_hat = self.model.predict_proba(train_X), self.model.predict(train_X)
            val_Y_proba, val_Y_hat = self.model.predict_proba(val_X), self.model.predict(val_X)
            test_Y_proba, test_Y_hat = self.model.predict_proba(test_X), self.model.predict(test_X)
            labels = self.model.classes_
            print(f'{self.model_name} Default Logloss values on train/val/test set: {log_loss(train_Y, train_Y_proba, labels=labels):.5f}/{log_loss(val_Y, val_Y_proba, labels=labels):.5f}/{log_loss(test_Y, test_Y_proba, labels=labels):.5f}')
            print(f'{self.model_name} Default accuracy on train/val/test set: {np.mean(train_Y == train_Y_hat):.5f}/{np.mean(val_Y == val_Y_hat):.5f}/{np.mean(test_Y == test_Y_hat):.5f}')
        elif self.dataset_type in {'regression'}:
            train_Y_hat = self.model.predict(train_X)
            val_Y_hat = self.model.predict(val_X)
            test_Y_hat = self.model.predict(test_X)
            print(f'{self.model_name} Default MAE values on train/val/test set: {mae(train_Y, train_Y_hat):.5f}/{mae(val_Y, val_Y_hat):.5f}/{mae(test_Y, test_Y_hat):.5f}')

    def preprocessing(self, train_X, val_X, test_X, cat_cols):
        return train_X, val_X, test_X

    def run(self):
        print('loading data...')
        train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols, columns, self.dataset_type, self.timeseries = \
            data_loader(self.dataset, self.dataset_path, self.sample, self.sample_num, self.split_method)

        print('preprocessing data...')
        train_X, val_X, test_X = self.preprocessing(train_X, val_X, test_X, cat_cols)

        # TODO: add parameter tuning by hyperopt

        print('training...')
        # TODO: take train/val/test into consideration
        t1 = time()
        self.fit(train_X, train_Y, val_X, val_Y, cat_cols=cat_cols)
        t2 = time()

        print('testing...')
        t3 = time()
        self.test(train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols)
        t4 = time()
        print(f'training/testing time: {(t2 - t1):.2f}/{(t4 - t3):.2f} s')
