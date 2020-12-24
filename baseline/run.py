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
import torch
import random

if __name__ == '__main__':
    # hyperparams
    SEED = 2021
    # Adult, Amazon, Click prediction, KDD appetency, KDD churn, KDD upselling, HIGGS, KDD internet, Kick prediction
    # San Francisco,
    # Rossmann,
    DATASET = 'Kick prediction'
    DATASET_PATH = "/home/v-tyan/NN_for_tabular/datasets/"
    MODEL = 'tabnet'  # cat, lgb, xgb, mlp, deepfm, tabnet
    SAMPLE = False
    SAMPLE_NUM = 50000
    SPLIT_METHOD = '6-2-2'  # TODO: add more splitting methods
    USE_CATEGORY = False  # TODO: other methods besides xgb

    # reproduce
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # choose model
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
    elif MODEL == 'tabnet':
        from models import TabNetModel
        Model = TabNetModel

    # run
    model = Model(DATASET, DATASET_PATH, sample=SAMPLE, sample_num=SAMPLE_NUM,
                  split_method=SPLIT_METHOD, use_category=USE_CATEGORY)
    model.run()
