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


if __name__ == '__main__':
    """hyperparams"""
    SEED = 2020
    # Adult, Amazon, Click prediction, KDD appetency, KDD churn, KDD internet, KDD upselling, KDD 98, Kick prediction
    DATASET = 'Click prediction'  
    DATASET_PATH = "/home/v-tyan/NN_for_tabular/datasets/"
    MODEL = 'lgb'  # xgb, lgb, cat, mlp, deepfm, xdeepfm, tabnet

    # reproduce
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    elif MODEL == 'xdeepfm':
        from models import XDeepFMModel
        Model = XDeepFMModel
    elif MODEL == 'tabnet':
        from models import TabNetModel
        Model = TabNetModel
    
    # run
    model = Model(DATASET, DATASET_PATH)
    model.run()    
