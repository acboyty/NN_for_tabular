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


# params
SEED = 2020
DATASET = 'KDD upselling'  # Adult, Amazon, Click prediction, KDD appetency, KDD churn, KDD internet, KDD upselling, KDD 98, Kick prediction
DATASET_PATH = "/home/v-tyan/NN_for_tabular/datasets/"
MODEL = 'lgb'  # xgb, lgb, cat, mlp


# seed
np.random.seed(SEED)


# dataloader - generate train/test_X/Y
if DATASET == 'Adult':
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    target_col = 'income'
    cat_cols = ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'native-country']
    num_cols = list(set(cols) - {target_col} - set(cat_cols))
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'Adult/adult.data'),
                           sep=', ', header=None, names=cols, na_values='?', engine='python')
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'Adult/adult.test'),
                          sep=', ', header=None, names=cols, na_values='?', engine='python')
    test_df.replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)
    train_X, train_Y = train_df.drop(target_col, axis=1), train_df[target_col]
    test_X, test_Y = test_df.drop(target_col, axis=1), test_df[target_col]
elif DATASET == 'Amazon':
    df = pd.read_csv(os.path.join(DATASET_PATH, 'Amazon/train.csv'))
    cols = list(df.columns)
    target_col = 'ACTION'
    cat_cols = list(set(set(cols) - {target_col}))
    num_cols = []  # assume all are categorial
    X, Y = df.drop(target_col, axis=1), df[target_col]
    train_idx = pd.read_csv(os.path.join(DATASET_PATH, "Amazon/stratified_train_idx.txt"), header=None)
    test_idx = pd.read_csv(os.path.join(DATASET_PATH, "Amazon/stratified_test_idx.txt"), header=None)
    train_X, test_X, train_Y, test_Y = X.iloc[train_idx[0]], X.iloc[test_idx[0]], Y.iloc[train_idx[0]], Y.iloc[test_idx[0]]
elif DATASET == 'Click prediction':
    cols = ['click', 'impression', 'url_hash', 'ad_id', 'advertiser_id', 'depth', 'position', 
            'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id']
    target_col = 'click'
    cat_cols = ['impression', 'url_hash', 'ad_id', 'position', 
                'query_id', 'keyword_id', 'title_id', 'description_id']
    num_cols = list(set(cols) - {target_col} - set(cat_cols))
    with open(os.path.join(DATASET_PATH, "Click prediction/track2/subsampling_idx.txt")) as fin:
        ids = list(map(int, fin.read().split()))
    unique_ids = set(ids)
    data_strings = {}
    with open(os.path.join(DATASET_PATH, "Click prediction/track2/training.txt")) as fin:
        for i, string in enumerate(fin):
            if i in unique_ids:
                data_strings[i] = string
    data_rows = []
    for i in ids:
        data_rows.append(data_strings[i])
    df = pd.read_table(StringIO("".join(data_rows)), header=None, names=cols)    
    X, Y = df.drop(target_col, axis=1), df[target_col].apply(lambda x: 1 if x == 0 else -1)
    def clean_string(s):
        return "v_" + re.sub('[^A-Za-z0-9]+', "_", str(s))
    for cat_col in cat_cols:
        X[cat_col] = X[cat_col].apply(clean_string)
    train_idx = pd.read_csv(os.path.join(DATASET_PATH, "Click prediction/track2/stratified_train_idx.txt"), header=None)
    test_idx = pd.read_csv(os.path.join(DATASET_PATH, "Click prediction/track2/stratified_test_idx.txt"), header=None)
    train_X, test_X, train_Y, test_Y = X.iloc[train_idx[0]], X.iloc[test_idx[0]], Y.iloc[train_idx[0]], Y.iloc[test_idx[0]]
elif DATASET in {'KDD appetency', 'KDD churn', 'KDD upselling'}:
    df = pd.read_csv(os.path.join(DATASET_PATH, "appetency_churn_upselling/orange_small_train.data"), sep = "\t")
    cols = list(df.columns)
    cat_cols = [cols[idx] for idx in [190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
                207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228]]
    num_cols = list(set(cols) - set(cat_cols))
    dataset = DATASET.split(' ')[-1]
    X, Y = df, -pd.read_csv(os.path.join(DATASET_PATH, "appetency_churn_upselling/orange_small_train_" + dataset + ".labels"), header=None)[0]
    train_idx = pd.read_csv(os.path.join(DATASET_PATH, f'appetency_churn_upselling/{dataset}/stratified_train_idx_{dataset}.txt'), header=None)
    test_idx = pd.read_csv(os.path.join(DATASET_PATH, f'appetency_churn_upselling/{dataset}/stratified_test_idx_{dataset}.txt'), header=None)
    train_X, test_X, train_Y, test_Y = X.iloc[train_idx[0]], X.iloc[test_idx[0]], Y.iloc[train_idx[0]], Y.iloc[test_idx[0]]
elif DATASET == '':
    pass
# prepare numerical features
train_X[num_cols], test_X[num_cols] = train_X[num_cols].astype(float), test_X[num_cols].astype(float)
# prepare category features
# catgory -> str
for cat_col in cat_cols:
    train_X[cat_col] = train_X[cat_col].apply(str)
    test_X[cat_col] = test_X[cat_col].apply(str)
# remove category in test set unseen in train set, if we need to encode category
if MODEL in {'lgb', 'xgb'}:
    for cat_col in cat_cols:
        unseen_set = list(set(test_X[cat_col]) - set(train_X[cat_col]))
        test_X[cat_col].replace(unseen_set, 'UNK', inplace=True)


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
    model_default = lgb.LGBMClassifier(lgb_default_params)
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
