import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# TODO: use dict

def data_loader(dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2'):
    """
    return train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols, columns, dataset_type, timeseries.
    """
    X, Y, cat_cols, columns, dataset_type, timeseries = np.load(os.path.join(dataset_path, f'{dataset}.npy'), allow_pickle=True)
    if sample:
        idx = list(range(X.shape[0]))
        if not timeseries:
            random.shuffle(idx)
        X, Y = X[idx[:sample_num]], Y[idx[:sample_num]]
    if split_method == '6-2-2':
        if not timeseries:
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=42)  # 42 is stable for reproducibilty
            val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y, test_size=0.5, random_state=42)
        else:
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, shuffle=False)  # 42 is stable for reproducibilty
            val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y, test_size=0.5, shuffle=False)
    # remove unseen label for imbalance data
    # label_set = set(train_Y)
    # val_idx = [True if y in label_set else False for y in val_Y]
    # val_X, val_Y = val_X[val_idx], val_Y[val_idx]
    # test_idx = [True if y in label_set else False for y in test_Y]
    # test_X, test_Y = test_X[test_idx], test_Y[test_idx]
    return train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols, columns, dataset_type, timeseries

def remove_unseen_category(train_X, val_X, test_X, cat_cols):
    """
    remove category in test set unseen in train set, if we need to encode category.
    """
    for cat_col in cat_cols:
        print(cat_col)
        unseen_set = (set(val_X[:, cat_col]) | set(test_X[:, cat_col])) - set(train_X[:, cat_col])
        for unseen in unseen_set:
            val_X[:, cat_col] = np.where(val_X[:, cat_col] == unseen, 'UNK', val_X[:, cat_col])
            test_X[:, cat_col] = np.where(test_X[:, cat_col] == unseen, 'UNK', test_X[:, cat_col])            
    return train_X, val_X, test_X


def label_encoding(train_X, val_X, test_X, cat_cols):
    """
    return train_X, test_X in ndarray format.
    """
    encoder = LabelEncoder()
    for cat_col in cat_cols:
        encoder.fit(list(train_X[:, cat_col]) + ['UNK'])
        val_X[:, cat_col] = encoder.transform(val_X[:, cat_col])
        test_X[:, cat_col] = encoder.transform(test_X[:, cat_col])
        train_X[:, cat_col] = encoder.transform(train_X[:, cat_col])
    
    return train_X, val_X, test_X


def one_hot_encoding(train_X, val_X, test_X, cat_cols):
    """
    return train_X, test_X in ndarray format.
    """
    encoder = OneHotEncoder()
    encoder.fit(np.concatenate((train_X[:, cat_cols], np.array([['UNK'] * len(cat_cols)]))))
    train_X_cat, test_X_cat = encoder.transform(train_X[:, cat_cols]), encoder.transform(test_X[:, cat_cols])
    val_X_cat = encoder.transform(val_X[:, cat_cols])
    train_X = np.concatenate((np.delete(train_X, cat_cols, axis=1), train_X_cat.A), axis=1)
    val_X = np.concatenate((np.delete(val_X, cat_cols, axis=1), val_X_cat.A), axis=1)
    test_X = np.concatenate((np.delete(test_X, cat_cols, axis=1), test_X_cat.A), axis=1)
    return train_X, val_X, test_X


def scaling(train_X, val_X, test_X, cat_cols):
    if train_X.shape[1] < len(cat_cols):
        mms = MinMaxScaler(feature_range=(0, 1))
        num_cols = [idx for idx in range(train_X.shape[1]) if idx not in cat_cols]
        train_X[:, num_cols] = mms.fit_transform(train_X[:, num_cols])
        val_X[:, num_cols] = mms.transform(val_X[:, num_cols])
        test_X[:, num_cols] = mms.transform(test_X[:, num_cols])
    return train_X, val_X, test_X


def remove_num_nan(train_X, test_X, num_cols):
    pass


if __name__ == "__main__":
    pass