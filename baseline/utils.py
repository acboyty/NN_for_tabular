import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re


def data_loader(dataset, dataset_path):
    """
    return train_X, train_Y, test_X, test_Y in ndarray format, cat_cols list in int.
    """
    return np.load(os.path.join(dataset_path, f'{dataset}.npy'), allow_pickle=True)


def remove_unseen_category(train_X, test_X, cat_cols):
    """
    remove category in test set unseen in train set, if we need to encode category.
    """
    for cat_col in cat_cols:
        unseen_set = set(test_X[:, cat_col]) - set(train_X[:, cat_col])
        for unseen in unseen_set:
            test_X[:, cat_col] = np.where(test_X[:, cat_col] == unseen, 'UNK', test_X[:, cat_col])
    return train_X, test_X


def label_encoding(train_X, test_X, cat_cols):
    """
    return train_X, test_X in ndarray format.
    """
    encoder = LabelEncoder()
    for cat_col in cat_cols:
        encoder.fit(list(train_X[:, cat_col]) + ['UNK'])
        test_X[:, cat_col], train_X[:, cat_col] = encoder.transform(test_X[:, cat_col]), encoder.transform(train_X[:, cat_col])
    return train_X, test_X


def one_hot_encoding(train_X, test_X, cat_cols):
    """
    return train_X, test_X in ndarray format.
    """
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(np.concatenate((train_X[:, cat_cols], np.array([['UNK'] * len(cat_cols)]))))
    train_X_cat, test_X_cat = encoder.transform(train_X[:, cat_cols]), encoder.transform(test_X[:, cat_cols])
    train_X = np.concatenate((np.delete(train_X, cat_cols, axis=1), train_X_cat), axis=1)
    test_X = np.concatenate((np.delete(test_X, cat_cols, axis=1), test_X_cat), axis=1)
    return train_X, test_X