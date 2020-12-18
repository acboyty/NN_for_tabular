import pandas as pd
import os
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
import re 


def data_loader(dataset, dataset_path):
    """
    return train_X, train_Y, test_X, test_Y, cat_cols, num_cols
    """
    return np.load(os.path.join(dataset_path, f'{dataset}.npy'))
