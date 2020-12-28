from base import BaseModel
import numpy as np 
from utils import *
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor


class TabNetModel(BaseModel):
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        BaseModel.__init__(self, dataset, dataset_path, sample, sample_num, split_method, use_category)
        self.model_name = 'tabnet'

    def preprocessing(self, train_X, val_X, test_X, cat_cols):
        train_X, val_X, test_X = remove_unseen_category(train_X, val_X, test_X, cat_cols)
        train_X, val_X, test_X = label_encoding(train_X, val_X, test_X, cat_cols)
        # tabnet predict_proba func does not accept objet ndarray
        train_X, val_X, test_X = train_X.astype(float), val_X.astype(float), test_X.astype(float)
        # tabnet does not accept np.nan
        train_X = np.where(np.isnan(train_X), 0, train_X)
        val_X = np.where(np.isnan(val_X), 0, val_X)
        test_X = np.where(np.isnan(test_X), 0, test_X)
        return train_X, val_X, test_X

    def fit(self, train_X, train_Y, val_X, val_Y, cat_cols):
        cat_dims = [len(set(train_X[:, idx])) + 1 for idx in cat_cols]
        if self.dataset_type in {'2-class', 'm-class'}:
            self.model = TabNetClassifier(cat_idxs=cat_cols, cat_dims=cat_dims)
            self.model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)], eval_metric=['logloss'], max_epochs=200, patience=20)
        elif self.dataset_type in {'regression'}:
            self.model = TabNetRegressor(cat_idxs=cat_cols, cat_dims=cat_dims)
            self.model.fit(train_X, train_Y[:, np.newaxis], eval_set=[(val_X, val_Y[:, np.newaxis])], eval_metric=['rmse'], max_epochs=200, patience=20)
