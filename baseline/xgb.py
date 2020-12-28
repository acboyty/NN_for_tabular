from base import BaseModel


class XGBModel(BaseModel):
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        BaseModel.__init__(self, dataset, dataset_path, sample, sample_num, split_method, use_category)
        self.model_name = 'xgb'

    def preprocessing(self, train_X, val_X, test_X, cat_cols):
        if self.use_category:
            train_X, val_X, test_X = remove_unseen_category(train_X, val_X, test_X, cat_cols)
            train_X, val_X, test_X = one_hot_encoding(train_X, val_X, test_X, cat_cols)
        else:
            train_X = np.delete(train_X, cat_cols, axis=1)
            val_X = np.delete(val_X, cat_cols, axis=1)
            test_X = np.delete(test_X, cat_cols, axis=1)
        return train_X, val_X, test_X

    def fit(self, train_X, train_Y, val_X, val_Y, cat_cols):
        if self.dataset_type in {'2-class', 'm-class'}:
            self.model = xgb.XGBClassifier(n_estimators=1000)
        elif self.dataset_type in {'regression'}:
            self.model = xgb.XGBRegressor(n_estimators=1000)
        self.model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)], early_stopping_rounds=200)

