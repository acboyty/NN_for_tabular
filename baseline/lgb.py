from base import BaseModel

class LGBModel(BaseModel):
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        BaseModel.__init__(self, dataset, dataset_path, sample, sample_num, split_method, use_category)
        self.model_name = 'lgb'

    def preprocessing(self, train_X, val_X, test_X, cat_cols):
        train_X, val_X, test_X = remove_unseen_category(train_X, val_X, test_X, cat_cols)
        train_X, val_X, test_X = label_encoding(train_X, val_X, test_X, cat_cols)
        return train_X, val_X, test_X

    def fit(self, train_X, train_Y, val_X, val_Y, cat_cols):
        if self.dataset_type in {'2-class', 'm-class'}:
            self.model = lgb.LGBMClassifier(n_estimators=1000)
        elif self.dataset_type in {'regression'}:
            self.model = lgb.LGBMRegressor(n_estimators=1000)
        self.model.fit(train_X, train_Y, categorical_feature=cat_cols, eval_set=[(val_X, val_Y)], eval_metric='MAE', early_stopping_rounds=200)
