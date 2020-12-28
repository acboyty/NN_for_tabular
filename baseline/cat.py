from base import BaseModel


class CATModel(BaseModel):
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        BaseModel.__init__(self, dataset, dataset_path, sample, sample_num, split_method, use_category)
        self.model_name = 'cat'

    def fit(self, train_X, train_Y, val_X, val_Y, cat_cols):
        if self.dataset_type in {'2-class', 'm-class'}:
            self.model = cat.CatBoostClassifier(verbose=50)
        elif self.dataset_type in {'regression'}:
            self.model = cat.CatBoostRegressor(verbose=50, eval_metric='RMSE')
        self.model.fit(train_X, train_Y, cat_features=cat_cols, eval_set=[(val_X, val_Y)], early_stopping_rounds=200)
