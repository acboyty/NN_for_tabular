from base import BaseModel
from utils import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, xDeepFM
import torch
from torch.optim import Adam
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from time import time


class DeepFMModel(BaseModel):
    def __init__(self, dataset, dataset_path, sample=False, sample_num=None, split_method='6-2-2', use_category=True):
        BaseModel.__init__(self, dataset, dataset_path, sample, sample_num, split_method, use_category)
        self.model_name = 'deepfm'

    def preprocessing(self, train_X, val_X, test_X, cat_cols):
        train_X, val_X, test_X = remove_unseen_category(train_X, val_X, test_X, cat_cols)
        train_X, val_X, test_X = label_encoding(train_X, val_X, test_X, cat_cols)
        train_X, val_X, test_X = scaling(train_X, val_X, test_X, cat_cols)
        # remove np.nan
        train_X, val_X, test_X = train_X.astype(float), val_X.astype(float), test_X.astype(float)
        train_X = np.where(np.isnan(train_X), 0, train_X)
        val_X = np.where(np.isnan(val_X), 0, val_X)
        test_X = np.where(np.isnan(test_X), 0, test_X)
        return train_X, val_X, test_X
     
    def fit_test(self, train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols):
        sparse_features = cat_cols
        dense_features = [idx for idx in range(train_X.shape[1]) if idx not in cat_cols]
        sparse_feature_columns = [SparseFeat(str(feat), vocabulary_size=len(set(train_X[:, feat]))+1, embedding_dim=4) for i, feat in enumerate(sparse_features)]
        dense_feature_columns = [DenseFeat(str(feat), 1,) for feat in dense_features]
        dnn_feature_columns = sparse_feature_columns + dense_feature_columns
        linear_feature_columns = sparse_feature_columns + dense_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        train_model_input = {name: train_X[:, int(name)] for name in feature_names}
        val_model_input = {name: val_X[:, int(name)] for name in feature_names}
        test_model_input = {name: test_X[:, int(name)] for name in feature_names}
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            self.device = 'cuda:0'
        self.model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=self.device)
        self.model.compile(Adam(self.model.parameters(), 0.0001), "binary_crossentropy", metrics=['binary_crossentropy'], )
        es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=30, mode='min')
        lbe = LabelEncoder()
        self.model.fit(train_model_input, lbe.fit_transform(train_Y), batch_size=512, epochs=21, verbose=2, validation_data=(val_model_input, lbe.transform(val_Y)))
        pred_ans = self.model.predict(test_model_input, batch_size=256)
        print(f'{log_loss(test_Y, pred_ans):.5f}')

    def run(self):
        print('loading data...')
        train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols, columns, self.dataset_type, self.timeseries = \
            data_loader(self.dataset, self.dataset_path, self.sample, self.sample_num, self.split_method)

        print('preprocessing data...')
        train_X, val_X, test_X = self.preprocessing(train_X, val_X, test_X, cat_cols)

        # TODO: add parameter tuning by hyperopt

        print('training & testing...')
        # TODO: take train/val/test into consideration
        t1 = time()
        self.fit_test(train_X, train_Y, val_X, val_Y, test_X, test_Y, cat_cols=cat_cols)
        t2 = time()
        print(f'training & testing time: {(t2 - t1):.2f} s')
