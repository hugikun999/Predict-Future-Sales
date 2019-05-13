import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon

class SaleDataset(gluon.data.Dataset):
    def __init__(self, root, feature, testroot, preday=1, PKL=True, Training=True, Testing=False):
        super(SaleDataset, self).__init__()
        self.root = root
        self.testroot = testroot
        self.preday = preday
        self.feature = feature
        self.xfeatureNum = len(self.feature) - 1
        self.yfeatureNum = 1
        self.Training = Training
        self.Testing = Testing
        if PKL == True:
            data = pd.read_pickle(self.root)
            data = data[self.feature]
            self.ProcessData(data)
        else:
            self.data = pd.read_csv(self.root)

    def ProcessData(self, data):
        if self.Training == True:
            train_x = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
            train_y = data[data.date_block_num < 33]['item_cnt_month']
            self.train_x = train_x.values
            self.train_y = train_y.values
        elif self.Testing == True:
            test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
            self.test = test.values
            self.test_Index = pd.read_csv(self.testroot).set_index('ID').index
        else:
            valid_x = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
            valid_y = data[data.date_block_num == 33]['item_cnt_month']
            self.valid_x = valid_x.values
            self.valid_y = valid_y.values

    def __getitem__(self, item):
        if self.Training == True:
            x = self.train_x[item]
            y = [self.train_y[item]]
        elif self.Testing:
            return mx.nd.array(self.test[item]).reshape(self.xfeatureNum, self.preday)
        else:
            x = self.valid_x[item]
            y = [self.valid_y[item]]

        return mx.nd.array(x).reshape(self.xfeatureNum, self.preday), mx.nd.array(y).reshape(1, self.yfeatureNum)

    def __len__(self):
        if self.Training == True:
            return self.train_x.shape[0]
        elif self.Testing == True:
            return self.test.shape[0]
        else:
            return self.valid_x.shape[0]

if __name__=='__main__':
    features = ['date_block_num','shop_id','item_id','item_cnt_month','city_code','item_category_id',
    'type_code','subtype_code','item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3',
    'item_cnt_month_lag_6','item_cnt_month_lag_12','date_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3','date_item_avg_item_cnt_lag_6','date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1','date_shop_avg_item_cnt_lag_2','date_shop_avg_item_cnt_lag_3','date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12','date_cat_avg_item_cnt_lag_1','date_shop_cat_avg_item_cnt_lag_1','date_shop_type_avg_item_cnt_lag_1',
    'date_shop_subtype_avg_item_cnt_lag_1','date_city_avg_item_cnt_lag_1','date_item_city_avg_item_cnt_lag_1','date_type_avg_item_cnt_lag_1',
    'date_subtype_avg_item_cnt_lag_1','delta_price_lag','month','days','item_shop_last_sale',
    'item_last_sale','item_shop_first_sale','item_first_sale']
    TrainDataset = SaleDataset('data.pkl', features)
    Train_iter = mx.gluon.data.DataLoader(TrainDataset, batch_size=10, shuffle=False)
    for idx, (x, y) in enumerate(Train_iter):
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
