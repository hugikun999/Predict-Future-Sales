import argparse
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import autograd
from SaleDataset import SaleDataset
from Model import LSTM, CNN

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Testing', action='store_true')
    parser.add_argument('--DataRoot', default='data.pkl', type=str)
    parser.add_argument('--TestRoot', default='data/test.csv', type=str)
    parser.add_argument('--SaveName', default='params/lstm', type=str)
    parser.add_argument('--LoadParams', default='params/lstm_Best.params', type=str)

    parser.add_argument('--model', default='lstm', type=str)
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--BatchSize', default=500, type=int)
    parser.add_argument('--TrainEpoch', default=5, type=int)
    parser.add_argument('--ValStep', default=1, type=int)
    parser.add_argument('--SaveStep', default=1, type=int)
    parser.add_argument('--LogStep', default=50, type=int)
    return parser.parse_args()

def GetModelCtx(args):
    if args.gpus < 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpus)

    if args.model =='lstm':
        model = LSTM()
    elif args.model =='cnn':
        model = CNN()
    model.initialize()
    model.collect_params().reset_ctx(ctx)
    print('Finish Model building...')
    return model, ctx

def Validate(model, Val_iter):
    errors = []
    for idx, (x, y) in enumerate(Val_iter):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx).asnumpy().reshape(-1)
        pred = model(x).asnumpy().reshape(-1)
        error = (pred - y) ** 2
        errors.append(error)
    errors = np.sqrt(np.mean(np.concatenate(errors)))
    return errors

def Train(model, args, FList):
    TrainDataset = SaleDataset(args.DataRoot, FList, None)
    ValDataset = SaleDataset(args.DataRoot, FList, None, Training=False)
    Train_iter = mx.gluon.data.DataLoader(TrainDataset, batch_size=args.BatchSize, shuffle=True)
    Val_iter = mx.gluon.data.DataLoader(ValDataset, batch_size=args.BatchSize, shuffle=False)
    Trainer = mx.gluon.Trainer(model.collect_params(), 'adam',
                               {'learning_rate': 1e-4})
    L2loss = mx.gluon.loss.L2Loss()

    print('Start training...')
    BestRMSE = np.inf
    Bestepoch = -1
    for epoch in range(args.TrainEpoch):
        for idx, (x, y) in enumerate(Train_iter):
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)

            with autograd.record():
                pred = model(x)
                pred = pred.reshape(y.shape)
                loss = L2loss(pred, y)
            loss.backward()
            mx.nd.waitall()
            Trainer.step(x.shape[0])

            if (idx + 1) % args.LogStep == 0:
                RMSE = mx.nd.mean(loss).asscalar()
                print('Epoch[{}]Iter[{}]\tRMSE:\t{}\tBest Epoch[{}]:\t{}'.format(epoch+1, idx+1, RMSE,
                                                                                 Bestepoch, BestRMSE))

        if (epoch + 1) % args.ValStep == 0:
            RMSE = Validate(model, Val_iter)
            print('Epoch[{}]Val RMSE:\t{}\tBest RMSE:\t{}'.format(epoch+1, RMSE, BestRMSE))
            if RMSE < BestRMSE:
                BestRMSE = RMSE
                Bestepoch = epoch + 1
                SaveName = args.SaveName + '_Best.params'
                model.save_parameters(SaveName)

        if (epoch + 1) % args.SaveStep == 0:
            SaveName = args.SaveName + '_{}.params'.format(epoch + 1)
            model.save_parameters(SaveName)

def Test(model, args, FList):
    TestDataset = SaleDataset(args.DataRoot, FList, args.TestRoot, Training=False, Testing=True)
    Test_iter = mx.gluon.data.DataLoader(TestDataset, batch_size=args.BatchSize, shuffle=False)
    Preds = []
    for idx, x in enumerate(Test_iter):
        x = x.as_in_context(ctx)

        pred = model(x).asnumpy().reshape(-1)
        Preds.append(pred)
    Preds = np.concatenate(Preds)

    submission = pd.DataFrame({
        "ID": TestDataset.test_Index,
        "item_cnt_month": Preds
    })
    submission.to_csv('lstm_submission.csv', index=False)

if __name__=='__main__':
    args = process_args()
    model, ctx = GetModelCtx(args)

    FeatureList = ['date_block_num','shop_id','item_id','item_cnt_month','city_code','item_category_id',
    'type_code','subtype_code','item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3',
    'item_cnt_month_lag_6','item_cnt_month_lag_12','date_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3','date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12','date_shop_avg_item_cnt_lag_1','date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3','date_shop_avg_item_cnt_lag_6','date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1','date_shop_cat_avg_item_cnt_lag_1','date_shop_type_avg_item_cnt_lag_1',
    'date_shop_subtype_avg_item_cnt_lag_1','date_city_avg_item_cnt_lag_1','date_item_city_avg_item_cnt_lag_1',
    'date_type_avg_item_cnt_lag_1','date_subtype_avg_item_cnt_lag_1','delta_price_lag','month','days',
    'item_shop_last_sale','item_last_sale','item_shop_first_sale','item_first_sale']
    if args.Testing == False:
        Train(model, args, FeatureList)
    else:
        model.load_parameters(args.LoadParams)
        print("Finish load parameters...")
        Test(model, args, FeatureList)