# -*- coding: utf-8 -*-
# author: soysouce

import pandas as pd
import talib
import logging
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, Flatten, Dropout, RNN, TimeDistributed, BatchNormalization
from tensorflow.python.keras.optimizers import RMSprop, Nadam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import time
from sklearn import preprocessing, svm
from tensorflow.python.keras.utils import to_categorical
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras import metrics
from scipy import stats
from scipy.stats import norm, skew
from tensorflow.python.keras import backend as K
from multiprocessing.pool import ThreadPool
from tensorflow.python.keras.utils.vis_utils import plot_model
import xgboost as xgb

standard = StandardScaler()
minmax = MinMaxScaler()


def re_ohlc(input_data, time_period=None):
    df = pd.DataFrame()
    segment = 0
    num = input_data.shape[0]
    a = 0
    if time_period:
        while True:
            if a < num:
                data = input_data.iloc[a:a+time_period, :]
                df.loc[segment, 'time'] = pd.to_datetime(data['timestamp'], unit='ms').iloc[-1]
                df.loc[segment, 'open'] = data['open'].iloc[0]
                df.loc[segment, 'high'] = data['high'].max()
                df.loc[segment, 'low'] = data['low'].min()
                df.loc[segment, 'close'] = data['close'].iloc[-1]
                df.loc[segment, 'volume'] = data['volume'].sum()
                segment += 1
                a += time_period
                print(segment)
            else:
                return df
    else:
        logging.warning('time_period empty')


def re_ohlc2(input_data, time_period=None):
    df = pd.DataFrame()
    segment = 0
    num = input_data.shape[0]
    a = 0
    input_data = input_data.dropna()
    if time_period:
        while True:
            if a < num:
                data = input_data.iloc[a:a+time_period, :]
                df.loc[segment, 'time'] = pd.to_datetime(data['timestamp'], unit='ms').iloc[-1]
                # df.loc[segment, 'time'] = data['time'].iloc[-1]
                df.loc[segment, 'open'] = data['open'].iloc[0]
                df.loc[segment, 'high'] = data['high'].max()
                df.loc[segment, 'low'] = data['low'].min()
                df.loc[segment, 'close'] = data['close'].iloc[-1]
                df.loc[segment, 'volume'] = data['volume'].sum()


                for value in ['signal', 'bbwidth', 'price_to_bblower',
       'price_to_bbupper', 'price_to_bbmiddle', '5m_change', '10m_change',
       '15m_change', '20m_change', '25m_change', '30m_change']:

                    df.loc[segment, '{}_std'.format(value)] = data['{}'.format(value)].std()
                    df.loc[segment, '{}_max'.format(value)] = data['{}'.format(value)].max()
                    df.loc[segment, '{}_min'.format(value)] = data['{}'.format(value)].min()
                    df.loc[segment, '{}_q95'.format(value)] = np.quantile(data['{}'.format(value)], 0.95)
                    df.loc[segment, '{}_q99'.format(value)] = np.quantile(data['{}'.format(value)], 0.99)
                    df.loc[segment, '{}_q75'.format(value)] = np.quantile(data['{}'.format(value)], 0.75)
                    df.loc[segment, '{}_q25'.format(value)] = np.quantile(data['{}'.format(value)], 0.25)
                    df.loc[segment, '{}_q05'.format(value)] = np.quantile(data['{}'.format(value)], 0.05)
                    df.loc[segment, '{}_q01'.format(value)] = np.quantile(data['{}'.format(value)], 0.01)
                    df.loc[segment, '{}_abs_max'.format(value)] = np.abs(data['{}'.format(value)]).max()
                    df.loc[segment, '{}_abs_mean'.format(value)] = np.abs(data['{}'.format(value)]).mean()
                    df.loc[segment, '{}_abs_std'.format(value)] = np.abs(data['{}'.format(value)]).std()
                    df.loc[segment, '{}_kurtosis'.format(value)] = data['{}'.format(value)].kurtosis()
                    df.loc[segment, '{}_skew'.format(value)] = data['{}'.format(value)].skew()
                    df.loc[segment, '{}_mad'.format(value)] = data['{}'.format(value)].mad()
                    df.loc[segment, '{}_sem'.format(value)] = data['{}'.format(value)].sem()


                segment += 1
                a += time_period
                print(segment)
            else:
                return df
    else:
        logging.warning('time_period empty')


def feature_extract(input_data):
    input_data['bbupper'], input_data['bbmiddle'], input_data['bblower'] = talib.BBANDS(input_data.close, 30, 2, 0)
    input_data['fast'], input_data['slow'], input_data['signal'] = talib.MACD(input_data.close, 15, 30, 10)

    input_data['bbwidth'] = input_data['bbupper'] - input_data['bblower']
    input_data['price_to_bblower'] = abs(input_data['bblower'] - input_data['close'])
    input_data['price_to_bbupper'] = abs(input_data['bbupper'] - input_data['close'])
    input_data['price_to_bbmiddle'] = abs(input_data['close'] - input_data['bbmiddle'])
    return input_data




def feature_extract2(input_data):
    input_data['5m_change'] = input_data['close'].pct_change(1)
    input_data['10m_change'] = input_data['close'].pct_change(2)
    input_data['15m_change'] = input_data['close'].pct_change(3)
    input_data['20m_change'] = input_data['close'].pct_change(4)
    input_data['25m_change'] = input_data['close'].pct_change(5)
    input_data['30m_change'] = input_data['close'].pct_change(6)

    return input_data


def selfprocessing(train ,test, target):
    scaled_train = standard.fit_transform(train.values)
    scaled_test = standard.fit_transform(test.values)
    # scaled_test = standard.fit_transform(test.drop(columns='seg_id').values)
    # target = minmax.fit_transform(target)

    X_train, X_test, y_train, y_test = train_test_split(scaled_train, target.values, test_size=0.2, shuffle=False)
    # print(y_train.shape)
    #
    # reshape_X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # reshape_X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #
    # # reshape_y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
    # # reshape_y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))
    #
    # # reshape_y_train = np.reshape(y_train, (y_train.shape[0], 1))
    # # reshape_y_test = np.reshape(y_test, (y_test.shape[0], 1))
    #
    # reshape_scaled_train = np.reshape(scaled_train, (scaled_train.shape[0], 1, scaled_train.shape[1]))
    # # reshape_scaled_test = np.reshape(scaled_test, (scaled_test.shape[0], 1, scaled_test.shape[1]))
    # # reshape_target = np.reshape(target, (target.shape[0], 1))
    #
    # validation = np.reshape(scaled_test, (scaled_test.shape[0], 1, scaled_test.shape[1]))

    return  X_train, X_test, y_train, y_test,scaled_train, scaled_test


def _model(train ,test, target):
    optimizer = Nadam(lr=0.0002)
    reshape_X_train, reshape_X_test, reshape_y_train, reshape_y_test, reshape_scaled_train, validation = selfprocessing(train ,test, target)

    num = train.shape[1]
    print(num)
    model = Sequential()
    model.add(LSTM(11, activation='linear', return_sequences=True, input_shape=(1, num)))
    model.add(Dropout(0.4))
    model.add(LSTM(22, activation='linear', return_sequences=True, input_shape=(1, num)))
    model.add(Dropout(0.4))
    model.add(LSTM(44, activation='linear', return_sequences=True, input_shape=(1, num)))
    model.add(Dropout(0.4))
    model.add(LSTM(88, activation='linear', input_shape=(1, num)))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])
    result = model.fit(reshape_X_train, reshape_y_train,
                       epochs=30000, shuffle=False,
                       callbacks=[EarlyStopping(monitor='loss', min_delta=0.001, mode='min', patience=30)],
                       batch_size=1280)
    evaluate = model.evaluate(reshape_X_test, reshape_y_test)
    pred = model.predict(validation)
    old = model.predict(reshape_scaled_train)
    print(minmax.inverse_transform(pred))
    print(evaluate)

    plt.figure(1)
    plt.plot(result.history['loss'])
    plt.title('loss' + str(round(result.history['loss'][-1], 6)))
    plt.show()
    plt.figure(2)
    plt.plot(minmax.inverse_transform(old))
    plt.plot(minmax.inverse_transform(target))
    plt.show()
    plt.figure(3)
    plt.plot(minmax.inverse_transform(pred))
    plt.show()


# 经过对比，1小时线在任何情况下穿过布林中线，24~48小时后上升的概率都很高
def bbband_feature_extract(resample_data, brust=True, label=None):
    featured_data = feature_extract(resample_data)
    brust_con = (featured_data['open'] < featured_data['bbmiddle']) & (
                featured_data['close'] > featured_data['bbmiddle'])
    drop_con = (featured_data['open'] > featured_data['bbmiddle']) & (
                featured_data['close'] < featured_data['bbmiddle'])

    if brust:
        con = brust_con
    else:
        con = drop_con


    select_feature = featured_data.loc[con, 'close']

    for i, num in enumerate([1, 2, 3,4,5,6,7,8]):
        select_target = featured_data.loc[con].shift(-num)['close']

        df = pd.DataFrame(data={'close': select_feature, 'close_24': select_target,
                                'change': select_feature - select_target}).dropna().reset_index(drop=True)
        final = df.change / df.change.abs()

        print(f'在{num*4}小时之后:....')
        print(final.groupby(final).count())
        print('.............')
        plt.subplot(4, 2, i + 1)
        sns.kdeplot(final.dropna(),label=label)
        plt.title(f'after{num*4}hours later:....')
        plt.legend()




#查看总数据24小时之后的涨跌分布，确定了数据上没有偏见
def hour_24_change():
    resample_data = pd.read_csv('bitfinex_btc_1h.csv')
    resample_data['target'] = resample_data['close'].shift(-24)
    data = resample_data['close'] - resample_data['target']
    final = (data/abs(data)).dropna()
    print(final.groupby(final).count())
    sns.kdeplot(final.dropna())
    plt.show()

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def address_analyze(dir):
    datalist = os.listdir(dir)
    df = pd.DataFrame(columns=['in', 'out', 'balance'])


    for file in datalist:

        if len(file) == 38:

            file_data = pd.read_csv(dir+'/'+file).dropna()

            # 每日入金总额
            for date, num in file_data.groupby('date (UTC)')['moneyIn (BTC)'].sum().iteritems():

                if date in df.index and not df.isnull().loc[date, 'in']:
                    df.loc[date, 'in'] += num
                else:
                    df.loc[date, 'in'] = num


            # 每日出金总额
            for date, num in file_data.groupby('date (UTC)')['moneyOut (BTC)'].sum().iteritems():

                if date in df.index and not df.isnull().loc[date, 'out']:
                    df.loc[date,'out'] += num

                else:
                    df.loc[date, 'out'] = num

            # 当日账户余额
            for date, num in file_data.groupby('date (UTC)')['current balance (BTC)'].first().iteritems():
                if date in df.index and not df.isnull().loc[date, 'balance']:
                    df.loc[date,'balance'] += num
                else:
                    df.loc[date, 'balance'] = num

    return df


def draw(column1, column2, final, trend=None):
    if trend:
        fig, (ax1, ax3) = plt.subplots(2,1)
        ax2 = ax1.twinx()
        ax1.plot(abs(final[f'{column1}{trend}'][:-1]),color='r')

        ax1.set_title('count-price compare')
        ax2.plot(abs(final[f'{column2}{trend}'][:-1]), color='b')
        plt.grid()
        plt.legend()


        ax4 = ax3.twinx()
        ax3.plot((final[f'{column1}{trend}'][:-1]),color='r')
        ax4.plot((final[f'{column2}{trend}'][:-1]), color='b')
        plt.grid()
        plt.legend()

    else:
        fig, (ax1, ax3) = plt.subplots(2, 1)
        ax2 = ax1.twinx()
        ax1.plot(abs(final[f'{column1}'][:-1]), color='r')

        ax1.set_title('count-price compare')
        ax2.plot(abs(final[f'{column2}'][:-1]), color='b')
        plt.grid()
        plt.legend()

        ax4 = ax3.twinx()
        ax3.plot((final[f'{column1}'][:-1]), color='r')
        ax4.plot((final[f'{column2}'][:-1]), color='b')
        plt.grid()
        plt.legend()

def draw1(final):
    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax1.plot(final[['slow', 'fast']])
    plt.grid()
    plt.legend()

    ax4 = ax3.twinx()
    ax3.plot(final[['c_slow', 'c_fast']])
    plt.grid()
    plt.legend()

def draw2(final):
    fig, (ax1, ax3) = plt.subplots(2,1)
    ax2 = ax1.twinx()
    ax1.plot(abs(final['close'][:-1]),color='r')
    ax2.plot(abs(final['transaction_count'][:-1]), color='b')
    plt.grid()
    plt.legend()


    ax4 = ax3.twinx()
    ax3.plot((final['close'][:-1]),color='r')
    ax4.plot((final['transaction_count'][:-1]), color='b')
    plt.grid()
    plt.legend()


def final_data(rate='d', column='transaction_count'):
    data = pd.read_csv('bitfinex_updated.csv')
    data.index = pd.to_datetime(data['timestamp'], unit='ms')

    day = data['close'].resample(rate, 'ohlc')
    block_time = pd.read_csv('blocks_time.csv')
    block_size = pd.read_csv('blocks_size.csv')
    block_count = pd.read_csv('blocks_count.csv')
    block_input = pd.read_csv('input_count.csv')
    block_output = pd.read_csv('output_count.csv')
    input_usd = pd.read_csv('input_usd.csv')
    output_usd = pd.read_csv('output_usd.csv')
    input_btc = pd.read_csv('input_btc.csv')
    output_btc = pd.read_csv('output_btc.csv')
    block_df = pd.concat(
        [block_time, block_count, block_size, block_input, block_output, input_usd, input_btc, output_btc, output_usd],
        axis=1)
    block_df.index = pd.to_datetime(block_df.time)
    block_df = block_df.drop(columns='time')
    df = block_df.resample(rate).sum()

    final = pd.concat([df, day], axis=1).dropna()
    final['close_change'] = final['close'].pct_change(1)
    final['count_change'] = final['transaction_count'].pct_change(1)
    final['size_change'] = final['size'].pct_change(1)
    test = final.dropna()
    final[f'{column}_fast'], final[f'{column}_slow'], final[f'{column}_signal'] = talib.MACD(final[column], 15, 30, 10)
    final['close_fast'], final['close_slow'], final['close_signal'] = talib.MACD(final.close, 15, 30, 10)
    final[f'{column}_RSI'] = talib.RSI(final[column], 15)
    final['close_RSI'] = talib.RSI(final.close, 15)

    return final.dropna()

def percent_change(data, forward=True):
    data = data.dropna()
    if forward:
        for column in data.columns:
            for day in range(1, 6):
                data[f'{column}_{day}change'] = data[column].pct_change(day)
    else:
        for column in data.columns:
            for day in range(-5, 0):
                data[f'{column}_{day}change'] = data[column].pct_change(day)

    return data.dropna()

def dense_model():
    # data = final_data(rate='d', column='transaction_count')
    # draw('slow', data)
    bitfinex = pd.read_csv('bitfinex_updated.csv')

    block_time = pd.read_csv('blocks_time.csv')
    block_size = pd.read_csv('blocks_size.csv')
    block_count = pd.read_csv('blocks_count.csv')
    block_input = pd.read_csv('input_count.csv')
    block_output = pd.read_csv('output_count.csv')
    input_usd = pd.read_csv('input_usd.csv')
    output_usd = pd.read_csv('output_usd.csv')
    input_btc = pd.read_csv('input_btc.csv')
    output_btc = pd.read_csv('output_btc.csv')
    block_df = pd.concat(
        [block_time, block_count, block_size, block_input, block_output, input_usd, input_btc, output_btc, output_usd],
        axis=1)
    block_df.index = pd.to_datetime(block_df.time)
    block_df = block_df.drop(columns='time')
    block_df = block_df.resample('d').sum()

    for column in block_df.columns:
        block_df[f'{column}_fast'], block_df[f'{column}_slow'], block_df[f'{column}_signal'] = talib.MACD(block_df[column], 15, 30, 10)

    block_df = percent_change(block_df, forward=False)
    bitfinex['timestamp'] = pd.to_datetime(bitfinex['timestamp'], unit='ms')
    bitfinex = bitfinex.set_index('timestamp', drop=True)

    bitfinex = bitfinex.resample('d').last()

    target = bitfinex.close.pct_change(1)
    target = pd.get_dummies(target.dropna() / abs(target.dropna()))
    target.columns = ['fall', 'up']

    train = pd.concat([block_df, target], axis=1).dropna()
    standard = StandardScaler()
    xtrain, ytrain, xtest, ytest = train_test_split(standard.fit_transform(train.drop(columns=['fall', 'up']).values),
                                                    target.values, test_size=0.2)

    # xtrain = standard.fit_transform(train.drop(columns='up').values)
    # xtest =  train['up'].values
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(xtrain))
    test_predictions = np.zeros(len(xtest))

    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(xtrain.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model = Sequential()
    # model.add(Dense(80, activation='relu', input_shape=(xtrain.shape[1],)))
    # model.add(BatchNormalization())
    # model.add(Dense(60, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(40, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(20, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(10, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain, xtest)):
        X_train, X_test = xtrain[trn_idx], xtrain[val_idx]
        y_train, y_test = xtest[trn_idx], xtest[val_idx]

        model.fit(X_train, y_train, epochs=300, batch_size=1,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0.001, mode='auto', patience=5)])
        print('内部数据验证')
        print(model.evaluate(X_test, y_test))
        print('外部数据验证')
        print(model.evaluate(ytrain, ytest))
    pred = model.predict(ytrain)
    from sklearn import metrics

    print('AUC: %.4f' % metrics.roc_auc_score(ytest, pred))
    print('ACC: %.4f' % metrics.accuracy_score(ytest, pred))
    print('Recall: %.4f' % metrics.recall_score(ytest, pred))
    print('F1-score: %.4f' % metrics.f1_score(ytest, pred))
    print('Precesion: %.4f' % metrics.precision_score(ytest, pred))

def xgb_model():

    # data = final_data(rate='d', column='transaction_count')
    # draw('slow', data)
    bitfinex = pd.read_csv('bitfinex_updated.csv')

    block_time = pd.read_csv('blocks_time.csv')
    block_size = pd.read_csv('blocks_size.csv')
    block_count = pd.read_csv('blocks_count.csv')
    block_input = pd.read_csv('input_count.csv')
    block_output = pd.read_csv('output_count.csv')
    input_usd = pd.read_csv('input_usd.csv')
    output_usd = pd.read_csv('output_usd.csv')
    input_btc = pd.read_csv('input_btc.csv')
    output_btc = pd.read_csv('output_btc.csv')
    block_df = pd.concat([block_time, block_count, block_size, block_input, block_output,input_usd,input_btc, output_btc, output_usd], axis=1)
    block_df.index = pd.to_datetime(block_df.time)
    block_df = block_df.drop(columns='time')
    block_df = block_df.resample('d').sum()

    for column in block_df.columns:
        block_df[f'{column}_fast'], block_df[f'{column}_slow'],block_df[ f'{column}_signal'] = talib.MACD(block_df[column], 15, 30 ,10)
    #
    block_df = percent_change(block_df, forward=False)
    bitfinex['timestamp'] = pd.to_datetime(bitfinex['timestamp'], unit='ms')
    bitfinex = bitfinex.set_index('timestamp', drop=True)

    bitfinex = bitfinex.resample('d').last()

    target = bitfinex.close.pct_change(1)
    target = pd.get_dummies(target.dropna() / abs(target.dropna()))
    target.columns = ['fall', 'up']
    target = target.drop(columns='fall')

    train = pd.concat([block_df, target], axis=1).dropna()
    standard = StandardScaler()
    xtrain, ytrain, xtest, ytest = train_test_split(standard.fit_transform(train.drop(columns='up').values), train['up'].values, test_size=0.2)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(xtrain))
    test_predictions = np.zeros(len(xtest))

    params ={}
    model = xgb.XGBClassifier(n_estimators=999999,learning_rate=0.025,colsample_bytree=0.75,subsample=0.75,max_depth=2)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain, xtest)):
        X_train, X_test = xtrain[trn_idx], xtrain[val_idx]
        y_train, y_test = xtest[trn_idx], xtest[val_idx]
        watch_list = [(X_test,y_test)]
        model.fit(X_train, y_train, eval_metric=['auc'],eval_set=watch_list,early_stopping_rounds=10000,verbose=0)
        pred = model.predict(ytrain)

        from sklearn import metrics

        print('AUC: %.4f' % metrics.roc_auc_score(ytest, pred[0:]))
        print('ACC: %.4f' % metrics.accuracy_score(ytest, pred[0:]))
        print('Recall: %.4f' % metrics.recall_score(ytest, pred[0:]))
        print('F1-score: %.4f' % metrics.f1_score(ytest, pred[0:]))
        print('Precesion: %.4f' % metrics.precision_score(ytest, pred[0:]))


def block_info():
    import requests
    for coin in ['bitcoin']:
        blocks_urls = [f'https://api.blockchair.com/{coin}/blocks?fields=size&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=time&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=transaction_count&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=input_count&s=id(desc)&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=output_count&s=id(desc)&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=input_total&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=input_total_usd&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=output_total&export=csv',
                       f'https://api.blockchair.com/{coin}/blocks?fields=output_total_usd&export=csv']

        file_names = ['blocks_size.csv', 'blocks_time.csv', 'blocks_count.csv',
                      'input_count.csv', 'output_count.csv', 'input_btc.csv',
                      'input_usd.csv', 'output_btc.csv', 'output_usd.csv']

        for url, file_name in zip(blocks_urls, file_names):
            data = requests.get(url)
            with open(file_name, 'wb') as f:
                f.write(data.content)
                print(f'finish  {file_name}')

def bitfinex_info():
    import requests
    t = time.time()
    now = lambda: int(round(t * 1000))

    def test(url, symbol, interval, start):
        proxies = {
            'https': 'https://127.0.0.1:1080',
            'http': 'http://127.0.0.1:1080'
        }
        url = url + '{interval}:{symbol}/hist'.format(interval=interval, symbol=symbol)
        params = {'limit': 5000, 'start': start, 'sort': 1}
        data = requests.get(url=url, params=params, proxies=proxies)

        return data.json()

    def bitfinex_csv(symbol):
        bitfinex = 'https://api.bitfinex.com/v2/candles/trade:'
        start = 1550789582000
        end = now()
        df = pd.DataFrame()

        while start < end:
            data = test(bitfinex, symbol, '1m', start)
            new_df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high',
                                                 'low', 'volume'])
            df = pd.concat([df, new_df])
            start = int(data[-1][0])
            print('processing....')
            print(start)
            time.sleep(10)

        return df

    old = pd.read_csv('D:/MLcrpto/bitfinex_btc_201936.csv')
    new = bitfinex_csv('tBTCUSD')
    return old, new
if __name__=='__main__':
    block_info()
    old, new = bitfinex_info()
    old = old.drop(columns='Unnamed: 0')
    df = pd.concat([old,new]).drop_duplicates('timestamp').reset_index(drop=True)
    df.to_csv('bitfinex_updated.csv', index=0)
    column1 = 'transaction_count_slow'
    column2 = 'close_slow'
    data = final_data(rate='d', column='transaction_count')
    draw(column1, column2, data[data.index>'2017'], )
    column1 = 'transaction_count_fast'
    column2 = 'close_fast'
    draw(column1, column2, data[data.index>'2017'], )



















