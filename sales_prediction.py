from itertools import product
from typing import List
import time
import gc
import pickle
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
from random import sample


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        df[col] = LabelEncoder().fit_transform(df[col].astype('str'))
    return df


def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df


def text_based_features(df, cols):
    # create text based features
    # text based feature generation - TF
    for col in cols:
        df[col] = df[col].apply(lambda x: re.sub('\W', ' ', x))

        # text based feature generation - TF
        countVec = CountVectorizer(max_features=5000, stop_words='english', min_df=.01, max_df=.90)

        # text based feature generation - TF
        countVec.fit(df[col])
        countVec_count = countVec.transform(df[col])
        occ = np.asarray(countVec_count.sum(axis=0)).ravel().tolist()
        tf = pd.DataFrame({'term': countVec.get_feature_names(), 'TF': occ / np.sum(occ)})
        d_dict_col_tf = {tf['term'].values[i]: tf['TF'].values[i] for i in range(len(tf['term'].values))}

        # text based feature generation - TF-idf
        Transformer = TfidfTransformer()
        Weights = Transformer.fit_transform(countVec_count)
        WeightsFin = np.asarray(Weights.mean(axis=0)).ravel().tolist()
        WeightFrame = pd.DataFrame({'term': countVec.get_feature_names(), 'TF-idf': WeightsFin})
        d_dict_col_tfidf = {WeightFrame['term'].values[i]: WeightFrame['TF-idf'].values[i] for i in
                            range(len(WeightFrame['term'].values))}

        del WeightFrame, Weights, countVec, tf
        gc.collect()

        # the sum of the TF values in the document is used as a composite item_category_name score for the document

        def f(x):
            total = 0
            for item in x.split(' '):
                word = re.sub(r'\s+', '', item)
                try:
                    y = d_dict[word]
                except:
                    y = 0
                total += y
            return total

        d_dict = d_dict_col_tf
        df[col + 'TF'] = df[col].apply(f)
        d_dict = d_dict_col_tfidf
        df[col + 'TFIDF'] = df[col].apply(f)
        df = df.drop(columns=[col], axis=1)

    return df


def build_dfs(data_types, paths):
    dfs = []
    for i in range(len(data_types)):
        if data_types[i] == 'csv':
            dfs.append(pd.read_csv(paths[i]))
    return dfs


def eda(sales_train, items, item_cats, shops, sales_test):
    # print(sales_train.head())
    # print(items.head())
    # print(item_cats.head())
    # print(shops.head())
    # print(sales_test.head())
    # print(sales_train.describe())

    sales_train['date_formatted'] = pd.to_datetime(sales_train.date, format="%d.%m.%Y")
    sales_train[['day', 'month', 'year']] = sales_train.date.str.split('.', expand=True)
    sales_train['sales'] = sales_train.item_price * sales_train.item_cnt_day

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    sales_train.date_block_num.hist(ax=axes[0])
    sales_train[sales_train['item_cnt_day'] < 10].item_cnt_day.hist(ax=axes[1])
    np.log(sales_train['item_price']).hist(ax=axes[2])
    sales_train.sales.hist(ax=axes[3])

    fig, axes = plt.subplots(1, 3, figsize=(33, 5))

    sales_train.year.value_counts().sort_index().plot.bar(ax=axes[0], title='year')
    sales_train.month.value_counts().sort_index().plot.bar(ax=axes[1], title='month')
    sales_train.day.value_counts().sort_index().plot.bar(ax=axes[2], title='day')

    sales_train.groupby('date_formatted').agg({"item_cnt_day": "sum"}).plot(figsize=(15, 6),
                                                                            title="Items transacted per day")

    fig, axes = plt.subplots(1, 2, figsize=(33, 5))

    sales_train.groupby('year').item_cnt_day.sum().plot.bar(figsize=(15, 6), title="Total Items transacted each year",
                                                            ax=axes[0])
    sales_train.groupby('year').item_cnt_day.mean().plot.bar(figsize=(15, 6),
                                                             title="Average item count per transaction each year",
                                                             ax=axes[1])

    sales_train.groupby('date_block_num').agg({"item_cnt_day": "sum"}).plot(figsize=(15, 6),
                                                                            title="Items transacted per day")

    sales_train['dayofweek'] = sales_train.date_formatted.dt.dayofweek  # The day of the week with Monday=0, Sunday=6
    sales_train.groupby("dayofweek").agg({"dayofweek": "count"}).plot.bar(figsize=(10, 6))

    sales_train['date_month'] = (sales_train.year + sales_train.month)
    fig, axes = plt.subplots(1, 2, figsize=(25, 5))
    ax = sales_train.groupby('date_month').item_cnt_day.mean().reset_index().plot(x_compat=True,
                                                                                  title="avg item_cnt_day by month",
                                                                                  figsize=(20, 6), ax=axes[0])
    ax.set(xlabel='date month', ylabel='average item_cnt_day')

    ax = sales_train.groupby('date_month').item_cnt_day.sum().reset_index().plot(x_compat=True,
                                                                                 title="sum item_cnt_day by month",
                                                                                 figsize=(20, 6), ax=axes[1])
    ax.set(xlabel='date month', ylabel='sum item_cnt_day')

    plt.scatter(sales_train.index, sales_train.date_block_num)

    plt.plot(sales_train.item_cnt_day, '.')

    # plot items for each shop -- training set vs test set
    fix, axes = plt.subplots(1, 2, figsize=(15, 3))

    sales_train.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkBlue',
                                                                            s=0.1,
                                                                            ax=axes[0],
                                                                            title="shops vs items sales_train")
    sales_test.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkBlue',
                                                                           s=0.1,
                                                                           ax=axes[1], title="shops vs items test")

    plt.show()

    return


def preprocessing(sales_train, items, item_cats, shops, sales_test):
    # debug
    sales_train = sales_train.sample(frac=0.001, replace=True, random_state=42)

    # median value fill
    median = sales_train[
        (sales_train.shop_id == 32) & (sales_train.item_id == 2973) & (sales_train.date_block_num == 4) & (
                sales_train.item_price > 0)].item_price.median()
    sales_train.loc[sales_train.item_price < 0, 'item_price'] = median

    # Якутск Орджоникидзе, 56
    sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
    sales_test.loc[sales_test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
    sales_test.loc[sales_test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
    sales_test.loc[sales_test.shop_id == 10, 'shop_id'] = 11

    # trivial feature
    sales_test['date_block_num'] = 34

    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id', 'city_code']]

    # split item_category_name into more meaningful parts
    item_cats['split'] = item_cats['item_category_name'].str.split('-')
    item_cats['type'] = item_cats['split'].map(lambda x: x[0].strip())
    item_cats['type_code'] = LabelEncoder().fit_transform(item_cats['type'])
    item_cats['subtype'] = item_cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    item_cats['subtype_code'] = LabelEncoder().fit_transform(item_cats['subtype'])
    item_cats = item_cats[['item_category_id', 'type_code', 'subtype_code']]
    items.drop(['item_name'], axis=1, inplace=True)

    # add text based features
    # items = text_based_features(items, ['item_name'])

    # For every month we create a all_data from all shops/items combinations from that month
    all_data = []
    index_cols = ['date_block_num', 'shop_id', 'item_id']
    for block_num in sales_train['date_block_num'].unique():
        sales = sales_train[sales_train.date_block_num == block_num]
        all_data.append(
            np.array(list(product([block_num], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    # turn the all_data into pandas dataframe
    all_data = pd.DataFrame(np.vstack(all_data), columns=index_cols, dtype=np.int32)
    all_data.sort_values(index_cols, inplace=True)

    # get aggregate sum
    gb = sales_train.groupby(index_cols, as_index=False).agg({'item_cnt_day': ['sum']})

    # fix column names
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    gb = gb.rename(columns={'sum': 'item_cnt_month'}, errors="raise")

    # join aggregated data to the all_data
    all_data = pd.merge(all_data, gb, how='left', on=index_cols)
    all_data['item_cnt_month'] = all_data['item_cnt_month'].fillna(0).clip(lower=0, upper=20)
    del gb
    gc.collect()

    # concat sales_test to all_data
    all_data = pd.concat([all_data, sales_test], ignore_index=True, sort=False, keys=index_cols)
    all_data.fillna(0, inplace=True)

    # merge rest of data
    all_data = pd.merge(all_data, shops, on=['shop_id'], how='left')
    all_data = pd.merge(all_data, items, on=['item_id'], how='left')
    all_data = pd.merge(all_data, item_cats, on=['item_category_id'], how='left')
    del shops, items, item_cats
    gc.collect()

    # create target month mean encodings and lags
    group = all_data.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num'], how='left')
    all_data = lag_feature(all_data, [1, 2, 3, 6, 12], 'item_cnt_month')
    all_data.drop(['item_cnt_mean_month'], axis=1, inplace=True)

    # create target month-item_id mean encodings and lags
    group = all_data.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_id']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id'], how='left')
    all_data = lag_feature(all_data, [1, 2, 3, 6, 12], 'item_cnt_mean_month_item_id')
    all_data.drop(['item_cnt_mean_month_item_id'], axis=1, inplace=True)

    # create target month-shop_id mean encodings and lags
    group = all_data.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'shop_id'], how='left')
    all_data = lag_feature(all_data, [1, 2, 3, 6, 12], 'item_cnt_mean_month_shop')
    all_data.drop(['item_cnt_mean_month_shop'], axis=1, inplace=True)

    # create target month-item_cat mean encodings and lags
    group = all_data.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_cat']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'item_category_id'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_cat')
    all_data.drop(['item_cnt_mean_month_item_cat'], axis=1, inplace=True)

    # create target month-shop-item_cat mean encodings and lags
    group = all_data.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_cat_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_cat_shop')
    all_data.drop(['item_cnt_mean_month_item_cat_shop'], axis=1, inplace=True)

    # create target month-shop-item_type mean encodings and lags
    group = all_data.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_type_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_type_shop')
    all_data.drop(['item_cnt_mean_month_item_type_shop'], axis=1, inplace=True)

    # create target month-shop-item_type mean encodings and lags
    group = all_data.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_subtype_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_subtype_shop')
    all_data.drop(['item_cnt_mean_month_item_subtype_shop'], axis=1, inplace=True)

    # create target month-city_code mean encodings and lags
    group = all_data.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_city_code']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'city_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_city_code')
    all_data.drop(['item_cnt_mean_month_city_code'], axis=1, inplace=True)

    # create target month-item_id_city_code mean encodings and lags
    group = all_data.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_city_code']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_city_code')
    all_data.drop(['item_cnt_mean_month_item_city_code'], axis=1, inplace=True)

    # create target month-item_type_code mean encodings and lags
    group = all_data.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_type_code']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'type_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_type_code')
    all_data.drop(['item_cnt_mean_month_item_type_code'], axis=1, inplace=True)

    # create target month-item_subtype_code mean encodings and lags
    group = all_data.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['item_cnt_mean_month_item_subtype_code']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'subtype_code'], how='left')
    all_data = lag_feature(all_data, [1], 'item_cnt_mean_month_item_subtype_code')
    all_data.drop(['item_cnt_mean_month_item_subtype_code'], axis=1, inplace=True)

    # create price item_id mean encodings and lags
    group = sales_train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_price_mean_item_id']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['item_id'], how='left')

    # create price month-item_id mean encodings and lags
    group = sales_train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_price_mean_month_item_id']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id'], how='left')

    # feature interaction mean encodings and lags
    sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']

    # create revenue month-shop_id mean encodings and lags
    group = sales_train.groupby(['date_block_num', 'shop_id']).agg({'revenue': ['sum']})
    group.columns = ['revenue_sum_month_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['date_block_num', 'shop_id'], how='left')
    del sales_train
    gc.collect()

    # create revenue month-shop_id meansum encodings and lags
    group = group.groupby(['shop_id']).agg({'revenue_sum_month_shop': ['mean']})
    group.columns = ['revenue_meansum_shop']
    group.reset_index(inplace=True)
    all_data = pd.merge(all_data, group, on=['shop_id'], how='left')

    # feature interaction trends and lags
    all_data['delta_revenue'] = (all_data['revenue_sum_month_shop'] - all_data['revenue_meansum_shop']) / all_data[
        'revenue_meansum_shop']
    all_data = lag_feature(all_data, [1], 'delta_revenue')
    all_data.drop(['revenue_sum_month_shop', 'revenue_meansum_shop', 'delta_revenue'], axis=1, inplace=True)

    # lags on feature interaction trends
    lags = [1, 2, 3, 4, 5, 6]
    all_data = lag_feature(all_data, lags, 'item_price_mean_month_item_id')
    for i in lags:
        all_data['delta_price_lag_' + str(i)] = \
            (all_data['item_price_mean_month_item_id_lag_' + str(i)] - all_data['item_price_mean_item_id']) / all_data[
                'item_price_mean_item_id']

    def filter_lag(row):
        for i in lags:
            if row['delta_price_lag_' + str(i)]:
                return row['delta_price_lag_' + str(i)]
        return 0

    all_data['delta_price_lag'] = all_data.apply(filter_lag, axis=1)
    all_data['delta_price_lag'] = all_data['delta_price_lag']
    all_data['delta_price_lag'].fillna(0, inplace=True)

    # time based features
    all_data['month'] = all_data['date_block_num'] % 12
    month_length = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    all_data['month_length'] = all_data['month'].map(month_length)

    cache = {}
    all_data['item_shop_last_sale'] = -1
    for idx, row in all_data.iterrows():
        key = str(row.item_id) + ' ' + str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            all_data.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

    cache = {}
    all_data['item_last_sale'] = -1
    all_data['item_last_sale'] = all_data['item_last_sale']
    for idx, row in all_data.iterrows():
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num > last_date_block_num:
                all_data.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num

    all_data['item_shop_first_sale'] = all_data['date_block_num'] - all_data.groupby(['item_id', 'shop_id'])[
        'date_block_num'].transform('min')
    all_data['item_first_sale'] = all_data['date_block_num'] - all_data.groupby('item_id')['date_block_num'].transform(
        'min')

    # filter out data from first year
    all_data = all_data[all_data.date_block_num >= 12]

    for col in [col for col in all_data if 'item_cnt' in col]:
        all_data[col] = all_data[col].fillna(0)

    all_data = optimize(all_data)

    return all_data


def validation(df, validation_params):
    model_name = validation_params['model_name']
    target = validation_params['target']
    train_blocks = validation_params['train_blocks']
    serialize = validation_params['serialize']
    plot_feature_importances = validation_params['plot_feature_importances']
    f_frac = validation_params['return_feature_fraction']

    print('Validating ...')

    rmse_folds = []
    best_alphas = []
    n_features = len(df.columns)
    important_features = set()
    for i, cur_block_num in enumerate(train_blocks):
        print('\t' + 'Training fold ' + str(i + 1) + '...')
        X_train_i = df.loc[df['date_block_num'] < cur_block_num].drop(columns=[target], axis=1)
        y_train_i = pd.DataFrame(data=df.loc[df['date_block_num'] < cur_block_num], columns=[target])
        if not serialize:
            X_validation_i = df.loc[df['date_block_num'] == cur_block_num].drop(columns=target, axis=1)
            y_validation_i = pd.DataFrame(data=df.loc[df['date_block_num'] == cur_block_num], columns=[target])

        if model_name == 'lr':
            # 1. fit linear regression model_name
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train_i.values, y_train_i)
            if not serialize:
                # score = r2_score(y_validation_i, pred_lr_i, squared=False)
                pred_lr_i = np.clip(model.predict(X_validation_i.values), 0, 20)
                score = mean_squared_error(y_validation_i, pred_lr_i, squared=False)
                rmse_folds.append(score)
                print('\tLR RMSE: ', score)
        elif model_name == 'xgb':
            model = XGBRegressor(
                max_depth=8,
                n_estimators=100,
                min_child_weight=300,
                colsample_bytree=0.8,
                subsample=0.8,
                eta=0.3,
                seed=42,
                nthread=-1)
            model.fit(
                X_train_i,
                y_train_i,
                eval_metric="rmse",
                eval_set=[(X_train_i, y_train_i)],
                verbose=False,
                early_stopping_rounds=10)
            if not serialize:
                pred_xgb_i = model.predict(X_validation_i).clip(0, 20)
                score = mean_squared_error(y_validation_i, pred_xgb_i, squared=False)
                rmse_folds.append(score)
                print('\tXGB RMSE: ', score)
                if plot_feature_importances:
                    plot_importance(booster=model)
                    plt.show()
                # fi_vals_i_temp = model.get_booster().get_score(importance_type='gain')
                if_i = model.get_booster().get_score(importance_type='gain')
                keys = list(if_i.keys())
                values = list(if_i.values())
                if_i_list = sorted([(values[i], keys[i]) for i in range(len(keys))], reverse=True)[
                            0:int(f_frac * n_features)]
                top = set([item[1] for item in if_i_list])
                important_features.update(top)
        elif model_name == 'lr_xgb_mix':  # out of order. need one hot encoding on categorical features and
            # feature scaling on numerical features
            model_lr = LinearRegression(n_jobs=-1, fit_intercept=True)
            model_lr.fit(X_train_i.values, y_train_i.values)
            model_xgb = model = XGBRegressor(
                max_depth=8,
                n_estimators=100,
                min_child_weight=300,
                colsample_bytree=0.8,
                subsample=0.8,
                eta=0.3,
                seed=42,
                nthread=-1)
            model_xgb.fit(
                X_train_i,
                y_train_i,
                eval_metric="rmse",
                eval_set=[(X_train_i, y_train_i)],
                verbose=False,
                early_stopping_rounds=10)
            if not serialize:
                pred_lr_i = np.clip(model_lr.predict(X_validation_i.values), 0, 20)
                pred_xgb_i = np.clip(model_xgb.predict(X_validation_i.values), 0, 20)
                alpha = np.linspace(0, 1, 100, endpoint=True)
                score_max = 999
                alpha_max = 0
                for j in range(len(alpha)):
                    composite = [alpha[j] * pred_lr_i[k] + (1 - alpha[j]) * pred_xgb_i[k] for k in
                                 range(len(pred_lr_i))]
                    score = mean_squared_error(y_validation_i, composite, squared=False)
                    if score < score_max:
                        score_max = score
                        alpha_max = alpha[j]
                print('\tConvex RMSE mix score: ', score_max)
                rmse_folds.append(score_max)
                best_alphas.append(alpha_max)

    if serialize:
        print('\n\tSerializing model...')
        filename = ''.join([serialize, '/model_', model_name, '.sav'])
        pickle.dump(model, open(filename, 'wb'))  # 'wb' write binary
    else:
        print('\n\tRMSE min, max, average: ', np.min(rmse_folds), np.max(rmse_folds), np.average(rmse_folds))
        if model_name == 'lr_xgb_mix':
            print('\n\tAlphas min, max, average: ', np.min(best_alphas), np.max(best_alphas),
                  np.average(best_alphas))

    return important_features


def prediction(data, test, serialize=False, ref=None):
    X_train = data[data.date_block_num <= 33].drop(['item_cnt_month'], axis=1)
    y_train = data[data.date_block_num <= 33]['item_cnt_month']
    X_validation = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    y_validation = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


    model = XGBRegressor(
        max_depth=8,
        n_estimators=100,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=42,
        nthread=-1)

    model.fit(
        X_train,
        y_train,
        eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_validation, y_validation)],
        verbose=True,
        early_stopping_rounds=10)

    y_test_pred = model.predict(X_test).clip(0, 20)

    if serialize:
        print('Serializing model...')
        filename = ''.join([serialize, '/model_xgb.sav'])
        pickle.dump(model, open(filename, 'wb'))  # 'wb' write binary

    submission = pd.DataFrame({'ID': test.index, 'item_cnt_month': y_test_pred})
    try:
        print('r2 with first submission : ', r2_score(ref.item_cnt_month.values, submission.item_cnt_month.values))
    except AttributeError:
        print('Cannot compute delta R2. Reference not assigned.')
    submission.to_csv(DATA_FOLDER + 'submission_xgb.csv', index=False)

    return r2_score(ref.item_cnt_month.values, submission.item_cnt_month.values)


if __name__ == '__main__':
    # build dataframes
    a = time.time()
    print('Building dataframes ...')

    DATA_FOLDER = '/home/yumitomo/Documents/Coursera/Advanced Machine Learning/' \
                  'How to Win a Data Science Competition/week5/competitive-data-science-final-project/'
    # DATA_FOLDER = '../data/'

    files_train = [os.path.join(DATA_FOLDER, 'sales_train.csv.gz'),
                   os.path.join(DATA_FOLDER, 'items.csv'),
                   os.path.join(DATA_FOLDER, 'item_categories.csv'),
                   os.path.join(DATA_FOLDER, 'shops.csv'),
                   os.path.join(DATA_FOLDER, 'test.csv'),
                   os.path.join(DATA_FOLDER, 'xgb_submission_refactored.csv')]
                   # os.path.join(DATA_FOLDER, 'submission_xgb_ref1.csv')]

    sales_train, items, cats, shops, sales_test, ref = build_dfs(['csv' for file in files_train],
                                                                 [file for file in files_train])

    # eda(sales_train, items, cats, shops, sales_test)

    data = preprocessing(sales_train, items, cats, shops, sales_test)

    validation_params = {
        'model_name': 'xgb',
        'target': 'item_cnt_month',
        'train_blocks': [27, 28, 29, 30, 31, 32, 33],
        'serialize': False,
        'plot_feature_importances': False,
        'return_feature_fraction': 0.25
    }

    # pick best features and iterate
    data_test = data[[
        'date_block_num',
        'shop_id',
        'item_id',
        'item_cnt_month',
        # 'ID',
        # 'city_code',
        'item_category_id',
        'type_code',
        'subtype_code',
        'item_cnt_month_lag_1',
        'item_cnt_month_lag_2',
        'item_cnt_month_lag_3',
        'item_cnt_month_lag_6',
        'item_cnt_month_lag_12',
        'item_cnt_mean_month_item_id_lag_1',
        'item_cnt_mean_month_item_id_lag_2',
        'item_cnt_mean_month_item_id_lag_3',
        # 'item_cnt_mean_month_item_id_lag_6',
        # 'item_cnt_mean_month_item_id_lag_12',
        'item_cnt_mean_month_shop_lag_1',
        'item_cnt_mean_month_shop_lag_2',
        'item_cnt_mean_month_shop_lag_3',
        # 'item_cnt_mean_month_shop_lag_6',
        # 'item_cnt_mean_month_shop_lag_12',
        'item_cnt_mean_month_item_cat_lag_1',
        # 'item_cnt_mean_month_item_cat_shop_lag_1',
        # 'item_cnt_mean_month_item_type_shop_lag_1',
        # 'item_cnt_mean_month_item_subtype_shop_lag_1',
        # 'item_cnt_mean_month_city_code_lag_1',
        # 'item_cnt_mean_month_item_city_code_lag_1',
        # 'item_cnt_mean_month_item_type_code_lag_1',
        # 'item_cnt_mean_month_item_subtype_code_lag_1',
        # 'item_price_mean_item_id',
        # 'item_price_mean_month_item_id',
        # 'delta_revenue_lag_1',
        # 'item_price_mean_month_item_id_lag_1',
        # 'item_price_mean_month_item_id_lag_2',
        # 'item_price_mean_month_item_id_lag_3',
        # 'item_price_mean_month_item_id_lag_4',
        # 'item_price_mean_month_item_id_lag_5',
        # 'item_price_mean_month_item_id_lag_6',
        # 'delta_price_lag_1',
        # 'delta_price_lag_2',
        # 'delta_price_lag_3',
        # 'delta_price_lag_4',
        # 'delta_price_lag_5',
        # 'delta_price_lag_6',
        # 'delta_price_lag',
        'month',
        'month_length',
        'item_shop_last_sale',
        'item_last_sale',
        'item_shop_first_sale',
        'item_first_sale'
    ]]

    top_gain_features = validation(data, validation_params)
    top_gain_features.update({'shop_id', 'item_id', 'date_block_num', 'item_cnt_month'})
    # print(list(top_gain_features))

    prediction(data_test, sales_test, serialize=DATA_FOLDER, ref=ref)

    print('Finished in ', (time.time()-a)/60, ' minutes')