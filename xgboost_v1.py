import xgboost
import pandas as pd
import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import datetime


def date_parser(df):
    # date_recorder = list(map(lambda x: str(x)[:8], df['Timestamp'].values))
    # date_recorder = list(map(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'), date_recorder))
    # df['year_recorder'] = list(map(lambda x: int(x.strftime('%Y')), date_recorder))
    # df['yearly_week_recorder'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
    # df['month_recorder'] = list(map(lambda x: int(x.strftime('%m')), date_recorder))
    del df['Timestamp']
    return df


def get_user_tag(df):
    tags_series = list(df['User_Tags'].values)
    tags_series = list(map(lambda x: x.split(','), tags_series))
    tags_list = []
    for search_item in tags_series:
        for tag in search_item:
            if not tag in tags_list:
                tags_list.append(tag)
    # print('There are %d ID tags' % len(tags_list))
    df_tags = pd.DataFrame(np.zeros((df.shape[0], len(tags_list))), index=df.index, columns=tags_list)
    df_tags_index = df.index.values
    for i, search_item in enumerate(tags_series):
        for tag in search_item:
            df_tags.at[df_tags_index[i], tag] = 1
    # print(df_tags)
    df = pd.concat([df, df_tags], axis=1)
    return df


def split_ip(df):
    ip_series = list(df['IP'].values)
    ip_series = list(map(lambda x: x.split('.'), ip_series))
    df['IP0'] = list(map(lambda x: int(x[0]), ip_series))
    df['IP1'] = list(map(lambda x: int(x[1]), ip_series))
    df['IP2'] = list(map(lambda x: int(x[2]), ip_series))
    return df
"""
Import data
"""
col_names = ['Click', 'Weekday', 'Hour', 'Timestamp', 'Log_Type', 'User_ID', 'User-Agent', 'IP', 'Region', 'City',
             'Ad_Exchange', 'Domain', 'URL', 'Anonymous_URL_ID', 'Ad_slot_ID', 'Ad_slot_width', 'Ad_slot_height',
             'Ad_slot_visibility', 'Ad_slot_format', 'Ad_slot_floor_price', 'Creative_ID', 'Key_Page_URL',
             'Advertiser_ID', 'User_Tags']
train = pd.read_csv('train.txt', sep='\t', names=col_names, index_col=False)

train_labels = train['Click']
print(train_labels.value_counts(normalize=True))
del train['Click']

# Remove label column name for test
col_names.pop(0)
test = pd.read_csv('test.txt', sep='\t', names=col_names, index_col=False)
test.index = test.index.values + train.shape[0]
test_index = test.index.values

# For faster iterations
sub_factor = 10
train = train.iloc[::sub_factor, :]
train_labels = train_labels.iloc[::sub_factor]

train_index = train.index.values

submission_file = pd.DataFrame.from_csv("sample_submission.csv")

# combing tran and test data
# helps working on all the data and removes factorization problems between train and test
dataframe = pd.concat([train, test], axis=0)

"""
Preprocess
"""
# print(dataframe['User_ID'].value_counts())  # Need to change to frequency
# print(dataframe['Domain'].value_counts())  # Need to change to frequency
# print(dataframe['Ad_slot_ID'].value_counts())  # Need to change to frequency, and parse important words

# Parse date (removing is the easiest)
dataframe = date_parser(dataframe)
# Dummy-variabling user IDs
dataframe = get_user_tag(dataframe)
# Split ip into 3 different columns
dataframe = split_ip(dataframe)


# Remove complicated values
dataframe = dataframe.drop(['User_ID', 'IP', 'URL', 'Domain', 'Anonymous_URL_ID', 'User_Tags', 'Ad_slot_ID'], axis=1)
# Factorize str columns
print(dataframe.columns.values)
num_cols = []
for col in dataframe.columns.values:
    if dataframe[col].dtype.name == 'object':
        print('For column %s there are %d values' % (col, dataframe[col].value_counts().shape[0]))
        dataframe[col] = dataframe[col].factorize()[0]
    else:
        num_cols.append(col)

# No need for normalizing in xgboost (using a factor of the derivative as a vector of convergence)

"""
Split into train and test
"""

train = dataframe.loc[train_index]
test = dataframe.loc[test_index]

"""
CV
"""
best_score = 0
best_params = 0
best_train_prediction = 0
best_prediction = 0
meta_solvers_train = []
meta_solvers_test = []
best_train = 0
best_test = 0

# Optimization parameters
early_stopping = 300
param_grid = [
              {
               'silent': [1],
               'nthread': [3],
               'eval_metric': ['auc'],
               'eta': [0.01],
               'objective': ['binary:logistic'],
               'max_depth': [4],
               # 'min_child_weight': [1],
               'num_round': [4000],
               'gamma': [0],
               'subsample': [1.0],
               'colsample_bytree': [1.0],
               'n_monte_carlo': [5],
               'cv_n': [4],
               'test_rounds_fac': [1.2],
               'count_n': [0],
               'mc_test': [True]
              }
              ]

print('start CV optimization')
mc_round_list = []
mc_acc_mean = []
mc_acc_sd = []
params_list = []
print_results = []
for params in ParameterGrid(param_grid):
    print(params)
    params_list.append(params)
    train_predictions = np.ones((train.shape[0],))
    print('There are %d columns' % train.shape[1])

    # CV
    mc_auc = []
    mc_round = []
    mc_train_pred = []
    # Use monte carlo simulation if needed to find small improvements
    for i_mc in range(params['n_monte_carlo']):
        cv_n = params['cv_n']
        kf = StratifiedKFold(train_labels.values.flatten(), n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)

        xgboost_rounds = []
        # Finding optimized number of rounds
        for cv_train_index, cv_test_index in kf:
            X_train, X_test = train.values[cv_train_index, :], train.values[cv_test_index, :]
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            xg_train = xgboost.DMatrix(X_train, label=y_train)
            xg_test = xgboost.DMatrix(X_test, label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            num_round = params['num_round']
            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist, early_stopping_rounds=early_stopping);
            xgboost_rounds.append(xgclassifier.best_iteration)

        num_round = int(np.mean(xgboost_rounds))
        print('The best n_rounds is %d' % num_round)

        # Calculate train predictions over optimized number of rounds
        for cv_train_index, cv_test_index in kf:
            X_train, X_test = train.values[cv_train_index, :], train.values[cv_test_index, :]
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            xg_train = xgboost.DMatrix(X_train, label=y_train)
            xg_test = xgboost.DMatrix(X_test, label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

            # predict
            predicted_results = xgclassifier.predict(xg_test)
            print(predicted_results)
            train_predictions[cv_test_index] = predicted_results

        print('Accuracy score ', roc_auc_score(train_labels.values, train_predictions))
        mc_auc.append(roc_auc_score(train_labels.values, train_predictions))
        mc_train_pred.append(train_predictions)
        mc_round.append(num_round)

    # Getting the mean integer
    mc_train_pred = (np.mean(np.array(mc_train_pred), axis=0) + 0.5).astype(int)

    mc_round_list.append(int(np.mean(mc_round)))
    mc_acc_mean.append(np.mean(mc_auc))
    mc_acc_sd.append(np.std(mc_auc))
    print('The accuracy range is: %.5f to %.5f and best n_round: %d' %
          (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print_results.append('The accuracy range is: %.5f to %.5f and best n_round: %d' %
                         (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print('For ', mc_auc)
    print('The accuracy of the average prediction is: %.5f' % roc_auc_score(train_labels.values, mc_train_pred))
    meta_solvers_train.append(mc_train_pred)

    # train machine learning
    xg_train = xgboost.DMatrix(train.values, label=train_labels.values)
    xg_test = xgboost.DMatrix(test.values)

    # predicting the test set
    if params['mc_test']:
        watchlist = [(xg_train, 'train')]

        num_round = int(mc_round_list[-1] * params['test_rounds_fac'])
        mc_pred = []
        for i_mc in range(params['n_monte_carlo']):
            params['seed'] = i_mc
            xg_train = xgboost.DMatrix(train, label=train_labels.values.flatten())
            xg_test = xgboost.DMatrix(test)

            watchlist = [(xg_train, 'train')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);
            predicted_results = xgclassifier.predict(xg_test)
            mc_pred.append(predicted_results)

        meta_solvers_test.append(np.mean(np.array(mc_pred), axis=0))
        """ Write opt solution """
        print('writing to file')
        mc_train_pred = mc_train_pred
        print(meta_solvers_test[-1])
        meta_solvers_test[-1] = meta_solvers_test[-1]
        pd.DataFrame(mc_train_pred).to_csv('train_xgboost_d6.csv')
        submission_file['status_group'] = meta_solvers_test[-1]
        submission_file.to_csv("test_xgboost_d6.csv")

    # saving best score for printing
    if mc_acc_mean[-1] < best_score:
        print('new best log loss')
        best_score = mc_acc_mean[-1]
        best_params = params
        best_train_prediction = mc_train_pred
        if params['mc_test']:
            best_prediction = meta_solvers_test[-1]

print(best_score)
print(best_params)

print(params_list)
print(print_results)
print(mc_acc_mean)
print(mc_acc_sd)
"""
Final Solution
"""
# optimizing:
# CV = 4, eta = 0.1
# Removed 'User_ID', 'IP', 'Region', 'URL', 'Domain', 'Anonymous_URL_ID', 'Key_Page_URL', 
# 'User_Tags', 'Ad_slot_ID', 'Timestamp':

