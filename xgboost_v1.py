import xgboost
import pandas as pd
import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import datetime


def date_parser(df):
    date_recorder = list(map(lambda x: str(x), df['Timestamp'].values))
    df['improved_hour'] = list(map(lambda x: int(x[8:12]), date_recorder))
    # date_recorder = list(map(lambda x: datetime.datetime.strptime(str(x)[:8], '%Y%m%d'), date_recorder))
    # df['yearly_week_recorder'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
    # df['month_recorder'] = list(map(lambda x: int(x.strftime('%m')), date_recorder))
    del df['Timestamp']
    return df


def count_tags(id_tags_cell):
    """
    calculate how many name tags
    :param id_tags_cell: id tags of a single cell
    :return: number of id tags
    """
    if 'null' in id_tags_cell:
        return 0
    else:
        return len(id_tags_cell)


def get_user_tag(df):
    tags_series = list(df['User_Tags'].values)
    tags_series = list(map(lambda x: x.split(','), tags_series))
    # count tags for each column
    tags_count = list(map(lambda x: count_tags(x), tags_series))
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
    df = pd.concat([df, df_tags], axis=1)
    df['n_tags'] = tags_count
    return df


def split_ip(df):
    ip_series = list(df['IP'].values)
    ip_series = list(map(lambda x: x.split('.'), ip_series))
    df['IP0'] = list(map(lambda x: int(x[0]), ip_series))
    df['IP1'] = list(map(lambda x: int(x[1]), ip_series))
    df['IP2'] = list(map(lambda x: int(x[2]), ip_series))
    return df


def col_to_freq(df, col_names):
    for col in col_names:
        print('Changing to frequency %s' %col)
        val_counts = df[col].value_counts()
        df[col + '_freq'] = np.zeros((df.shape[0],))
        for i, val in enumerate(df[col].values):
            df[col + '_freq'].iat[i] = int(val_counts.at[val])
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
sub_factor = 5
# train = train.iloc[::sub_factor, :]
# train_labels = train_labels.iloc[::sub_factor]

train_index = train.index.values

submission_file = pd.DataFrame.from_csv("sample_submission.csv")

# combing tran and test data
# helps working on all the data and removes factorization problems between train and test
dataframe = pd.concat([train, test], axis=0)

del train
del test

"""
Preprocess
"""
# print(dataframe['Ad_slot_ID'].value_counts())  # Need to parse important words

# Parse date (removing is the easiest)
dataframe = date_parser(dataframe)
# Dummy-variabling user IDs
dataframe = get_user_tag(dataframe)
# Split ip into 3 different columns
dataframe = split_ip(dataframe)
# Change features to frequency of features
dataframe = col_to_freq(dataframe, ['User_ID', 'Domain', 'Ad_slot_ID'])

# Remove complicated values
dataframe = dataframe.drop(['User_ID', 'IP', 'URL', 'Domain', 'Anonymous_URL_ID', 'User_Tags', 'Ad_slot_ID', 'Hour'],
                           axis=1)
# Factorize str columns
print(dataframe.columns.values)
num_cols = []
for col in dataframe.columns.values:
    if dataframe[col].dtype.name == 'object':
        print('For column %s there are %d values' % (col, dataframe[col].value_counts().shape[0]))
        dataframe[col] = dataframe[col].factorize()[0]
    else:
        num_cols.append(col)

print(dataframe)

# No need for normalizing in xgboost (using a factor of the derivative as a vector of convergence)

"""
Split into train and test not done apriori in order to save space
"""

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
early_stopping = 150
param_grid = [
              {
               'silent': [1],
               'nthread': [3],
               'eval_metric': ['auc'],
               'eta': [0.03],
               'objective': ['binary:logistic'],
               'max_depth': [4],
               # 'min_child_weight': [1],
               'num_round': [5000],
               'gamma': [0],
               'subsample': [0.5, 0.7, 0.9],
               'colsample_bytree': [0.7],
               'scale_pos_weight': [0.8],
               'n_monte_carlo': [1],
               'cv_n': [4],
               'test_rounds_fac': [1.1],
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
    train_predictions = np.ones((train_index.shape[0],))
    print('There are %d columns' % dataframe.shape[1])

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
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            xg_train = xgboost.DMatrix(dataframe.loc[train_index].values[cv_train_index, :], label=y_train)
            xg_test = xgboost.DMatrix(dataframe.loc[train_index].values[cv_test_index, :], label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            num_round = params['num_round']
            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist, early_stopping_rounds=early_stopping);
            xgboost_rounds.append(xgclassifier.best_iteration)

        num_round = int(np.mean(xgboost_rounds))
        print('The best n_rounds is %d' % num_round)

        # Calculate train predictions over optimized number of rounds
        local_auc = []
        for cv_train_index, cv_test_index in kf:
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            xg_train = xgboost.DMatrix(dataframe.loc[train_index].values[cv_train_index, :], label=y_train)
            xg_test = xgboost.DMatrix(dataframe.loc[train_index].values[cv_test_index, :], label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

            # predict
            predicted_results = xgclassifier.predict(xg_test)
            train_predictions[cv_test_index] = predicted_results
            local_auc.append(roc_auc_score(y_test, predicted_results))

        print('Accuracy score ', np.mean(local_auc))
        mc_auc.append(np.mean(local_auc))
        mc_train_pred.append(train_predictions)
        mc_round.append(num_round)

    # Getting the mean integer
    mc_train_pred = np.mean(np.array(mc_train_pred), axis=0)

    mc_round_list.append(int(np.mean(mc_round)))
    mc_acc_mean.append(np.mean(mc_auc))
    mc_acc_sd.append(np.std(mc_auc))
    print('The AUC range is: %.5f to %.5f and best n_round: %d' %
          (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print_results.append('The accuracy range is: %.5f to %.5f and best n_round: %d' %
                         (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print('For ', mc_auc)
    print('The AUC of the average prediction is: %.5f' % roc_auc_score(train_labels.values, mc_train_pred))
    meta_solvers_train.append(mc_train_pred)

    # predicting the test set
    if params['mc_test']:
        watchlist = [(xg_train, 'train')]

        num_round = int(mc_round_list[-1] * params['test_rounds_fac'])
        mc_pred = []
        for i_mc in range(params['n_monte_carlo']):
            params['seed'] = i_mc
            xg_train = xgboost.DMatrix(dataframe.loc[train_index], label=train_labels.values.flatten())
            xg_test = xgboost.DMatrix(dataframe.loc[test_index])

            watchlist = [(xg_train, 'train')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);
            predicted_results = xgclassifier.predict(xg_test)
            mc_pred.append(predicted_results)

        meta_solvers_test.append(np.mean(np.array(mc_pred), axis=0))
        """ Write the last solution (ready for ensemble creation)"""
        print('writing to file')
        mc_train_pred = mc_train_pred
        # print(meta_solvers_test[-1])
        meta_solvers_test[-1] = meta_solvers_test[-1]
        pd.DataFrame(mc_train_pred).to_csv('train_xgboost_opt_eta003.csv')
        submission_file['Prediction'] = meta_solvers_test[-1]
        submission_file.to_csv('test_xgboost_opt_eta003.csv')

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
# CV = 4, eta = 0.03, SS = 4:0.74901148422397401
# Optimize subsample = [0.5, 0.75, 1] opt = 0.75: 0.75949645412709754
# Optimize colbytree = [0.25, 0.5, 0.75, 1] opt = 0.5: 0.762793051698
# Optimize scale_pos_weight = [0.8, 0.4, 0.2, 0.1] opt = 0.8: 0.767150084533
""" Changed AUC as a mean of all the predicted tests and not of the whole training set (imbalance problems)"""
# Added number of tags, hour of the day now includes minutes: 0.76060684965984882
# Optimize subsample = [0.5, 0.7, 0.9] opt = 0.9: 0.76237735572352383
# Optimize colbytree = [0.3, 0.5, 0.7, 0.9] opt = 0.7: 0.76337050129882611
""" Final Submission - no subsampling and finer eta """
# Optimize subsample = [4, 6, 8] opt = 4: 0.78115610846018946
