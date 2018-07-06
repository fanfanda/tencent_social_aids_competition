import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy import sparse
import gc
import sys

def train(sector, version, old_version):
    print("------------------load data-----------------")
    # sector = "0"

    all_train_data_x = sparse.load_npz('../data/split_data/npz/train_split_' + sector + version + '.npz')#训练集特征
    test_data_x = sparse.load_npz('../data/split_data/npz/test_split_' + sector + version + '.npz')#测试集特征
    new_test_data = pd.read_csv("../data/split_data/testdata/test_data_uaid_" + sector + old_version + ".csv", sep = ',')

    all_train_data_y = pd.read_csv('../data/split_data/label/split_data_' + sector + old_version + '_label.csv', sep = ',')#训练集标签
    all_train_data_y = all_train_data_y[['label']]
    all_train_data_y[all_train_data_y == -1] = 0

    best_iteration = 3400

    clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 63, reg_alpha = 0.0, reg_lambda = 1,
        max_depth = 9, n_estimators = best_iteration, objective = 'binary',
        subsample=0.7, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.05, min_child_weight = 50, random_state = 0, categorical_feature="0,1,2,3,4,5,6,7,8", n_jobs = -1)

    clf.fit(all_train_data_x, all_train_data_y.label, eval_set=[(all_train_data_x, all_train_data_y.label)], eval_metric='auc')


    result = clf.predict_proba(test_data_x)[:,1]

    
    new_test_data['score'] = result
    new_test_data['score'] = new_test_data.score.apply(lambda x: float('%.6f' % x))
    new_test_data = new_test_data[['aid','uid','score']]

    res = pd.read_csv('../data/test2.csv')
    split_aid = list(set(new_test_data.aid))
    res = res[res.aid.isin(split_aid)]
    res = pd.merge(res, new_test_data, on = ['aid','uid'], how = 'left')
    res.to_csv("./result/result_split" + sector + version + ".csv", index = False)




