import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy import sparse
import gc
import sys

print("------------------load data-----------------")

all_train_data_x = sparse.load_npz('../data/ffd/all_train_data_x.npz')#训练集特征
test_data_x = sparse.load_npz('../data/ffd/test_data_x.npz')#测试集特征

all_train_data_y = pd.read_csv('../data/ffd/all_train_data_y_add.csv', sep = ',')#训练集标签
all_train_data_y[all_train_data_y == -1] = 0


print("------------------split data-----------------")
x_validata_train, x_validata_test, y_validata_train, y_validata_test = train_test_split(all_train_data_x, all_train_data_y, test_size=0.8, random_state=42)


# del x_validata_train, x_validata_test, y_validata_train, y_validata_test
# gc.collect()

clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3300, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.1, min_child_weight=50, random_state=57, n_jobs=20)

clf.fit(x_validata_train, y_validata_train.label, eval_set=[(x_validata_test, y_validata_test.label)], eval_metric = 'auc',early_stopping_rounds = 300)

# del Dvalidata_train,Dvalidata_test
# gc.collect()


best_iteration = clf.best_iteration
# best_iteration = 3000

# Dtestdata = lgb.Dataset()

clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=best_iteration, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.1, min_child_weight=50, random_state=57, n_jobs=20)
clf.fit(all_train_data_x, all_train_data_y.label, eval_set=[(all_train_data_x, all_train_data_y.label)], eval_metric='auc',early_stopping_rounds=100)

# model = lgb.train(lgb_params, Dtraindata, valid_sets = watch_list, num_boost_round = best_iteration)

result = clf.predict_proba(test_data_x)[:,1]

new_test_data = pd.read_csv("../data/ffd/new_test_data.csv", sep = ',')
new_test_data['score'] = result
new_test_data['score'] = new_test_data.score.apply(lambda x: float('%.6f' % x))
new_test_data = new_test_data[['aid','uid','score']]

res=pd.read_csv('../data/test1.csv')
res = pd.merge(res, new_test_data, on = ['aid','uid'], how = 'left')
res.to_csv("submission_ffd_5_24.csv", index = False)

lgb_params={'boosting':'gbdt',
	    'application': 'binary',
	    'metric':'auc',
	    # 'min_split_gain':0.1,
	    'min_child_weight':50,
	    'max_depth':9,
        'num_leaves':31,
	    'lambda_l2':10,
	    'bagging_fraction':0.7,
	    'feature_fraction':0.7,
	    'bagging_freq':5,
	    'learning_rate': 0.1,
        'min_data_in_leaf':10,
	    'bagging_seed':5,
        'feature_fraction_seed':0,
        'verbose':0,
	    'num_threads':10
	    }



