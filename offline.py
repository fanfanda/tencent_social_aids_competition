
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy import sparse
import gc
import sys
print("------------------load data-----------------")
sector = "4"
version = "_612"
all_train_data_x = sparse.load_npz('../data/split_data/npz/train_split_' + sector + version + '.npz')#训练集特征
# all_train_data_x = sparse.load_npz('../data/ffd/all_train_data_x_1000.npz')#训练集特征
# test_data_x = sparse.load_npz('../data/split_data/npz/test_split_' + sector + version + '.npz')#测试集特征
print(all_train_data_x.shape)
# all_train_data_y = pd.read_csv('../data/split_data/label/split_data_' + sector + version + '_label.csv', sep = ',')#训练集标签
all_train_data_y = pd.read_csv('../data/split_data/label/split_data_' + sector + '_new_label.csv', sep = ',')#训练集标签
# all_train_data_y = pd.read_csv('../data/ffd/all_train_data_y_add.csv', sep = ',')#训练集标签
all_train_data_y = all_train_data_y[['label']]
all_train_data_y[all_train_data_y == -1] = 0


print("------------------split data-----------------")
x_validata_train, x_validata_test, y_validata_train, y_validata_test = train_test_split(all_train_data_x, all_train_data_y, test_size=0.05, random_state=42)


# print("generate data")
# Dtrain_data = lgb.Dataset(x_validata_train,label=y_validata_train.label, categorical_feature=[0,1,2,3,4,5,6,7,8,9])
# Dvalidata_data = lgb.Dataset(x_validata_test,label=y_validata_test.label, categorical_feature=[0,1,2,3,4,5,6,7,8,9])

# Dtrain_data.save_binary
# watch_list = [Dtrain_data,Dvalidata_data]
# lgb_params={'boosting':'gbdt',
# 	    'application': 'binary',
# 	    'metric':'auc',
# 	    'min_child_weight':50,
# 	    'max_depth':9,
#             'num_leaves':63,
# 	    'lambda_l2':10,
# 	    'bagging_fraction':0.7,
# 	    'feature_fraction':0.7,
# 	    'bagging_freq':5,
# 	    'learning_rate': 0.05,
#             'min_data_in_leaf':10,
# 	    # # 'tree_method':'exact',
# 	    'bagging_seed':5,
#             'feature_fraction_seed':0,
#         # 'verbose':0,
# 	    'num_threads':30
# 	    }
# lgb.train(lgb_params, Dtrain_data, valid_sets = watch_list,num_boost_round = 6000, categorical_feature=[0,1,2,3,4,5,6,7,8,9])

# # del x_validata_train, x_validata_test, y_validata_train, y_validata_test
# # gc.collect()

clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 63, reg_alpha = 0.0, reg_lambda = 1,
        max_depth = 9, n_estimators = 8000, objective = 'binary',
        subsample=0.7, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.05, min_child_weight = 50, random_state = 0, categorical_feature="0,1,2,3,4,5,6,7,8", n_jobs = 35)
#categorical_feature="0,1,2,3,4,5,6,7,8"
clf.fit(x_validata_train, y_validata_train.label, eval_set=[(x_validata_train,y_validata_train.label),(x_validata_test, y_validata_test.label)], eval_metric = 'auc',early_stopping_rounds = 300)
