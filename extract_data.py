import pandas as pd
import numpy as np
import gc,os
from tencent_utility import *

print("------------------load data-----------------")
userFeature_data = pd.read_csv("../data/userFeature.csv", sep = ',')
adFeature = pd.read_csv("../data/adFeature.csv", sep = ',')
train_data = pd.read_csv("../data/train.csv", sep = ',')
test_data = pd.read_csv("../data/test2.csv", sep = ',')

print("------------------merge data-----------------")
all_data = train_data.append(test_data, ignore_index = True, sort = True)
all_data = pd.merge(all_data, adFeature, on = 'aid', how = 'left')
all_data = pd.merge(all_data, userFeature_data, on = 'uid', how = 'left')

all_data['instance_id'] = all_data.index
all_data.to_csv('../data/ffd/all_data.csv', index = False)

sys.exit()
all_data = pd.read_csv("../preliminary_contest_data/ffd/all_data.csv", sep = ',')

# all_data['sector'] = pd.cut(all_data.instance_id, bins = 5, labels = False)

all_data.drop(['kw1','kw2','kw3','topic1','topic2','topic3'], axis = 1, inplace = True)
# del adFeature,userFeature_data
gc.collect()

all_features = set(all_data.columns)
drop_set = set()

if not os.path.exists("../preliminary_contest_data/ffd"):
    print("未发现preliminary_contest_data中的ffd文件夹,创建ffd的文件夹")
    os.makedirs("../preliminary_contest_data/ffd")


portition_dict = defaultdict(dict)

print("------------------关于列表的特征-----------------")
#生成interest list，初始化占比列表
set_of_aid = set(all_data.aid)
# list_feature = ['interest1']
list_feature = ['interest1','interest2','interest3','interest4','interest5']
for count,feature in enumerate(list_feature):
    print("正在处理" + feature)
    all_data[feature] = all_data[feature].apply(lambda x:x.split(' ') if isinstance(x,str) else [])

    print("正在初始化" + feature)
    all_data[feature + '_portition_pos'] = np.nan
    all_data[feature + '_portition_neg'] = np.nan

    if not os.path.exists("../preliminary_contest_data/ffd/" + feature + "_portition.json"):
        print("生成" + feature + "占比字典")
        portition_dict[feature] = generate_portition_dict(all_data, feature)
        print("写入" + feature + "portition dict文件")
        restore_dict(path = "../preliminary_contest_data/ffd/" + feature + "_portition.json", store_dict = portition_dict)
    else:
        print("载入" + feature + "占比字典")
        portition_dict = load_dict(path = "../preliminary_contest_data/ffd/" + feature + "_portition.json")
    
    print("生成" + feature + "的占比list")
    all_data[feature + '_portition_pos'] = apply_parallel(df_grouped = all_data[[feature,'aid']].groupby(lambda x: x % 20), func = feature_pos_portition, feature_dict = portition_dict[feature], set_of_aid = set_of_aid, feature = feature)
    all_data[feature + '_portition_neg'] = apply_parallel(df_grouped = all_data[[feature,'aid']].groupby(lambda x: x % 20), func = feature_neg_portition, feature_dict = portition_dict[feature], set_of_aid = set_of_aid, feature = feature)    

    print("生成max,min,std,avg")
    all_data[feature + '_portition_pos_max'] = all_data[feature + '_portition_pos'].apply(lambda x:max(x) if x else -1)
    all_data[feature + '_portition_pos_min'] = all_data[feature + '_portition_pos'].apply(lambda x:min(x) if x else -1)
    all_data[feature + '_portition_pos_avg'] = all_data[feature + '_portition_pos'].apply(lambda x:np.mean(x) if x else -1)
    all_data[feature + '_portition_pos_std'] = all_data[feature + '_portition_pos'].apply(lambda x:np.std(x) if x else -1)

    all_data[feature + '_portition_neg_max'] = all_data[feature + '_portition_neg'].apply(lambda x:max(x) if x else -1)
    all_data[feature + '_portition_neg_min'] = all_data[feature + '_portition_neg'].apply(lambda x:min(x) if x else -1)
    all_data[feature + '_portition_neg_avg'] = all_data[feature + '_portition_neg'].apply(lambda x:np.mean(x) if x else -1)
    all_data[feature + '_portition_neg_std'] = all_data[feature + '_portition_neg'].apply(lambda x:np.std(x) if x else -1)
    
    
    all_data.drop([feature, feature + '_portition_pos', feature + '_portition_neg'], axis = 1, inplace = True)
    drop_set = drop_set|set([feature, feature + '_portition_pos', feature + '_portition_neg'])
    
    print(count, '/', len(list_feature))

# for i in set(all_data.sector):
#     pos_data = all_data[(all_data.sector != i)&(all_data.label == 1)]

#     for feature in ['gender']
#     pos_data

print("只保留instance_id和新添的特征")
drop_set = all_features - (set(['instance_id'])|drop_set)
all_data.drop(list(drop_set), axis = 1, inplace = True)

all_data.to_csv("../preliminary_contest_data/ffd/interest_feature.csv", index = False)
