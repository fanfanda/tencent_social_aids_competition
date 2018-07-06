import pandas as pd
import numpy as np
import gc,os
from tencent_utility import *

print("------------------load data-----------------")
all_data = pd.read_csv("../data/ffd/all_data.csv", sep = ',', usecols = ['instance_id','aid','gender','house','consumptionAbility','education','label'])


all_data['sector'] = pd.cut(all_data.instance_id, bins = 10, labels = False)
all_data['house'] = all_data.house.fillna(0)
all_features = set(all_data.columns)
drop_set = set()

for feature in ['gender','house','consumptionAbility','education']:
    print("计算" + feature + "_pos占比")
    sector_data = normal_feature_pos_portition(all_data[['sector','label','aid',feature]], feature)
    print("merge" + feature + "_pos占比")
    all_data = pd.merge(all_data, sector_data, how = 'left', on = ['aid',feature,'sector'])

    print("计算" + feature + "_neg占比")
    sector_data = normal_feature_neg_portition(all_data[['sector','label','aid',feature]], feature)
    print("merge" + feature + "_neg占比")
    all_data = pd.merge(all_data, sector_data, how = 'left', on = ['aid',feature,'sector'])



print("只保留instance_id和新添的特征")
drop_set = all_features - (set(['instance_id'])|drop_set)
all_data.drop(list(drop_set), axis = 1, inplace = True)

all_data.to_csv("../data/ffd/normal_feature.csv", index = False)
