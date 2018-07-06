import pandas as pd
import multiprocessing
from collections import defaultdict
import numpy as np
import sys,json,codecs
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

def normal_feature_pos_portition(all_data, feature):
    for i in range(10):
        sector_aid = all_data[(all_data.label == 1)&(all_data.sector != i)][['aid',feature]]
        sector_aid = sector_aid.groupby(['aid',feature]).size().reset_index(name = feature + '_aid_count_pos')
        sector_aid[feature + '_aid_count_pos'] = sector_aid[feature + '_aid_count_pos'] / sector_aid.groupby(['aid'])[feature + '_aid_count_pos'].transform('sum')
        sector_aid['sector'] = i
        if i == 0:
            sector_data = sector_aid.reset_index(drop = True)
        else:
            sector_data = sector_data.append(sector_aid, ignore_index = True)
    return sector_data
def normal_feature_neg_portition(all_data, feature):
    for i in range(10):
        sector_aid = all_data[(all_data.label == -1)&(all_data.sector != i)][['aid',feature]]
        sector_aid = sector_aid.groupby(['aid',feature]).size().reset_index(name = feature + '_aid_count_neg')
        sector_aid[feature + '_aid_count_neg'] = sector_aid[feature + '_aid_count_neg'] / sector_aid.groupby(['aid'])[feature + '_aid_count_neg'].transform('sum')
        sector_aid['sector'] = i
        if i == 0:
            sector_data = sector_aid.reset_index(drop = True)
        else:
            sector_data = sector_data.append(sector_aid, ignore_index = True)
    return sector_data

def add_features(func, lists, args = None):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i,args)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result

def apply_parallel(df_grouped, func, feature_dict, set_of_aid, feature):
    """利用 Parallel 和 delayed 函数实现并行运算"""
    results = Parallel(n_jobs = 20)(delayed(func)(group, feature_dict, set_of_aid, feature) for name, group in df_grouped)
    return pd.concat(results)

def helper_feature_portition(temp_list, aid, feature_dict, sign):
    #过滤掉没有出现在训练样本中的interest
    temp_list = list(filter(lambda interest:interest in feature_dict['portition_' + sign + '_' + str(aid)].keys(), temp_list))
    return list(map(lambda x:feature_dict['portition_' + sign + '_' + str(aid)][x], temp_list))
    

def feature_pos_portition(all_data, feature_dict, set_of_aid, feature):
    for aid in set_of_aid:
        all_data.loc[all_data.aid == aid,feature + '_portition_pos'] = all_data[all_data.aid == aid][feature].apply(lambda x:helper_feature_portition(x,aid,feature_dict,'pos'))
    return all_data[feature + '_portition_pos']

def feature_neg_portition(all_data, feature_dict, set_of_aid, feature):
    for aid in set_of_aid:
        all_data.loc[all_data.aid == aid,feature + '_portition_neg'] = all_data[all_data.aid == aid][feature].apply(lambda x:helper_feature_portition(x,aid,feature_dict,'neg'))
    return all_data[feature + '_portition_neg']

def generate_portition_dict(all_data, args):
    print("正在生成第" + args + "的的占比字典")
    set_of_aid = set(all_data.aid)
    feature_dict = defaultdict(dict)
    all = len(set_of_aid)
    for count,aid in enumerate(set_of_aid):
        #生成正样本的各个feature占比字典
        all_feature_list = []
        temp_data = all_data[(all_data.aid == aid)&(all_data.label == 1)][args]
        len_temp_data = len(temp_data)
        for feature_list in temp_data:
            all_feature_list += feature_list
        temp_set_feature = set(all_feature_list)
        for item in temp_set_feature:
            feature_dict['portition_pos_' + str(aid)][item] = all_feature_list.count(item) / len_temp_data

        #生成负样本的各个feature占比字典
        all_feature_list = []
        temp_data = all_data[(all_data.aid == aid)&(all_data.label == -1)][args]
        len_temp_data = len(temp_data)
        for feature_list in temp_data:
            all_feature_list += feature_list
        temp_set_feature = set(all_feature_list)
        for item in temp_set_feature:
            feature_dict['portition_neg_' + str(aid)][item] = all_feature_list.count(item) / len_temp_data
        
        view_bar(count, all)
    return feature_dict

def restore_dict(path,store_dict):
    with codecs.open(path,'a', 'utf-8') as outf:
        json.dump(store_dict, outf, ensure_ascii=False)
        outf.write('\n')

def load_dict(path):
    data = []
    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    return data[0]


#进度条函数
def view_bar(num, all_num):
    sys.stdout.write('\r')
    if num == all_num - 1:
        sys.stdout.write("%s%%|%s" % (int((num/all_num) * 80), int((num/all_num) * 80) * '#' + '| \n'))
    else:
        sys.stdout.write("%s%%|%s" % (int((num/all_num) * 80), int((num/all_num) * 80) * '#'))
    sys.stdout.flush()

def generate_npz(sector, add_feature, old_name, new_name):
    all_train_data_x = sparse.load_npz('../data/split_data/npz/train_split_' + sector + old_name + '.npz')#训练集特征
    test_data_x = sparse.load_npz('../data/split_data/npz/test_split_' + sector + old_name + '.npz')#测试集特征

    train_data = pd.read_csv('../data/split_data/label/split_data_' + sector +  '_new_label.csv', sep = ',', usecols = ['instance_id'])#训练集标签
    train_data = pd.merge(train_data, add_feature, on = 'instance_id', how = 'left')
    
    add_features = set(train_data.columns) - set(['instance_id']) - set(['pos_productType','neg_productType','pos_productId','neg_productId'])

    test_data = pd.read_csv("../data/split_data/testdata/test_data_uaid_" + sector + "_new.csv", sep = ',', usecols = ['instance_id'])
    test_data = pd.merge(test_data, add_feature, on = 'instance_id', how = 'left')

    cv = CountVectorizer(token_pattern='\w+', max_features = 20000) # max_features = 20000
    print("开始cv.....")
    for feature in ['pos_productType','neg_productType','pos_productId','neg_productId']:
        print("生成" + feature + "cv")
        cv.fit(train_data[feature].astype('str'))
        print("开始转换" + feature + "cv")
        train_temp=cv.transform(train_data[feature].astype('str'))
        test_temp=cv.transform(test_data[feature].astype('str'))
        
        all_train_data_x = sparse.hstack((all_train_data_x,train_temp))
        test_data_x = sparse.hstack((test_data_x,test_temp))

        print(feature + " is over")
    print('cv prepared')

    add_all_train_data_x = train_data[list(add_features)]
    add_test_data_x = test_data[list(add_features)]

    print("------------------add data-----------------")
    all_train_data_x = sparse.hstack((all_train_data_x, add_all_train_data_x))
    test_data_x = sparse.hstack((test_data_x, add_test_data_x))


    sparse.save_npz('../data/split_data/npz/train_split_' + sector + new_name + '.npz', all_train_data_x)
    sparse.save_npz('../data/split_data/npz/test_split_' + sector + new_name +'.npz', test_data_x)


def add_feature_607(sector, old_name, new_name):
    print("------------------load data-----------------")
    all_data = pd.read_csv("../data/split_data/basedata/split_data_" + sector + ".csv", sep = ',', usecols = ['instance_id','aid','gender','house','consumptionAbility','education','label','carrier','age','advertiserId','campaignId','creativeId','adCategoryId','productId','productType'])
    all_data.loc[all_data.label == -1, 'label'] = 0

    all_data['sector'] = pd.cut(all_data.instance_id, bins = 10, labels = False)
    all_data['house'] = all_data.house.fillna(-1)
    all_data['gender'] = all_data.gender.fillna(-1)
    all_data['house'] = all_data.house.apply(lambda x:int(x))
    all_data['gender'] = all_data.gender.apply(lambda x:int(x))

    all_features = set(all_data.columns)
    drop_set = set()

    for feature in ['aid','gender','house','consumptionAbility','education','carrier','age','advertiserId','campaignId','creativeId','adCategoryId','productId','productType']:
        print("计算" + feature + "转换率")
        all_data[feature + '_convert'] = all_data.groupby([feature]).label.transform('sum') / all_data.groupby([feature]).instance_id.transform('nunique')
        # sector_data = normal_feature_pos_portition(all_data[['sector','label','aid',feature]], feature)
        # print("merge" + feature + "_pos占比")
        # all_data = pd.merge(all_data, sector_data, how = 'left', on = ['aid',feature,'sector'])

        # print("计算" + feature + "_neg占比")
        # sector_data = normal_feature_neg_portition(all_data[['sector','label','aid',feature]], feature)
        # print("merge" + feature + "_neg占比")
        # all_data = pd.merge(all_data, sector_data, how = 'left', on = ['aid',feature,'sector'])

    print("只保留instance_id和新添的特征")
    drop_set = all_features - (set(['instance_id'])|drop_set)
    all_data.drop(list(drop_set), axis = 1, inplace = True)
    
    all_data.to_csv("../data/split_data/feature/split_data_" + sector + new_name + ".csv", index = False)
    print("generate npz files")
    generate_npz(sector, all_data, old_name, new_name)

def add_feature_609(sector, old_name, new_name):
    print("------------------load data-----------------")
    all_data = pd.read_csv("../data/split_data/basedata/split_data_" + sector + "_5.csv", sep = ',', usecols = ['instance_id','topic1','topic2','topic3','kw1','kw2','kw3','interest1','interest2','interest3','interest4','interest5','label','uid','aid'])
    
    train_uid_negandpos = pd.read_csv("../data/split_data/train_uid_negandpos.csv", sep = ',')
    test_uid_negandpos = pd.read_csv("../data/split_data/test_uid_negandpos.csv", sep = ',')
    test_uid_negandpos = test_uid_negandpos.drop_duplicates().reset_index(drop = True)

    all_train_data = all_data[~all_data.label.isna()]
    test_data = all_data[all_data.label.isna()]

    all_train_data = pd.merge(all_train_data, train_uid_negandpos, on = ['uid','aid'], how = 'left')
    test_data = pd.merge(test_data, test_uid_negandpos, on = ['uid'], how = 'left')

    all_data = all_train_data.append(test_data, ignore_index = True)
    all_features = set(all_data.columns)
    drop_set = set()

    for feature in ['topic1','topic2','topic3','kw1','kw2','kw3','interest1','interest2','interest3','interest4','interest5','pos_aid','neg_aid']:
        print("计算" + feature + "长度")
        all_data['lens_' + feature] = all_data[feature].apply(lambda x:len(x.split(' ')) if isinstance(x,str) else 0)
    #计算转换率
    print("计算转换率")
    all_data['uid_convert'] = -1
    all_data.loc[all_data.lens_pos_aid + all_data.lens_neg_aid != 0, 'uid_convert'] = all_data.lens_pos_aid / (all_data.lens_pos_aid + all_data.lens_neg_aid)

    print("只保留instance_id和新添的特征")
    drop_set = all_features - (set(['instance_id'])|drop_set)
    all_data.drop(list(drop_set), axis = 1, inplace = True)
    
    all_data.to_csv("../data/split_data/feature/split_data_" + sector + new_name + ".csv", index = False)
    print("generate npz files")
    generate_npz(sector, all_data, old_name, new_name)

def add_feature_612(sector, old_name, new_name):
    print("------------------load data-----------------")
    all_data = pd.read_csv("../data/split_data/basedata/split_data_" + sector + "_5.csv", sep = ',', usecols = ['instance_id','productType','productId','label','uid'])
    all_features = set(all_data.columns)

    train_uid_negandposa = pd.read_csv("../data/split_data/train_uid_negandposadd.csv", sep = ',', usecols = ['uid','productType','pos_productType','neg_productType','label'])
    train_uid_negandposa = train_uid_negandposa.drop_duplicates().reset_index(drop = True)
    train_uid_negandposb = pd.read_csv("../data/split_data/train_uid_negandposadd.csv", sep = ',', usecols = ['uid','productId','pos_productId','neg_productId','label'])
    train_uid_negandposb = train_uid_negandposb.drop_duplicates().reset_index(drop = True)

    test_uid_negandpos = pd.read_csv("../data/split_data/test_uid_negandposadd.csv", sep = ',')
    test_uid_negandpos = test_uid_negandpos.drop_duplicates().reset_index(drop = True)

    all_train_data = all_data[~all_data.label.isna()]
    test_data = all_data[all_data.label.isna()]
    print("train:",len(all_train_data),"test:",len(test_data))
    all_train_data = pd.merge(all_train_data, train_uid_negandposa, on = ['uid','productType','label'], how = 'left')
    all_train_data = pd.merge(all_train_data, train_uid_negandposb, on = ['uid','productId','label'], how = 'left')

    test_data = pd.merge(test_data, test_uid_negandpos, on = ['uid'], how = 'left')
    print("after train:",len(all_train_data),"test:",len(test_data))
    all_data = all_train_data.append(test_data, ignore_index = True)
    

    drop_set = set()

    print("只保留instance_id和新添的特征")
    drop_set = all_features - (set(['instance_id'])|drop_set)
    all_data.drop(list(drop_set), axis = 1, inplace = True)
    
    all_data.to_csv("../data/split_data/feature/split_data_" + sector + new_name + ".csv", index = False)
    print("generate npz files")
    generate_npz(sector, all_data, old_name, new_name)

    
