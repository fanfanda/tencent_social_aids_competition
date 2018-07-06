import pandas as pd

print("------- load data ------")
all_data = pd.read_csv("../data/ffd/all_data.csv", sep = ',', usecols=['uid','productType','productId','label'])

for feature in ['productType','productId']:
    temp_feature = all_data[all_data.label == 1][['uid',feature]]
    temp_feature[feature] = temp_feature[feature].astype('str')
    temp = temp_feature.groupby('uid')[feature].agg(lambda x:' '.join(x)).reset_index(name = "pos_" + feature)
    print("去除重复")
    temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: x.split(' '))
    temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: list(set(x)))
    temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: ' '.join(x))
    #merge
    all_data = pd.merge(all_data, temp, on = 'uid', how = 'left')
    
    temp_feature = all_data[all_data.label == -1][['uid',feature]]
    temp_feature[feature] = temp_feature[feature].astype('str')
    temp = temp_feature.groupby('uid')[feature].agg(lambda x:' '.join(x)).reset_index(name = "neg_" + feature)
    print("去除重复")
    temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: x.split(' '))
    temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: list(set(x)))
    temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: ' '.join(x))

    #merge
    all_data = pd.merge(all_data, temp, on = 'uid', how = 'left')

test_feature = all_data[all_data.label.isna()][['uid','pos_productType','neg_productType','pos_productId','neg_productId']].drop_duplicates().reset_index(drop = True)
test_feature.to_csv("../data/split_data/test_uid_negandposadd.csv", index = False)
#去掉当前记录的属性
all_data['productType_num'] = all_data.groupby(['uid','label','productType']).productType.transform('count')

all_data.loc[(all_data.productType_num == 1)&(all_data.label == 1), 'pos_productType'] += " " + all_data.productType.astype('str')
all_data.loc[(all_data.productType_num == 1)&(all_data.label == 1),'pos_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == 1)].pos_productType.apply(lambda x: x.split(' '))

all_data.loc[(all_data.productType_num == 1)&(all_data.label == 1),'pos_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == 1)].pos_productType.apply(lambda x: list(filter(lambda item: item != x[-1],x)))
all_data.loc[(all_data.productType_num == 1)&(all_data.label == 1),'pos_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == 1)].pos_productType.apply(lambda x: " ".join(x))

all_data.loc[(all_data.productType_num == 1)&(all_data.label == -1), 'neg_productType'] += " " + all_data.productType.astype('str')
all_data.loc[(all_data.productType_num == 1)&(all_data.label == -1),'neg_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == -1)].neg_productType.apply(lambda x: x.split(' '))

all_data.loc[(all_data.productType_num == 1)&(all_data.label == -1),'neg_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == -1)].neg_productType.apply(lambda x: list(filter(lambda item: item != x[-1],x)))
all_data.loc[(all_data.productType_num == 1)&(all_data.label == -1),'neg_productType'] = all_data[(all_data.productType_num == 1)&(all_data.label == -1)].neg_productType.apply(lambda x: " ".join(x))

#去掉当前记录的属性
all_data['productId_num'] = all_data.groupby(['uid','label','productId']).productId.transform('count')

all_data.loc[(all_data.productId_num == 1)&(all_data.label == 1), 'pos_productId'] += " " + all_data.productId.astype('str')
all_data.loc[(all_data.productId_num == 1)&(all_data.label == 1),'pos_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == 1)].pos_productId.apply(lambda x: x.split(' '))

all_data.loc[(all_data.productId_num == 1)&(all_data.label == 1),'pos_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == 1)].pos_productId.apply(lambda x: list(filter(lambda item: item != x[-1],x)))
all_data.loc[(all_data.productId_num == 1)&(all_data.label == 1),'pos_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == 1)].pos_productId.apply(lambda x: " ".join(x))

all_data.loc[(all_data.productId_num == 1)&(all_data.label == -1), 'neg_productId'] += " " + all_data.productId.astype('str')
all_data.loc[(all_data.productId_num == 1)&(all_data.label == -1),'neg_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == -1)].neg_productId.apply(lambda x: x.split(' '))

all_data.loc[(all_data.productId_num == 1)&(all_data.label == -1),'neg_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == -1)].neg_productId.apply(lambda x: list(filter(lambda item: item != x[-1],x)))
all_data.loc[(all_data.productId_num == 1)&(all_data.label == -1),'neg_productId'] = all_data[(all_data.productId_num == 1)&(all_data.label == -1)].neg_productId.apply(lambda x: " ".join(x))

train_feature = all_data[~all_data.label.isna()][['uid','productType','productId','pos_productType','neg_productType','pos_productId','neg_productId','label']].drop_duplicates().reset_index(drop = True)
train_feature.to_csv("../data/split_data/train_uid_negandposadd.csv", index = False)