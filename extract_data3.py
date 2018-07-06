import pandas as pd

print("------------------load data-----------------")
all_data = pd.read_csv("../data/ffd/all_data.csv", sep = ',')
# ['instance_id','interest1','interest2','interest3','interest4','topic1','topic2','topic3','kw1','kw2','kw3','label']
# ['interest1','interest2','interest3','interest4','topic1','topic2','topic3','kw1','kw2','kw3']
# 'advertiserId','productId','adCategoryId'
for feature in ['interest1','interest2','interest3','interest4','topic1','topic2','topic3','kw1','kw2','kw3','marriageStatus','appIdAction','appIdInstall','os','ct']:
    feature_list = []
    all_data[feature] = all_data[feature].apply(lambda x:x.split(' ') if isinstance(x,str) else [])

    train_data = all_data[~all_data.label.isna()]
    for item in train_data[feature]:
        feature_list += item
    
    feature_set = set(feature_list)
    print("过滤" + feature)
    all_data.loc[all_data.label.isna(),feature] = all_data[all_data.label.isna()][feature].apply(lambda x:list(filter(lambda item:item in feature_set,x)))
    all_data[feature] = all_data[feature].apply(lambda x:" ".join(x))

all_data.to_csv("../data/ffd/filter_all_data.csv", index = False)