import pandas as pd
import gc,sys
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def generate_data(sector,version):
    print("------------------load data-----------------")
    # sector = "1"
    all_data = pd.read_csv("../data/split_data/basedata/split_data_" + sector + "_5.csv", sep = ',')
    # normal_feature = pd.read_csv("../preliminary_contest_data/ffd/normal_feature.csv", sep = ',')
    # interest_feature = pd.read_csv("../preliminary_contest_data/ffd/interest_feature.csv", sep = ',')
    train_uid_negandpos = pd.read_csv("../data/split_data/train_uid_negandpos.csv", sep = ',')
    test_uid_negandpos = pd.read_csv("../data/split_data/test_uid_negandpos.csv", sep = ',')
    test_uid_negandpos = test_uid_negandpos.drop_duplicates().reset_index(drop = True)
    print("------------------merge data-----------------")
    # all_data = pd.merge(all_data, normal_feature, on = 'instance_id', how = 'left')
    # all_data = pd.merge(all_data, interest_feature, on = 'instance_id', how = 'left')

    all_features = set(all_data.columns)

    # fill_value = max(all_data[~all_data.LBS.isna()].LBS) + 1
    # all_data['LBS'].fillna(fill_value, inplace = True)

    # fill_value = max(all_data[~all_data.house.isna()].house) + 1
    # all_data['house'].fillna(fill_value, inplace = True)

    # fill_value = max(all_data[~all_data.gender.isna()].gender) + 1
    # all_data[''].fillna(fill_value, inplace = True)

    category_feature=['advertiserId','campaignId','creativeId','productId','aid','education','carrier','adCategoryId','productType']
    
    remainder_feature = ['creativeSize','age','consumptionAbility','LBS','house','gender']
    vector_feature = list(all_features - (set(remainder_feature)|set(['instance_id','label','uid'])|set(category_feature)))
    vector_feature = list(set(vector_feature)|set(['pos_aid','neg_aid']))
    # vector_feature = list(all_features - (set(remainder_feature)|set(['instance_id','label','uid','interest1','interest3','interest4','interest5','kw3','topic3'])|set(one_hot_feature)))
    for feature in category_feature:
        try:
            if feature == 'aid':
                continue
            all_data[feature] = LabelEncoder().fit_transform(all_data[feature].apply(int))
        except:
            print("ooha")
            sys.exit()
            # all_data[feature] = LabelEncoder().fit_transform(all_data[feature])

    all_train_data = all_data[~all_data.label.isna()]
    test_data = all_data[all_data.label.isna()]

    all_train_data = pd.merge(all_train_data, train_uid_negandpos, on = ['uid','aid'], how = 'left')
    test_data = pd.merge(test_data, test_uid_negandpos, on = ['uid'], how = 'left')

    all_train_data_x = all_train_data[category_feature + remainder_feature]
    test_data_x = test_data[category_feature + remainder_feature]

    all_train_data_y = all_train_data[['instance_id','label']]
    all_train_data_y.to_csv("../data/split_data/label/split_data_" + sector + version + "_label.csv", index = False)

    # enc = OneHotEncoder()

    # for feature in one_hot_feature:
    #     enc.fit(all_data[feature].values.reshape(-1, 1))
    #     train_temp = enc.transform(all_train_data[feature].values.reshape(-1, 1))
    #     test_temp = enc.transform(test_data[feature].values.reshape(-1, 1))
    #     all_train_data_x = sparse.hstack((all_train_data_x, train_temp))
    #     test_data_x = sparse.hstack((test_data_x, test_temp))
    #     print(feature + ' finish')
    # print('one-hot prepared !')
    cv = CountVectorizer(token_pattern='\w+', max_features = 20000) # max_features = 20000
    print("开始cv.....")
    for feature in vector_feature:
        print("生成" + feature + "cv")
        cv.fit(all_train_data[feature].astype('str'))
        print("开始转换" + feature + "cv")
        train_temp=cv.transform(all_train_data[feature].astype('str'))
        test_temp=cv.transform(test_data[feature].astype('str'))
        
        all_train_data_x = sparse.hstack((all_train_data_x,train_temp))
        test_data_x = sparse.hstack((test_data_x,test_temp))

        print(feature + " is over")
    print('cv prepared')

    sparse.save_npz("../data/split_data/npz/train_split_" + sector + version + ".npz",all_train_data_x)
    sparse.save_npz("../data/split_data/npz/test_split_" + sector + version + ".npz",test_data_x)
    
    print("train shape: ", all_train_data_x.shape)
    print("test shape: ", test_data_x.shape)
    print("len category_feature: ", len(category_feature))
    test_data[['instance_id','uid','aid']].to_csv("../data/split_data/testdata/test_data_uaid_" + sector + version + ".csv", index = False)
    print('-------------')
