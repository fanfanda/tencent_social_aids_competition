import pandas as pd

print("------------------load data-----------------")
all_data = pd.read_csv("../data/ffd/all_data.csv", sep = ',')


aid = list(set(all_data.aid))

size = 5
version = "_5"
for i in range(size):
    temp_data = all_data[all_data.aid.isin(aid[i::size])]
    all = len(temp_data)
    pos = len(temp_data[temp_data.label == 1])
    print(i,pos,all,pos/all)
    temp_data.to_csv("../data/split_data/basedata/split_data_" + str(i) + version + ".csv", index = False)