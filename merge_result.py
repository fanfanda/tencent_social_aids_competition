import pandas as pd
import os
for i in range(5):
    if i == 3:
        temp = pd.read_csv("./result/result_split" + str(i) + "_612.csv", sep = ',')
    else:
        temp = pd.read_csv("./result/result_split" + str(i) + "_612.csv", sep = ',')
    if i == 0:
        res = temp.reset_index(drop = True)
    else:
        res = res.append(temp, ignore_index = True)
# res = pd.read_csv("submission.csv", sep = ',')
test = pd.read_csv("../data/test2.csv", sep = ',')
test = pd.merge(test, res, on = ['aid','uid'], how = 'left')
test.to_csv("submission.csv", index = False)

os.system('zip ./ffda.zip ./submission.csv')