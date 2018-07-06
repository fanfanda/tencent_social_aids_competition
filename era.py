from mark import train
from tencent_utility import add_feature_612,add_feature_609
import sys
import gc
from generate_data import generate_data
excu_list = eval(sys.argv[1])
# add_feature_612("4", "_609", "_612")
# generate_data("4","_new")
# sys.exit()
for i in excu_list:
    print("第" + str(i) + "个模型")
    if i != 4:
        # add_feature_609(str(i), "_new", "_609")
        add_feature_612(str(i), "_new", "_612")
    else:
        print("passing")
    
    # generate_data(str(i),"_new")
    # generate_data(str(i),"_new")
    
    train(str(i),"_612","_new")
    
    # if i != 1:
    #     generate_data(str(i),"_v")
    # else:
    #     print("has gen")
    # add_feature_526(str(i), "_base", "_526")
    # train(str(i), "")

    gc.collect()