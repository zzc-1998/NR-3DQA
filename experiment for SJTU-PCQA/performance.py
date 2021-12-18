import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import scale
import pandas as pd
from scipy import stats
def get_feature_name():
    features = ["l","a","b","curvature","anisotropy","linearity","planarity","sphericity"]
    nss_list = [["mean","std","entropy"],["ggd1","ggd2"],["aggd1","aggd2","aggd3","aggd4"],["gamma1","gamma2"]]
    total_nss_list = ["mean","std","entropy","ggd1","ggd2","aggd1","aggd2","aggd3","aggd4","gamma1","gamma2"]
    feature_list = []
    for feature in features[:]:
        for nss in total_nss_list[:3]:
             feature_list.append(feature + "_" + nss)
    return feature_list

def get_data(train_name_list,test_name_list):
    feature_data = pd.read_csv("pc_features.csv",index_col = 0,keep_default_na=False)
    feature_data = feature_data[get_feature_name()]
    score_data = pd.read_csv("score.csv")
    train_set = []
    train_score = []
    test_set = []
    test_score = []
    for name in train_name_list:
        score = score_data[name].tolist()
        train_score = train_score + score
        for i in range(42):
            name_pc = name+str(i)
            data = feature_data.loc[name_pc,:].tolist()
            while '' in data:
                data.remove('')
            train_set.append(data)
    
    for name in test_name_list:
        score = score_data[name].tolist()
        test_score = test_score + score 
        for i in range(42):
            name_pc = name+str(i)
            data = feature_data.loc[name_pc,:].tolist()
            while '' in data:
                data.remove('')
            test_set.append(data)
    return scale(train_set),train_score,scale(test_set),test_score




plcc = []
srcc =[]
rmse = []
krcc = []
# generate train_name_list and test_name_list
train_name_list = ['Romanoillamp','loot','ULB Unicorn','longdress','statue','shiva','hhi']
test_name_list = ['redandblack','soldier']
# get data
train_set,train_score,test_set,test_score = get_data(train_name_list,test_name_list)
# begin training
svr = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
svr.fit(train_set, train_score)
predict_score = svr.predict(test_set)
print("SRCC:  "+ str((stats.pearsonr(predict_score, test_score)[0])))
print("PLCC:  "+ str(stats.spearmanr(predict_score, test_score)[0]))
print("KRCC:  "+ str(np.sqrt(((predict_score-test_score) ** 2).mean())))
print("RMSE:  "+ str(stats.stats.kendalltau(predict_score, test_score)[0]))
