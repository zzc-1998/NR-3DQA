import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats

name_list = ['bag','banana','biscuits','cake','cauliflower','flowerpot','glasses_case','honeydew_melon','house','litchi','mushroom','pen_container','pineapple','ping-pong_bat','puer_tea','pumpkin','ship','statue','stone','tool_box']
train_name_list = ['bag','biscuits','cake','flowerpot','glasses_case','honeydew_melon','house','litchi','pen_container','ping-pong_bat','puer_tea','pumpkin','ship','statue','stone','tool_box']
test_name_list = ['banana','cauliflower','mushroom','pineapple']


def get_data(train_name_list,test_name_list):
    feature_data = pd.read_csv("features.csv",index_col = 0,keep_default_na=False)
    feature_data = feature_data[feature_data.columns.values]
    # print(feature_data)
    score_data = pd.read_csv("mos.csv")
    mos = score_data['mos'].tolist()
    total_obj_names = score_data['name']
    score_data = pd.read_csv("mos.csv",index_col = 0)
    train_set = []
    train_score = []
    test_set = []
    test_score = []
    for name in train_name_list:
        obj_names = []
        for obj in total_obj_names: 
            if name in obj: 
                obj_names.append(obj)
        for i in obj_names:
            data = feature_data.loc[i,:].tolist()
            while '' in data:
                data.remove('')
            train_set.append(data)
            train_score.append(score_data.loc[i,:].tolist()[0])
    
    for name in test_name_list:
        obj_names = []
        for obj in total_obj_names: 
            if name in obj: 
                obj_names.append(obj)
        for i in obj_names:
            data = feature_data.loc[i,:].tolist()
            while '' in data:
                data.remove('')
            test_set.append(data)
            test_score.append(score_data.loc[i,:].tolist()[0])
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set,np.array(train_score)/100,test_set,np.array(test_score)/100
    


train_set,train_score,test_set,test_score = get_data(train_name_list,test_name_list)
svr = SVR(kernel='rbf', degree=4, gamma='scale', coef0=0.1, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
svr.fit(train_set, train_score)
predict_score = svr.predict(test_set) 
print("PLCC:  "+ str(stats.pearsonr(predict_score, test_score)[0]))
print("SRCC:  "+ str(stats.spearmanr(predict_score, test_score)[0]))
print("KRCC:  "+ str(stats.stats.kendalltau(predict_score, test_score)[0]))
print("RMSE:  "+ str(np.sqrt(((predict_score * 100 - test_score * 100) ** 2).mean())))
