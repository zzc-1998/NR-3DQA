import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy import stats

name_list = ['bag','banana','biscuits','cake','cauliflower','flowerpot','glasses_case','honeydew_melon','house','litchi','mushroom','pen_container','pineapple','ping-pong_bat','puer_tea','pumpkin','ship','statue','stone','tool_box']


# get data according to the train test name lists, return scaled train and test set
def get_data(train_name_list,test_name_list):
    feature_data = pd.read_csv("features.csv",index_col = 0,keep_default_na=False)
    feature_data = feature_data[feature_data.columns.values]
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
            train_set.append(data)
            train_score.append(score_data.loc[i,:].tolist()[0])
    
    for name in test_name_list:
        obj_names = []
        for obj in total_obj_names: 
            if name in obj: 
                obj_names.append(obj)
        for i in obj_names:
            data = feature_data.loc[i,:].tolist()
            test_set.append(data)
            test_score.append(score_data.loc[i,:].tolist()[0])
    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set,np.array(train_score)/100,test_set,np.array(test_score)/100
    




if __name__ == '__main__':
    plcc = []
    srcc =[]
    krcc = []
    for i in range(5):
        # generate 5 folder cross validation split name lists
        train_name_list = name_list.copy()
        # get test set and remove the test content from the training set
        test_name_list = [train_name_list.pop(4*i + 3),train_name_list.pop(4*i + 2),train_name_list.pop(4*i + 1),train_name_list.pop(4*i)]
        print('Begin split ' + str(i+1) + ' and use the following list as test set:')
        print(test_name_list)
        # get the data according to the name lists
        train_set,train_score,test_set,test_score = get_data(train_name_list,test_name_list)
        # begin training and predicting
        print('Begin training!')
        svr = SVR(kernel='rbf')
        svr.fit(train_set, train_score)
        predict_score = svr.predict(test_set)
        # record the result
        plcc.append(stats.pearsonr(predict_score, test_score)[0])
        srcc.append(stats.spearmanr(predict_score, test_score)[0])
        krcc.append(stats.stats.kendalltau(predict_score, test_score)[0])
        print('Training complete!')
        print('------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------------------')
    print('Final Results presentation:')
    print("SRCC:  "+ str(sum(srcc)/len(srcc)))
    print("PLCC:  "+ str(sum(plcc)/len(plcc)))
    print("KRCC:  "+ str(sum(krcc)/len(krcc)))


