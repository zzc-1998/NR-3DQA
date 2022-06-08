import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy import stats

# get data according to the train test name lists, return scaled train and test set
def get_data(train_name_list,test_name_list):
    feature_data = pd.read_csv("features.csv",index_col = 0,keep_default_na=False)
    feature_data = feature_data[feature_data.columns.values]
    score_data = pd.read_csv("mos.csv")
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
            train_set.append(data)
    
    for name in test_name_list:
        score = score_data[name].tolist()
        test_score = test_score + score
        for i in range(42):
            name_pc = name+str(i)
            data = feature_data.loc[name_pc,:].tolist()
            test_set.append(data)
    # preprocessing      
    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set,np.array(train_score)/10,test_set,np.array(test_score)/10




if __name__ == '__main__':
    plcc = []
    srcc =[]
    krcc = []
    cnt = 0
    # begin 9-folder cross data validation split
    for i in range(9):
        cnt =cnt+1
        print(cnt)
        # generate train_name_list and test_name_list
        train_name_list = ['redandblack','Romanoillamp','loot','soldier','ULB Unicorn','longdress','statue','shiva','hhi']
        test_name_list = [train_name_list.pop(i)]
        # get data
        print('Begin split ' + str(i+1) + ' and use the following list as test set:')
        print(test_name_list)
        train_set,train_score,test_set,test_score = get_data(train_name_list,test_name_list)
        # begin training
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

    print("PLCC:  "+ str(sum(plcc)/len(plcc)))
    print("SRCC:  "+ str(sum(srcc)/len(srcc)))
    print("KRCC:  "+ str(sum(krcc)/len(krcc)))

