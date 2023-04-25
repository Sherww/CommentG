import csv
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, explained_variance_score ,accuracy_score ,precision_score ,recall_score,f1_score,hamming_loss
import random
from tqdm import tqdm     
import random
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns



# why 0  how 1  what 2
def load_dataset(filename):
    file_reader = csv.reader(open(filename,'rt'), delimiter=',')
    rows=[]
    for row in file_reader:
        rows.append(row)

    X, y = [], []
    for dat in rows:
        X.append(dat[1:5])
        y.append(dat[-1])

 
    # print(X)
    #Extract feature names
    feature_names = np.array(['why_key','how_key','comm_len','com_code_len'])
    
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names
def main():
    # 定义指标列表
    acc_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    hamming_losses=[]
    for fold in range(10):
        X, y, feature_names = load_dataset("Sample_result_feature.csv")
    
        '''
        :param K: 要把数据集分成的份数。如十次十折取K=10
        :param fold: 要取第几折的数据。如要取第5折则 flod=5
        :param data: 需要分块的数据
        :param label: 对应的需要分块标签
        :return: 对应折的训练集、测试集和对应的标签
        '''
        split_list = []
        kf = KFold(n_splits=10)
        for train, test in kf.split(X):
                split_list.append(train.tolist())
                split_list.append(test.tolist())
        #取第几折就改变2*后面的那个数字，这是第三折数据
        train,test=split_list[2 * fold],split_list[2 * fold + 1]
        X_train=X[train]
        y_train= y[train]
        X_test=X[test]
        y_test=y[test] 
 
        rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=2,random_state=0)
        # !!!!!!!!!!!!对比一下看看，这俩谁效果好，上面的效果好...
        # rf_classifier = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2,random_state=0)
        

        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        evs = explained_variance_score(y_test,y_pred)
        accu = accuracy_score(y_pred,y_test)
        # pre1 = precision_score(y_test,y_pred,average='micro')
        pre2 = precision_score(y_test,y_pred,average='macro')
        # pre3 = precision_score(y_test,y_pred,average='weighted')
    
        # rec1 = recall_score(y_test,y_pred,average='micro')
        rec2 = recall_score(y_test,y_pred,average='macro')
        # rec3 = recall_score(y_test,y_pred,average='weighted')
        # f1_score1=f1_score(y_test, y_pred, average='micro')
        f1_score2=f1_score(y_test, y_pred, average='macro')
        # f1_score3=f1_score(y_test, y_pred, average='weighted')
        hamming=hamming_loss(y_test, y_pred)
        # 将指标加入列表
        acc_list.append(accu)
        precision_list.append(pre2)
        recall_list.append(rec2)
        f1_score_list.append(f1_score2)
        hamming_losses.append(hamming_loss(y_test, y_pred))

        # 输出指标
        print(f"accuracy: {accu:.4f}, precision: {pre2:.4f}, recall: {rec2:.4f}, f1-score: {f1_score2:.4f}")
            
    # 输出十折交叉验证结果的平均指标
    print("Average of accuracy: {:.4f}".format(np.mean(acc_list)))
    print("Average of precision: {:.4f}".format(np.mean(precision_list)))
    print("Average of recall: {:.4f}".format(np.mean(recall_list)))
    print("Average of f1-score: {:.4f}".format(np.mean(f1_score_list))) 
    print("Average of hamming: {:.4f}".format(np.mean(hamming_losses))) 

    # color = sns.color_palette()
    # sns.set_style('darkgrid')
    # features_list = feature_names
    # feature_importance = rf_classifier.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
     
    # plt.figure(figsize=(8, 7))
    # plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    # plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    # plt.xlabel('Importance')
    # plt.title('Feature importances')
    # plt.draw()
    # plt.show()

    with open('result_random.txt','w',encoding='utf-8')as f:
        for i in range(len(acc_list)):
            f.write(f'Fold {i+1} Accuracy: {acc_list[i]:.4f}'+'\n')
            f.write(f'Fold {i+1} Precision: {precision_list[i]:.4f}'+'\n')
            f.write(f'Fold {i+1} Recall: {recall_list[i]:.4f}'+'\n')
            f.write(f'Fold {i+1} F1-score: {f1_score_list[i]:.4f}'+'\n')
            f.write(f'Fold {i+1} Hamming Loss: {hamming_losses[i]:.4f}'+'\n')
        f.write(f'Average Accuracy: {sum(acc_list) / len(acc_list):.4f}'+'\n')
        f.write(f'Average Precision: {sum(precision_list) / len(precision_list):.4f}'+'\n')
        f.write(f'Average Recall: {sum(recall_list) / len(recall_list):.4f}'+'\n')
        f.write(f'Average F1-score: {sum(f1_score_list) / len(f1_score_list):.4f}'+'\n')
        f.write(f'Average Hamming Loss: {sum(hamming_losses) / len(hamming_losses):.4f}'+'\n')    
if __name__=='__main__':

        main()





#十折交叉验证参考的代码
# import numpy as np  # 导入numpy包
# from sklearn.model_selection import KFold  # 从sklearn导入KFold包

# #输入数据推荐使用numpy数组，使用list格式输入会报错
# def K_Flod_spilt(K,fold,data,label):
#     '''
#     :param K: 要把数据集分成的份数。如十次十折取K=10
#     :param fold: 要取第几折的数据。如要取第5折则 flod=5
#     :param data: 需要分块的数据
#     :param label: 对应的需要分块标签
#     :return: 对应折的训练集、测试集和对应的标签
#     '''
#     split_list = []
#     kf = KFold(n_splits=K)
#     for train, test in kf.split(data):
#         split_list.append(train.tolist())
#         split_list.append(test.tolist())
#     train,test=split_list[2 * fold],split_list[2 * fold + 1]
#     return  data[train], data[test], label[train], label[test]  #已经分好块的数据集