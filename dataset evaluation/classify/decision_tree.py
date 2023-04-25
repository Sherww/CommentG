import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.model_selection import KFold

# 读取CSV文件，前三列为特征，最后一列为标签
df = pd.read_csv('Sample_result_feature.csv', header=None)
X = df.iloc[:, 1:5]
y = df.iloc[:, -1]

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=False)

acc_scores, prec_scores, rec_scores, f1_scores, hamming_losses = [], [], [], [], []

for train_index, test_index in kf.split(X):
    # 划分训练集和测试集
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    # 训练决策树模型
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = clf.predict(X_test)

    # 计算性能指标
    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred, average='macro'))
    rec_scores.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    hamming_losses.append(hamming_loss(y_test, y_pred))


# 输出每一折的性能指标
for i in range(len(acc_scores)):
    print(f'Fold {i+1} Accuracy: {acc_scores[i]:.4f}')
    print(f'Fold {i+1} Precision: {prec_scores[i]:.4f}')
    print(f'Fold {i+1} Recall: {rec_scores[i]:.4f}')
    print(f'Fold {i+1} F1-score: {f1_scores[i]:.4f}')
    print(f'Fold {i+1} Hamming Loss: {hamming_losses[i]:.4f}')

# 输出平均性能指标
print(f'Average Accuracy: {sum(acc_scores) / len(acc_scores):.4f}')
print(f'Average Precision: {sum(prec_scores) / len(prec_scores):.4f}')
print(f'Average Recall: {sum(rec_scores) / len(rec_scores):.4f}')
print(f'Average F1-score: {sum(f1_scores) / len(f1_scores):.4f}')
print(f'Average Hamming Loss: {sum(hamming_losses) / len(hamming_losses):.4f}')


with open('result_dct.txt','w',encoding='utf-8')as f:
    for i in range(len(acc_scores)):
        f.write(f'Fold {i+1} Accuracy: {acc_scores[i]:.4f}'+'\n')
        f.write(f'Fold {i+1} Precision: {prec_scores[i]:.4f}'+'\n')
        f.write(f'Fold {i+1} Recall: {rec_scores[i]:.4f}'+'\n')
        f.write(f'Fold {i+1} F1-score: {f1_scores[i]:.4f}'+'\n')
        f.write(f'Fold {i+1} Hamming Loss: {hamming_losses[i]:.4f}'+'\n')
    f.write(f'Average Accuracy: {sum(acc_scores) / len(acc_scores):.4f}'+'\n')
    f.write(f'Average Precision: {sum(prec_scores) / len(prec_scores):.4f}'+'\n')
    f.write(f'Average Recall: {sum(rec_scores) / len(rec_scores):.4f}'+'\n')
    f.write(f'Average F1-score: {sum(f1_scores) / len(f1_scores):.4f}'+'\n')
    f.write(f'Average Hamming Loss: {sum(hamming_losses) / len(hamming_losses):.4f}'+'\n')

