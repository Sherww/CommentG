import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

# 读取CSV数据
df = pd.read_csv('Sample_result_feature.csv', header=None)

# 提取特征和标签
X = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values

# 将标签转换为one-hot编码
num_classes = len(np.unique(y))
y = np.eye(num_classes)[y]

# 将文本数据转换为数值序列
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X[:, 0])
X = tokenizer.texts_to_sequences(X[:, 0])
X = pad_sequences(X, maxlen=50)

# 定义十折交叉验证
kfold = KFold(n_splits=10, shuffle=False)
# kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 定义指标列表
acc_list = []
precision_list = []
recall_list = []
f1_score_list = []
hamming_losses=[]

# 循环进行十折交叉验证
for train_idx, test_idx in kfold.split(X, y):
    # 划分训练集和测试集
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 构建CNN模型
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=50))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 编译和训练模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    hamming=hamming_loss(y_test, y_pred)
    # 将指标加入列表
    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)
    hamming_losses.append(hamming_loss(y_test, y_pred))

    # 输出指标
    print(f"accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1-score: {f1:.4f}, hamming:{hamming:.4f}")

# 输出十折交叉验证结果的平均指标
print("Average of accuracy: {:.4f}".format(np.mean(acc_list)))
print("Average of precision: {:.4f}".format(np.mean(precision_list)))
print("Average of recall: {:.4f}".format(np.mean(recall_list)))
print("Average of f1-score: {:.4f}".format(np.mean(f1_score_list))) 
print("Average of hamming: {:.4f}".format(np.mean(hamming_losses))) 


with open('result_cnn_without_feature.txt','w',encoding='utf-8')as f:
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
