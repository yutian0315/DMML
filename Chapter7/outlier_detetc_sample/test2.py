# -*- coding:utf-8 -*-

import numpy as np
from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from collections import Counter
#测试样本数

def get_train_test_data(normal_path,abnormal_path,label_path):
    # 读取用户命令文件
    def read_user_commands(file_path):
        with open(file_path, "r") as f:
            return f.read().splitlines()

    # 获取命令块（每100行一个块）
    def get_command_blocks(commands):
        return [commands[i:i + 100] for i in range(0, len(commands), 100)]

    # 读取标签文件
    def read_labels(file_path):
        return np.loadtxt(file_path, dtype=int)

    # 读取用户3和用户10的命令文件
    normal_commands = read_user_commands(normal_path)
    abnormal_commands = read_user_commands(abnormal_path)

    # 分块
    normal_blocks = get_command_blocks(normal_commands)
    abnormal_blocks = get_command_blocks(abnormal_commands)

    # 读取标签矩阵
    labels_matrix = read_labels(label_path)

    # 找到用户3的异常块索引
    normal_labels = labels_matrix[:, 2]  # 第3个用户的标签在第3列
    abnormal_indices = [i + 50 for i, label in enumerate(normal_labels, start=0) if label == 1]

    # 替换用户3的异常块为用户10的相应块
    modified_normal_blocks = normal_blocks[:50] + [
        abnormal_blocks[i] if i in abnormal_indices else normal_blocks[i] for i in range(50, 150)
    ]

    # 构建训练集
    train_data = normal_blocks[:50] + abnormal_blocks[:50]
    train_labels = [0] * 50 + [1] * 50

    # 构建测试集（替换后的用户3的后100块）
    test_data = modified_normal_blocks[50:]
    test_labels = normal_labels.tolist()
    # # 输出结果
    # print("训练集大小:", len(train_data))
    # print("训练标签:", train_labels)
    # print("测试集大小:", len(test_data))
    # print("测试标签:", test_labels)

    return train_data,train_labels,test_data,test_labels
def get_larget_least_command_frequence(train_data):

    # 将前50个命令块展开为一个列表
    first_50_blocks = [command for block in train_data[:50] for command in block]

    # 计算前50个命令块中每个命令的频率
    first_50_freq = Counter(first_50_blocks)

    # 获取前50个命令块中最常用和最不常用的50个命令
    most_common_50_first = set([cmd for cmd, _ in first_50_freq.most_common(50)])
    least_common_50_first = set([cmd for cmd, _ in first_50_freq.most_common()[-50:]])

    # 将后50个命令块展开为一个列表
    last_50_blocks = [command for block in train_data[50:] for command in block]

    # 计算后50个命令块中每个命令的频率
    last_50_freq = Counter(last_50_blocks)

    # 获取后50个命令块中最常用和最不常用的50个命令
    most_common_50_last = set([cmd for cmd, _ in last_50_freq.most_common(50)])
    least_common_50_last = set([cmd for cmd, _ in last_50_freq.most_common()[-50:]])

    return most_common_50_first,least_common_50_first,most_common_50_last,least_common_50_last

# 此函数的作用是得到各个命令块的特征，输入是
def get_train_commands_features(train_cmd_list,most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal):
    freatures=[]
    i=0
    for cmd_block in train_cmd_list:
        feature1=len(set(cmd_block))
        command_frequence = list(FreqDist(cmd_block).keys())
        feature2=command_frequence[0:10]
        feature3=command_frequence[-10:]
        if i < 50:
            feature2 = len(set(feature2) & set(most_common_50_normal))
            feature3 = len(set(feature3) & set(least_common_50_normal))
        else:
            feature2 = len(set(feature2) & set(most_common_50_abnormal))
            feature3 = len(set(feature3) & set(least_common_50_abnormal))
        x = [feature1, feature2, feature3]
        freatures.append(x)
        i+=1
    return freatures
def get_test_commands_features(test_cmd_list,most_common_50_normal, least_common_50_normal):
    freatures = []
    for cmd_block in test_cmd_list:
        feature1 = len(set(cmd_block))
        command_frequence = list(FreqDist(cmd_block).keys())
        feature2 = command_frequence[0:10]
        feature3 = command_frequence[-10:]
        feature2 = len(set(feature2) & set(most_common_50_normal))
        feature3 = len(set(feature3) & set(least_common_50_normal))
        x = [feature1, feature2, feature3]
        freatures.append(x)
    return freatures
if __name__ == '__main__':
    # 文件路径
    normal_file = "../data/masquerade-data/User3"
    abnormal_file = "../data/masquerade-data/User1"
    label_file = "../data/masquerade-data/label.txt"

    # 调用上面的三个函数得到训练数据和测试数据
    train_data, train_labels, test_data, test_labels = get_train_test_data(normal_file,abnormal_file,label_file)
    most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal = get_larget_least_command_frequence(train_data)
    x_train = get_train_commands_features(train_data,most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal)
    x_test = get_test_commands_features(test_data,most_common_50_normal, least_common_50_normal)
    y_train = train_labels
    y_test = test_labels

    # 训练模型
    neigh = KNeighborsClassifier(n_neighbors=10,p=2,algorithm='auto',weights='uniform')
    neigh.fit(x_train, y_train)

    # 评估模型
    y_predict=neigh.predict(x_test)
    score=np.mean(y_test==y_predict)*100
    print(classification_report(y_test, y_predict))


