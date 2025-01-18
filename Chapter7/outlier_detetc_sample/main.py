# -*- coding:utf-8 -*-

from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import numpy as np
from collections import Counter


# 定义计算欧式距离的函数
def euclidean_distance(point1, point2):
    point1, point2 = np.array(point1), np.array(point2)  # 转换为NumPy数组
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 定义KNN算法
class KNN:
    def __init__(self, k=3):
        self.k = k  # K值：最近邻居的数量

    # 训练函数：仅保存训练数据和标签
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 单点预测函数
    def _predict_one(self, x):
        # 计算目标点到所有训练数据点的距离
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # 根据距离排序，并获取距离最近的K个点的标签
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 投票选出最常见的类别
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # 批量预测函数
    def predict(self, X_test):
        return [self._predict_one(x) for x in X_test]
def getTrainTestData(normal_path, abnormal_path, label_path):
    # 读取用户命令文件
    def readUserCommands(file_path):
        with open(file_path, "r") as f:
            return f.read().splitlines()

    # 获取命令块（每100行一个块）
    def getCommandBlocks(commands):
        return [commands[i:i + 100] for i in range(0, len(commands), 100)]

    # 读取标签文件
    def readLabels(file_path):
        return np.loadtxt(file_path, dtype=int)

    # 读取正常用户和异常用户的命令文件
    normal_commands = readUserCommands(normal_path)
    abnormal_commands = readUserCommands(abnormal_path)

    # 分块
    normal_blocks = getCommandBlocks(normal_commands)
    abnormal_blocks = getCommandBlocks(abnormal_commands)

    # 读取标签矩阵
    labels_matrix = readLabels(label_path)

    # 找到正常用户的异常块索引
    normal_labels = labels_matrix[:, 2]  # 第3个用户的标签在第3列
    abnormal_indices = [i + 50 for i, label in enumerate(normal_labels, start=0) if label == 1]

    # 替换正常用户的异常块为异常用户的相应块
    modified_normal_blocks = normal_blocks[:50] + [
        abnormal_blocks[i] if i in abnormal_indices else normal_blocks[i] for i in range(50, 150)
    ]

    # 构建训练集
    train_data = normal_blocks[:50] + abnormal_blocks[:50]
    train_labels = [0] * 50 + [1] * 50

    # 构建测试集（替换后的正常用户的后100块）
    test_data = modified_normal_blocks[50:]
    test_labels = normal_labels.tolist()
    # # 输出结果
    # print("训练集大小:", len(train_data))
    # print("训练标签:", train_labels)
    # print("测试集大小:", len(test_data))
    # print("测试标签:", test_labels)

    return train_data,train_labels,test_data,test_labels
def getLargetLeastCommandFrequence(train_data):

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

def getTrainCommandsFeatures(train_cmd_list, most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal):
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
def getTestCommandsFeatures(test_cmd_list, most_common_50_normal, least_common_50_normal):
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
    train_data, train_labels, test_data, test_labels = getTrainTestData(normal_file, abnormal_file, label_file)
    most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal = getLargetLeastCommandFrequence(train_data)
    x_train = getTrainCommandsFeatures(train_data, most_common_50_normal, least_common_50_normal, most_common_50_abnormal, least_common_50_abnormal)
    x_test = getTestCommandsFeatures(test_data, most_common_50_normal, least_common_50_normal)
    y_train = train_labels
    y_test = test_labels

    # 训练模型
    # neigh = KNeighborsClassifier(n_neighbors=10,p=2,algorithm='auto',weights='uniform')
    neigh = KNN(k=3)
    neigh.fit(x_train, y_train)

    # 评估模型
    y_predict=neigh.predict(x_test)
    score=np.mean(y_test==y_predict)*100
    print(classification_report(y_test, y_predict))
