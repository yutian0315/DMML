import numpy as np

def selectKPoints(dataSet, k):  # 在数据集dataset中随机选择k个点作为中心
    # 随机选择k个不重复的索引
    indexes = np.random.choice(len(dataSet), k, replace=False)
    return dataSet[indexes]

def closestCenterIndex(dataSet, centers):  # 找出距离每个数据点最近的中心的索引数组
    # 计算每个点到每个中心的欧式距离
    distances = np.linalg.norm(dataSet[:, np.newaxis] - centers, axis=2)
    return np.argmin(distances, axis=1)  # 找出每个点的对应索引

def updateCenters(dataSet, labels, k):  # 计算每个簇的中心
    centers = []
    for i in range(k):
        clusterPoints = dataSet[labels == i]  # 提取属于簇 i 的数据点
        center = clusterPoints.mean(axis=0)  # 计算该簇的中心
        centers.append(center)
    return np.array(centers)

# 将数据集dataSet划分为k个簇，迭代maxIters次，容忍值为tol
def kmeans(dataSet, k, maxIters=100, tol=1e-4):
    centers = selectKPoints(dataSet, k)  # 初始化k个中心
    for i in range(maxIters):
        labels = closestCenterIndex(dataSet, centers)  # 找出每个点的所属类别
        newCenters = updateCenters(dataSet, labels, k)  # 计算新的簇中心
        if np.all(np.abs(newCenters - centers) < tol):  # 算法收敛
            break
        centers = newCenters  # 更新簇
    return labels, centers

# 示例数据
dataSet = np.array([[1, 2, 5], [2, 2, 8], [3, 3, 10], [8, 7, 15], [8, 8, 16], [25, 80, 32]])

# 运行 K-means
# labels, centroids = kmeans(dataSet, k=2)
# print("Cluster labels:", labels)
# print("Centroids:", centroids)
