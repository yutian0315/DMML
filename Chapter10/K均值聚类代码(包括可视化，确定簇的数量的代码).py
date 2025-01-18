import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
# 设置字体为中文字体，这里以SimHei为例
matplotlib.rcParams['font.family'] = 'Microsoft Yahei'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = 14
# 解决负号'-'显示为方框的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 实际标签
# 2. 数据预处理 - 标准化
scaler = StandardScaler() # 创建一个 StandardScaler 对象，用于对数据进行标准化
# 将数据X进行标准化的过程
# fit方法会计算数据X的统计信息(如均值和标准差)，以确定如何转换数据。
# transform方法根据这些统计信息，将原数据缩放到指定的范围或分布上。
X_scaled = scaler.fit_transform(X)

class kmeans:
    def __init__(self, nClusters=3, init='random', maxIter=500, tol=1e-4, randomState=None):
        """
        参数:
            - n_clusters: 聚类的簇数
            - init: 质心初始化方法（这里只支持 'random'）
            - max_iter: 最大迭代次数
            - tol: 收敛判断的容忍度
            - random_state: 随机数种子
        """
        self.n_clusters = nClusters
        self.init = init
        self.max_iter = maxIter
        self.tol = tol
        self.random_state = randomState
        self.centroids = None
    def fit(self, X):
        # 设置随机种子，使得每次运行时结果一致，便于实验可复现性。
        if self.random_state:
            np.random.seed(self.random_state)
        # 初始化质心，若 init 参数设置为 'random'，则从数据 X 中随机选择 n_clusters 个样本作为初始质心。
        if self.init == 'random':
            # 随机选择 n_clusters 个数据点的索引。
            random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
            # 将这些数据点用作初始质心。
            self.centroids = X[random_indices]
        for _ in range(self.max_iter):
            # Step 1: 计算每个样本到质心的距离，并分配到最近的簇
            # X[:, np.newaxis] - self.centroids：为每个数据点与每个质心之间计算距离差值。
            # np.linalg.norm(..., axis=2)：对这些差值计算欧氏距离，得到一个距离矩阵distances，其中每
            # 行表示每个样本点到所有质心的距离。
            # labels = np.argmin(distances, axis=1)：选择每行最小值的索引作为簇标签，即每个样本被分
            # 配到最近的簇。
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            # Step 2: 计算新的质心
            # 对每个簇(k 表示簇编号)计算属于该簇的数据点的平均值。
            # X[labels == k] 选择所有标签为 k 的数据点，mean(axis=0) 计算这些点在各特征维度上的平均值，作为新的质心位置。
            # new_centroids 是所有簇的新质心。
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Step 3: 检查收敛（如果质心变化小于 tol）
            # 计算每个质心与上一轮质心位置的差距。若所有质心位置的变化均小于tol，即np.all() 返回True，
            # 则认为已收敛，终止循环。
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            # 更新质心
            self.centroids = new_centroids
        self.labels_ = labels
        self.cluster_centers_ = self.centroids
    def fitPredict(self, X):
        # 执行聚类并返回每个样本的簇标签
        self.fit(X)
        return self.labels_
# 3. 确定簇的数量 - 使用肘部法，肘部法用于确定最优簇数K
def elbowMethod(X, max_k=10):
    distortions = [] # 初始化一个空列表 distortions，用于存储每个 k 值对应的失真度。
    for k in range(1, max_k + 1):
        # 创建一个 K-Means 模型，簇数为 k，并设置随机种子为 42。
        km = kmeans(nClusters=k, randomState=42)
        km.fit(X) # 进行拟合
        # 计算失真度（Distortion），即每个数据点到其分配的簇中心的距离平方和的平均值。
        distortion = np.mean([np.linalg.norm(x - km.cluster_centers_[km.labels_[i]])**2 for i, x in enumerate(X)])
        # 将每个 k 值对应的失真度添加到 distortions 列表。
        distortions.append(distortion)
    return distortions
# 轮廓系数法确定最佳K值
def silhouetteMethod(X, max_k=10):
    silhouette_scores = [] # 初始化一个空列表 silhouette_scores，用于存储每个 k 值对应的轮廓系数。
    for k in range(2, max_k + 1):
        # 创建一个 K-Means 模型，簇数为 k，并设置随机种子为 42。
        km = kmeans(nClusters=k, randomState=42)
        # 将数据 X 聚类并返回每个样本的簇标签 labels。
        labels = km.fitPredict(X)
        # 计算轮廓系数，计算当前 k 值下的轮廓系数 score，用于评估聚类质量。
        # 轮廓系数衡量聚类效果，取值范围为 -1 到 1。值越高，表示聚类效果越好，聚类间的分离度和一致性越高。
        score = silhouette_score(X, labels)
        # 将当前 k 值对应的轮廓系数添加到 silhouette_scores 列表。
        silhouette_scores.append(score)
    return silhouette_scores
# 4. 绘制肘部图和轮廓系数图
def plot_elbow_silhouette():
    # 应用肘部法
    distortions = elbowMethod(X_scaled, max_k=10)
    # 应用轮廓系数法
    silhouette_scores = silhouetteMethod(X_scaled, max_k=10)
    # 绘制肘部法图像
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('簇的数量 (K)')
    plt.ylabel('失真度')
    # plt.title('求解最优K值的肘部法')
    # 绘制轮廓系数法图像
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('簇的数量 (K)')
    plt.ylabel('轮廓系数')
    # plt.title('确定最优K的轮廓法')
    plt.show()
plot_elbow_silhouette()
# 5. 运行 K-Means 算法
km = kmeans(nClusters=3, randomState=42)
labels = km.fitPredict(X_scaled)
# 使用 PCA 将数据降维到二维，便于可视化
# 创建一个 PCA 对象，n_components=2 表示我们希望将数据降到 2 个主成分（也就是二维空间）。
pca = PCA(n_components=2)
# PCA是一种无监督的降维方法，通过寻找数据中方差最大的方向（主成分）来降低数据维度，从而保留数据的主要特征。
# fit部分：PCA 根据 X_scaled 数据计算主成分方向（即方差最大的方向）。
# transform部分：将原始数据 X_scaled 投影到这些主成分上，从而生成降维后的数据。
X_pca = pca.fit_transform(X_scaled)
# 6. 可视化聚类结果（使用前两个特征）
def plotPca(labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    # 将K-means聚类的中心点（每个簇的质心）传递给PCA模型，然后通过PCA将这些高维的聚类中心数据降维，
    # 映射到PCA所定义的低维空间中。
    centroidsPca = pca.transform(km.cluster_centers_)
    plt.scatter(centroidsPca[:, 0], centroidsPca[:, 1], s=200, c='red', marker='X', label='质心')
    # plt.title("Iris数据集的K-means聚类")
    plt.xlabel("萼片长度 (cm)")
    plt.ylabel("萼片宽度 (cm)")
    plt.legend()
    plt.show()
# plotPca(labels)



