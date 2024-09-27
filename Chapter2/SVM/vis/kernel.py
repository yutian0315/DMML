import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 生成环形数据
np.random.seed(0)

# 类别1：环形数据
theta = np.linspace(0, 2 * np.pi, 100)  # 增加到100个点
r_outer = 1.0
class1 = np.column_stack((r_outer * np.cos(theta), r_outer * np.sin(theta))) + np.random.randn(100, 2) * 0.05

# 类别2：中心点数据
class2 = np.random.randn(100, 2) * 0.1 + np.array([0, 0])  # 中心在 (0, 0)

# 合并数据集
X = np.vstack((class1, class2))
y = np.array([0]*100 + [1]*100)

# 训练SVM分类器（使用RBF核）
svm = SVC(kernel='rbf', C=1, gamma=1)
svm.fit(X, y)

# RBF核函数将数据映射到高维空间
def rbf_kernel(X, gamma=1):
    pairwise_sq_dists = np.square(X[:, np.newaxis] - X[np.newaxis, :]).sum(axis=2)
    return np.exp(-gamma * pairwise_sq_dists)

# 计算映射后的特征
K = rbf_kernel(X, gamma=1)

# 选择前两个主成分进行3D可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(K)

# 创建一个窗口包含两个子图
fig = plt.figure(figsize=(12, 6))

# 子图1: 原始二维数据的散点图
ax1 = fig.add_subplot(121)
ax1.scatter(class1[:, 0], class1[:, 1], color='b', label='Class 1 (Ring)')
ax1.scatter(class2[:, 0], class2[:, 1], color='r', label='Class 2 (Center)')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_title('Original 2D Data with Ring')
ax1.axis('equal')
ax1.legend()

# 子图2: 映射到高维空间后的三维散点图
ax2 = fig.add_subplot(122, projection='3d')
# 根据类别设置z值
z_values = np.where(y == 0, 1, 0)  # 类别1的z值为1，类别2的z值为0
# 增加透明度，使得重叠部分更清晰
ax2.scatter(X_pca[:100, 0], X_pca[:100, 1], z_values[:100], color='b', alpha=0.6, label='Class 1 (Ring)')
ax2.scatter(X_pca[100:, 0], X_pca[100:, 1], z_values[100:], color='r', alpha=0.6, label='Class 2 (Center)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_zlabel('Class')
ax2.set_title('3D Visualization of Data with RBF Kernel')
ax2.view_init(elev=20, azim=30)  # 调整视角
ax2.legend()

# 显示图形
plt.tight_layout()
plt.show()
