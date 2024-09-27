import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

# 生成两类的3D数据集
np.random.seed(0)
class1 = np.random.randn(10, 3) + np.array([1, 1, 1])
class2 = np.random.randn(10, 3) + np.array([5, 5, 5])

# 合并数据集
X = np.vstack((class1, class2))
y = np.array([0]*10 + [1]*10)

# 训练SVM分类器
svm = SVC(kernel='linear')
svm.fit(X, y)

# 获取支持向量
support_vectors = svm.support_vectors_

# 绘制3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制class1和class2的散点图
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], color='b', label='Class 1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], color='r', label='Class 2')

# 突出显示支持向量
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], support_vectors[:, 2], 
           s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# 创建超平面的网格
xx, yy = np.meshgrid(np.linspace(-3, 9, 10), np.linspace(-3, 9, 10))

# 超平面的方程
zz = (-svm.coef_[0][0] * xx - svm.coef_[0][1] * yy - svm.intercept_[0]) / svm.coef_[0][2]

# 绘制超平面
ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)

# 标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Hyperplane and Support Vectors')
ax.legend()

# 显示图形
plt.show()
