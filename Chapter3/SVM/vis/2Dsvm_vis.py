import numpy as np
from sklearn import svm

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

plt.rcParams['axes.unicode_minus'] = False  # 使负号显示正常
# 生成两类数据
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# 拟合SVM模型
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, Y)

# 获取超平面的参数
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 计算间隔边界
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# 绘制散点图、超平面和间隔边界
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.plot(xx, yy, 'k-', label='超平面')
plt.plot(xx, yy_down, 'k--', label='决策边界')
plt.plot(xx, yy_up, 'k--')

# 标记支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5)

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('$x_1$', fontweight='bold')
plt.ylabel('$x_2$', fontweight='bold')
plt.title('SVM 决策边界和间隔', fontweight='bold')

plt.legend()
plt.grid()

# 保存图像，设置 DPI
plt.savefig('svm_decision_boundary.png', dpi=1600)

# 显示图像
plt.show()
