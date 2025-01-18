import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加常数项
X_b = np.c_[np.ones((100, 1)), X]

# 最小二乘法求解回归系数
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 预测值
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

# 绘制数据和拟合直线
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='Fitted line', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression using Least Squares Method")
plt.show()
