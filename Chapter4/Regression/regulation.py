import numpy as np
import matplotlib.pyplot as plt

# 创建网格数据
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# L1 正则化 (曼哈顿距离)
L1 = np.abs(X) + np.abs(Y)

# L2 正则化 (欧几里得距离)
L2 = X**2 + Y**2

# 绘图
plt.figure(figsize=(12, 6))

# L1 正则化
plt.subplot(1, 2, 1)
plt.contour(X, Y, L1, levels=np.linspace(0, 6, 7), colors='blue', alpha=0.5)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title('L1 Regularization (Lasso)')
plt.xlabel('x1')
plt.ylabel('x2')

# 在坐标轴上标注 L1 正则化的约束条件
for level in np.linspace(1, 5, 5):
    plt.plot([level, -level], [0, 0], color='blue', lw=2)
    plt.plot([0, 0], [level, -level], color='blue', lw=2)

plt.grid()

# L2 正则化
plt.subplot(1, 2, 2)
plt.contour(X, Y, L2, levels=np.linspace(0, 20, 7), colors='red', alpha=0.5)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title('L2 Regularization (Ridge)')
plt.xlabel('x1')
plt.ylabel('x2')

# 在坐标轴上标注 L2 正则化的约束条件
for level in np.linspace(1, 5, 5):
    plt.plot([level, -level], [np.sqrt(level), -np.sqrt(level)], color='red', lw=2)
    plt.plot([np.sqrt(level), -np.sqrt(level)], [level, -level], color='red', lw=2)

plt.grid()

plt.tight_layout()
plt.show()
