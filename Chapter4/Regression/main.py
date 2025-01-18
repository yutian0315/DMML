from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用其他中文字体
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载加州房价数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练岭回归模型
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# 进行房价预测
y_train_pred = ridge_reg.predict(X_train)
y_test_pred = ridge_reg.predict(X_test)

# 计算均方误差和R^2系数
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training set MSE: {train_mse:.3f}")
print(f"Testing set MSE: {test_mse:.3f}")
print(f"Training set R^2: {train_r2:.3f}")
print(f"Testing set R^2: {test_r2:.3f}")

# Plot the results
plt.figure(figsize=(10, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("训练集中的真实值与预测值")
plt.text(0.05, 0.95, f"MSE: {train_mse:.3f}\nR^2: {train_r2:.3f}", transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("测试集中的真实值与预测值")
plt.text(0.05, 0.95, f"MSE: {test_mse:.3f}\nR^2: {test_r2:.3f}", transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
