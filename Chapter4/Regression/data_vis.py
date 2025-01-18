import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用其他中文字体
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Load the California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

# Combine features and target variable
data = pd.concat([X, y], axis=1)



# Plot the correlation heatmap of features
plt.figure(figsize=(14, 10))
correlation_matrix = data.corr().round(2)

# 使用imshow来绘制热图，并将相关系数值显示在每个方块中
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)

# 在每个方块内显示相关系数值
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, correlation_matrix.iloc[i, j], ha='center', va='center', color='black', fontsize=12)

plt.title('特征相关性')
plt.tight_layout()
plt.show()

# 设置ggplot风格
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')

# Plot the distribution of the target variable (median house value)
plt.hist(data['MedHouseVal'], bins=30, edgecolor='black', alpha=0.7, color='lightblue')  # 更深的蓝色
plt.title('房价中位数分布')
plt.xlabel('中位数')
plt.ylabel('次数')
plt.show()


# Plot the relationship between features and target variable
plt.figure(figsize=(14, 10))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    plt.scatter(data[col], data['MedHouseVal'], alpha=0.7, color='royalblue')  # 更深的蓝色
    plt.xlabel(col)
    plt.ylabel('房价中位数')
    plt.title(f'{col} vs 房价中位数')

plt.tight_layout()
plt.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_lr))
print("R-squared (R²):", r2_score(y_test, y_pred_lr))

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print("\nRidge Regression")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_ridge))
print("R-squared (R²):", r2_score(y_test, y_pred_ridge))

# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Regression")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_dt))
print("R-squared (R²):", r2_score(y_test, y_pred_dt))
