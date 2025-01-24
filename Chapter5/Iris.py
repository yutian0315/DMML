# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import pandas as pd
# import numpy as np
#
# # 加载鹫尾花数据集
# iris = load_iris()
# iris_data = iris.data
# iris_features = iris.feature_names
# iris_target = iris.target
# iris_target_names = iris.target_names
#
# # 将数据转换为DataFrame，便于处理
# iris_df = pd.DataFrame(iris_data, columns=iris_features)
# iris_df['species'] = pd.Categorical.from_codes(iris_target, iris_target_names)
#
# # 创建一个图形对象，设置图的大小
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#
# # 绘制每一对特征之间的散点图
# axes = axes.flatten()  # 将二维数组展平，方便使用
#
# # 选择每个特征的组合进行绘制
# feature_combinations = [
#     (0, 1),  # Sepal length vs Sepal width
#     (0, 2),  # Sepal length vs Petal length
#     (0, 3),  # Sepal length vs Petal width
#     (1, 2),  # Sepal width vs Petal length
# ]
#
# # 为每个子图绘制数据
# for i, (x_idx, y_idx) in enumerate(feature_combinations):
#     axes[i].scatter(iris_data[:, x_idx], iris_data[:, y_idx], c=iris_target, cmap=plt.cm.Set1)
#     axes[i].set_xlabel(iris_features[x_idx])
#     axes[i].set_ylabel(iris_features[y_idx])
#     axes[i].set_title(f"{iris_features[x_idx]} vs {iris_features[y_idx]}")
#
# # 调整子图布局
# plt.tight_layout()
# plt.suptitle("Iris Dataset Feature Scatter Plots", y=1.02)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import matplotlib
# # 设置字体为中文字体，这里以SimHei为例
# matplotlib.rcParams['font.family'] = 'Microsoft Yahei'
# matplotlib.rcParams['font.weight'] = 'bold'
# matplotlib.rcParams['font.size'] = 10
# # 解决负号'-'显示为方框的问题
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# # 加载鹫尾花数据集
# iris = load_iris()
# X = iris.data  # 特征矩阵
# y = iris.target  # 类别标签
# target_names = iris.target_names  # 类别名称
#
# # 设置图形的大小和子图布局
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#
# # 绘制特征对之间的散点图
# for ax, (i, j) in zip(axs.flat, [(0, 1), (0, 2), (1, 2), (1, 3)]):  # 注意：我们没有绘制(2, 3)因为鹫尾花只有4个特征
#     # 使用不同颜色表示不同类别
#     for k, color in zip(range(3), ['red', 'green', 'blue']):
#         ix = np.where(y == k)
#         ax.scatter(X[ix, i], X[ix, j], color=color, label=target_names[k], alpha=0.6)
#
#     # 设置子图的标题和标签
#     ax.set_title(f'Feature {iris.feature_names[i]} vs Feature {iris.feature_names[j]}')
#     ax.set_xlabel(iris.feature_names[i])
#     ax.set_ylabel(iris.feature_names[j])
#     ax.legend(loc='upper left')
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图形
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import matplotlib.font_manager as fm
#
# # 加载鹫尾花数据集
# iris = load_iris()
# X = iris.data  # 特征矩阵
# y = iris.target  # 类别标签
# target_names = iris.target_names  # 类别名称
# feature_names = iris.feature_names  # 特征名称
#
# # 如果你的系统中没有默认支持中文的字体，你可以指定一个中文字体的路径
# # 例如，在Windows上，你可以使用C:\Windows\Fonts\simhei.ttf
# # 在Linux或Mac上，你需要找到相应的中文字体文件路径
# # 这里我们假设系统中已经安装了支持中文的字体，并且matplotlib可以访问它
# # 如果你不确定，可以使用matplotlib.font_manager.findSystemFonts()来查找系统中的字体
#
# # 设置matplotlib以支持中文显示
# # 指定默认字体（这里假设系统中有SimHei字体，如果没有，请替换为实际存在的中文字体名称或路径）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 如果你知道中文字体的具体路径，也可以这样设置：
# # my_font = fm.FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')  # Windows示例路径
# # 但是对于坐标轴标签等，我们通常不需要单独设置FontProperties，因为上面的rcParams已经全局设置了
#
# # 设置图形的大小和子图布局
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
# # 绘制特征对之间的散点图
# feature_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]  # 鹫尾花有4个特征，所以绘制这些组合
# for ax, (i, j) in zip(axs.flat, feature_pairs):
#     # 使用不同颜色表示不同类别
#     for k, color in zip(range(3), ['red', 'green', 'blue']):
#         ix = np.where(y == k)
#         ax.scatter(X[ix, i], X[ix, j], color=color, label=target_names[k], alpha=0.8)
#
#     # 设置子图的标题和标签（使用中文）
#     ax.set_title(f'特征 {feature_names[i]} 与 特征 {feature_names[j]} 的关系', fontsize=14)
#     ax.set_xlabel(feature_names[i], fontsize=12)
#     ax.set_ylabel(feature_names[j], fontsize=12)
#     ax.legend(loc='upper left', fontsize=10)
#
# # 调整子图之间的间距和布局
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # 稍微调整以避免标题重叠
#
# # 显示图形
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris

# 加载鹫尾花数据集
iris = load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 类别标签
target_names = iris.target_names  # 类别名称
feature_names = iris.feature_names  # 特征名称

# 设置matplotlib以支持中文显示
# 这里指定了SimHei字体，如果你的系统中没有这个字体，请替换为实际存在的中文字体名称或路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.size'] = 14

# 设置图形的大小和子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 定义一个颜色列表，用于区分不同的类别
colors = ['red', 'green', 'blue']

# 绘制特征对之间的散点图
for ax, (i, j) in zip(axs.flat, [(0, 1), (0, 2), (1, 2), (2, 3)]):
    # 遍历每个类别，并绘制对应的散点
    for k, color in zip(range(3), colors):
        ix = np.where(y == k)
        ax.scatter(X[ix, i], X[ix, j], color=color, label=target_names[k], alpha=0.8)

    # 设置子图的标题和坐标轴标签（使用中文）
    ax.set_title(f'特征 {feature_names[i]} 与 特征 {feature_names[j]} 的关系', fontsize=14)
    ax.set_xlabel(feature_names[i], fontsize=12)
    ax.set_ylabel(feature_names[j], fontsize=12)

    # 显示图例
    ax.legend(loc='upper left', fontsize=10)

# 调整子图之间的间距和布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 显示图形
plt.show()