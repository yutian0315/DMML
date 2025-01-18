import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(22)

# Generate data for three clusters
cluster1 = np.random.normal(loc=(3, 6), scale=1, size=(20, 2))
cluster2 = np.random.normal(loc=(4, 3), scale=1, size=(20, 2))
cluster3 = np.random.normal(loc=(6, 5), scale=1, size=(20, 2))

# Define the target point and circle radii
target_point = (5, 4)

# 所有数据点和中心点
all_points = np.vstack([cluster1, cluster2, cluster3])
center_point = np.array(target_point)
# 计算到目标点的距离并找到5个最近邻居
distances = np.linalg.norm(all_points - center_point, axis=1)
nearest_indices = np.argsort(distances)[:5]
nearest_neighbors = all_points[nearest_indices]

# 找到第5个和第10个最近邻居的距离
sorted_distances = np.sort(distances)
radius1 = sorted_distances[4]  # 第5个最近邻居
radius2 = sorted_distances[9]  # 第10个最近邻居

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 左侧子图：绘制同心圆
axes[1].scatter(cluster1[:, 0], cluster1[:, 1], color='yellow', edgecolor='black', marker='o', label='类别 1')
axes[1].scatter(cluster2[:, 0], cluster2[:, 1], color='red', edgecolor='black', marker='^', label='类别 2')
axes[1].scatter(cluster3[:, 0], cluster3[:, 1], color='green', edgecolor='black', marker='s', label='类别 3')
axes[1].scatter(*target_point, color='black', marker='x', s=100, label='目标点')
circle1 = plt.Circle(target_point, radius1, color='black', fill=False, linestyle='--', linewidth=1)
circle2 = plt.Circle(target_point, radius2, color='black', fill=False, linestyle='--', linewidth=1)
axes[1].add_patch(circle1)
axes[1].add_patch(circle2)
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].set_xlabel('X', fontsize=18)
axes[1].set_ylabel('Y', fontsize=18)
# axes[0].set_title("同心圆表示不同k值的影响", fontsize=20)
axes[1].grid(True)
axes[1].legend(loc='upper left', fontsize=18)

# 右侧子图：绘制箭头
axes[0].scatter(cluster1[:, 0], cluster1[:, 1], color='yellow', edgecolor='black', marker='o', label='类别 1')
axes[0].scatter(cluster2[:, 0], cluster2[:, 1], color='red', edgecolor='black', marker='^', label='类别 2')
axes[0].scatter(cluster3[:, 0], cluster3[:, 1], color='green', edgecolor='black', marker='s', label='类别 3')
axes[0].scatter(*target_point, color='black', marker='x', s=100, label='目标点')
for neighbor in nearest_neighbors:
    axes[0].arrow(center_point[0], center_point[1], neighbor[0] - center_point[0], neighbor[1] - center_point[1],
                  head_width=0.1, head_length=0.1, fc='black', ec='black')
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, 10)
axes[0].set_xlabel('X', fontsize=18)
axes[0].set_ylabel('Y', fontsize=18)
# axes[1].set_title("箭头指示最近5个邻居", fontsize=20)
axes[0].grid(True)
axes[0].legend(loc='upper left', fontsize=18)

# 调整布局并保存
plt.tight_layout()
plt.savefig('寻找K个最近邻居.pdf', format='pdf')
plt.show()
