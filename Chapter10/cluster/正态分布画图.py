import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# Matplotlib字体设置为微软雅黑、加粗、字体大小18号
plt.rcParams['font.family'] = 'Microsoft YaHei'
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建 x 值
x_norm = np.linspace(0, 8, 1000)  # 标准正态分布（右移到 4 处）
x_lognorm = np.linspace(0, 10, 1000)  # 对数正态分布

# 计算标准正态分布的 y 值，均值设为 4，标准差为 1
mean_norm = 4
std_norm = 1
y_norm = norm.pdf(x_norm, loc=mean_norm, scale=std_norm)

# 计算对数正态分布的 y 值
# 增加标准差来降低高度
s = 0.8  # 增大对数正态分布的标准差
scale = np.exp(0)  # 均值
y_lognorm = lognorm.pdf(x_lognorm, s, scale=scale)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号的显示

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x_norm, y_norm, label='标准正态分布（均值=4）', color='blue')  # 实线
plt.plot(x_lognorm, y_lognorm, label='对数正态分布（标准差=0.8）', color='red', linestyle='--', dashes=(10, 10))  # 虚线
#plt.title('标准正态分布与对数正态分布', fontsize=24)  # 标题字体大小
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)  # x 和 y 轴刻度标签字体大小
plt.grid()
plt.savefig(r'C:\Users\Jack\Documents\log2.png', format='png', dpi=300, bbox_inches='tight')

plt.show()