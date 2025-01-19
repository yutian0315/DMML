import matplotlib.pyplot as plt
import numpy as np

# 设置绘图字体为中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 读取 AUC 数据
auc_data = {}
with open('AUC.txt', 'r', encoding='utf-8') as auc_file:
    lines = auc_file.readlines()[1:]  # 跳过文件头
    for line in lines:
        feature, auc_change = line.strip().split('\t')
        auc_data[feature] = float(auc_change)

# 读取 AUPRC 数据
auprc_data = {}
with open('AUPRC.txt', 'r', encoding='utf-8') as auprc_file:
    lines = auprc_file.readlines()[1:]  # 跳过文件头
    for line in lines:
        feature, auprc_change = line.strip().split('\t')
        auprc_data[feature] = float(auprc_change)

# 绘制 AUC 变化纵向柱状图并显示数值
plt.figure(figsize=(10, 6))
bars_auc = plt.bar(list(auc_data.keys()), list(auc_data.values()), color='#1f77b4', alpha=0.7, width=0.6)
plt.xlabel('特征', fontsize=14)
plt.ylabel('AUC 变化', fontsize=14)
plt.gca().invert_xaxis()

# 添加 y=0 虚线
plt.axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('AUC_impact.pdf', format='pdf')
plt.show()

# 绘制 AUPRC 变化纵向柱状图并显示数值
plt.figure(figsize=(10, 6))
bars_auprc = plt.bar(list(auprc_data.keys()), list(auprc_data.values()), color='#1f77b4', alpha=0.7, width=0.6)
plt.xlabel('特征', fontsize=14)
plt.ylabel('AUPRC 变化', fontsize=14)
plt.gca().invert_xaxis()

# 添加 y=0 虚线
plt.axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('AUPRC_impact.pdf', format='pdf')
plt.show()
