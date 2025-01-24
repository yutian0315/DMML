import numpy as np
import pandas as pd
from kmeans import kmeans
from scipy.special import comb
from plotnine import ggplot, aes, geom_point, theme_bw, theme, labs, element_text, guides
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


# 计算归一化互信息（NMI）和调整兰德指数（ARI）
def evaluation(trueLabel, predLabel):
    # 检查输入标签长度是否一致
    if len(trueLabel) != len(predLabel):
        raise ValueError("trueLabel 和 predLabel 必须具有相同的长度")

    total = len(trueLabel)
    xIds = np.unique(trueLabel)  # 获取真实标签的唯一值
    yIds = np.unique(predLabel)  # 获取预测标签的唯一值

    # 计算互信息（MI）
    mutualInformation = 0.0
    for idx in xIds:
        for idy in yIds:
            # 获取真实标签和预测标签中分别等于 idx 和 idy 的索引
            idxOccur = np.where(trueLabel == idx)[0]
            idyOccur = np.where(predLabel == idy)[0]
            idxyOccur = np.intersect1d(idxOccur, idyOccur)  # 找到交集
            if len(idxyOccur) > 0:
                # 计算互信息公式
                mutualInformation += (len(idxyOccur) / total) * np.log2((len(idxyOccur) * total) /
                                                                      (len(idxOccur) * len(idyOccur)))

    # 计算真实标签的熵 Hx
    entropyHx = 0
    for idx in xIds:
        idxOccurCount = np.sum(trueLabel == idx)
        entropyHx -= (idxOccurCount / total) * np.log2(idxOccurCount / total)

    # 计算预测标签的熵 Hy
    entropyHy = 0
    for idy in yIds:
        idyOccurCount = np.sum(predLabel == idy)
        entropyHy -= (idyOccurCount / total) * np.log2(idyOccurCount / total)

    # 计算归一化互信息（NMI）
    normalizedMutualInformation = 2 * mutualInformation / (entropyHx + entropyHy) if (entropyHx + entropyHy) != 0 else 0

    # 计算调整兰德指数（ARI）
    contingencyTable = np.zeros((len(xIds), len(yIds)))
    for i, idx in enumerate(xIds):
        for j, idy in enumerate(yIds):
            # 填充列联表，表示真实标签和预测标签的每一对组合出现的次数
            contingencyTable[i, j] = np.sum((trueLabel == idx) & (predLabel == idy))

    n = np.sum(contingencyTable)  # 样本总数
    ni = np.sum(contingencyTable, axis=1)  # 每个真实标签的出现次数
    nj = np.sum(contingencyTable, axis=0)  # 每个预测标签的出现次数
    n2 = comb(n, 2)  # 总样本对数
    nis2 = np.sum([comb(x, 2) for x in ni if x > 1])  # 每个真实标签的组合对数
    njs2 = np.sum([comb(x, 2) for x in nj if x > 1])  # 每个预测标签的组合对数

    # 计算ARI公式
    adjustedRandIndex = (np.sum([comb(x, 2) for x in contingencyTable.flatten() if x > 1]) -
                         (nis2 * njs2) / n2) / ((nis2 + njs2) / 2 - (nis2 * njs2) / n2)
    # 返回结果字典
    return {"NMI": normalizedMutualInformation, "ARI": adjustedRandIndex}


# 加载并预处理数据
def loadData(filePath):
    # 加载数据集并排除第一列
    curData = pd.read_csv(filePath).iloc[:, 1:]
    # 加载第一列作为基因名称
    geneNames = pd.read_csv(filePath).iloc[:, 0]
    # 将基因名称中的 "|" 替换为 "-"，并移除 "_" 字符
    geneNames = geneNames.str.replace("|", "-", regex=True)
    geneNames = geneNames.str.replace("_", "", regex=True)
    # 将基因名称设置为行索引
    curData.index = geneNames
    return curData


# scRNA-seq 数据处理
def rnaseqProcessing(data, log2Transformation=True):
    # 过滤掉零值超过94%的行
    validRows = data.index[data.apply(lambda x: np.sum(x == 0) < 0.94 * data.shape[1], axis=1)]
    data = data.loc[validRows]
    # 过滤掉零值超过6%的行
    validRows = data.index[data.apply(lambda x: np.sum(x == 0) > 0.06 * data.shape[1], axis=1)]
    data = data.loc[validRows]
    # 如果指定，进行 log2 转换
    if log2Transformation:
        data = np.log2(data + 1)
    return data


# 测试
allData = ["Biase", "Chu", "Chung", 'Goolam', "Grover"]

# 为KMeans聚类的NMI和ARI结果创建dataframe
resmat_Kmeans_NMI = pd.DataFrame(index=allData, columns=["NMI"])
resmat_Kmeans_ARI = pd.DataFrame(index=allData, columns=["ARI"])


for curData in allData:  # 循环测试
    print(f"处理 {curData} 中...")

    # 加载数据集
    dataSet = pd.read_csv(f"./dataset/{curData}.csv", index_col=0)
    label = pd.read_csv(f"./dataset/{curData}_label.csv")["label"]  # 加载标签
    K = len(np.unique(label))  # 聚类的数量
    dataNew = dataSet.dropna()  # 排除空数据
    print("Kmeans 聚类...")
    NMI = []
    ARI = []
    for r in range(10):  # 循环10次
        kmeansClusterResult, _ = kmeans(dataNew.values.T, K)
        NA = evaluation(label, kmeansClusterResult)
        NMI.append(round(NA['NMI'], 3))
        ARI.append(round(NA['ARI'], 3))
    # 存储平均值
    resmat_Kmeans_NMI.loc[curData, 'NMI'] = round(np.mean(NMI), 3)
    resmat_Kmeans_ARI.loc[curData, 'ARI'] = round(np.mean(ARI), 3)

    print(f"{curData} 数据集处理完成！")


# 将结果保存至文件中
resmat_Kmeans_NMI.to_csv("C:/Users/20693/Desktop/result_for_only_clustering/Kmeans_NMI.csv", index=False)
resmat_Kmeans_ARI.to_csv("C:/Users/20693/Desktop/result_for_only_clustering/Kmeans_ARI.csv", index=False)


# 加载数据集
dataSet = pd.read_csv(f"./dataset/Chu.csv", index_col=0)
label = pd.read_csv(f"./dataset/Chu_label.csv")["label"]  # 加载标签
K = len(np.unique(label))  # 聚类的数量
dataNew = dataSet.dropna()  # 排除空数据
print("Kmeans 聚类...")
kmeansClusterResult, _ = kmeans(dataNew.values.T, K)

# 执行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
tsne_res = tsne.fit_transform(dataNew.T)  # 数据转置

# 将 t-SNE 结果转换为 DataFrame
df = pd.DataFrame(tsne_res, columns=["tSNE 1", "tSNE 2"])  # 使用标准列名
df['Label'] = label  # 添加真实标签列
df['KMeans'] = kmeansClusterResult  # 添加 KMeans 聚类标签列

# 使用 plotnine 绘制真实标签图并保存，移除图例
p_true_label = (ggplot(df, aes(x='tSNE 1', y='tSNE 2', color='factor(Label)'))
                + geom_point(size=1, alpha=1)
                + theme_bw()
                + theme(plot_title=element_text(hjust=0.5, size=15))
                + labs(title='真实标签', x='第一主成分', y='第二主成分', color=None)  # 设置中文坐标轴标签
                + guides(color='none'))  # 显式地移除图例
p_true_label.save("true_labels.png", dpi=100, width=8, height=6)

# 使用 plotnine 绘制 KMeans 聚类结果图并保存
p_kmeans = (ggplot(df, aes(x='tSNE 1', y='tSNE 2', color='factor(KMeans)'))
            + geom_point(size=1, alpha=1)
            + theme_bw()
            + theme(plot_title=element_text(hjust=0.5, size=15))
            + labs(title='KMeans 聚类结果', x='第一主成分', y='第二主成分', color='聚类')  # 设置中文坐标轴标签
            )
p_kmeans.save("kmeans_clusters.png", dpi=100, width=8, height=6)

# 创建包含两张图的合成图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 读取并显示保存的图片
img1 = plt.imread("true_labels.png")
img2 = plt.imread("kmeans_clusters.png")

axes[0].imshow(img1)
axes[0].axis('off')  # 隐藏坐标轴

axes[1].imshow(img2)
axes[1].axis('off')

# 显示合成图
plt.tight_layout()
plt.show()