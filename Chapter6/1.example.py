import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from matplotlib import cm


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Microsft YaHei'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12


def getInfo(filePath):
    # 读取数据
    data = pd.read_excel(filePath)
    # 查看数据集维度
    print(data.shape)  # (541909, 8)
    # 查看数据集中各列数据格式
    print(data.dtypes)
    # 查看各列数据分布情况
    print(data.info())
    print(data.describe())


def preprocessing(filePath):
    # 读取数据
    data = pd.read_excel(filePath)

    # 去重
    data.drop_duplicates(inplace=True)
    print("去重后数据基本信息：")
    #查看数据集维度
    print(data.shape)   # (536641, 8)
    #查看数据集中各列数据格式
    print(data.dtypes)
    #查看各列数据分布情况
    print(data.info())
    print(data.describe())

    # 异常值处理
    data = data.loc[(data["Quantity"]>0) & (data["UnitPrice"]>=0)]
    print(data.describe())

    # 缺失值处理
    msno.matrix(data)   # 矩阵图z
    plt.show()
    print(data.info())

    # 过滤掉 CustomerID 为空的行
    data = data[data["CustomerID"].notna()]
    print(data.info())

    data.to_excel("../../data/online_retail_cleaned.xlsx", index=False)

    return data


# 统计不同国家销售量
def status(data):
    # 对数据集按照"Country"字段值进行分组
    groupByCountry = data.groupby("Country")
    # 对各组中数据的"Quantity"字段进行求和
    countryQuantityList = list(dict(groupByCountry["Quantity"].sum()).items())
    # 对数据集按照产品销售总量进行排序
    countryQuantityList.sort(key=lambda x: x[1])
    print("countryQuantityList：")
    print(countryQuantityList)

    # country_quantity_list.pop()  #在可视化时去除/保留销量第一的国家销量数据
    norm = plt.Normalize(0, countryQuantityList[-1][1])
    normValues = norm([i[1] for i in countryQuantityList])
    mapVir = cm.get_cmap(name='jet')
    colors = mapVir(normValues)
    plt.barh([i[0] for i in countryQuantityList], [i[1] for i in countryQuantityList], height = 0.4,color=colors)
    plt.tick_params(labelsize=8)
    sm = cm.ScalarMappable(cmap=mapVir, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    # 给条形图添加数据标注
    for index, yValue in enumerate([i[1] for i in countryQuantityList]):
        plt.text(yValue + 12, index-0.2, "%s" %yValue)
    plt.ylabel("Country", fontdict={'size':18})
    plt.title("不同国家产品总销售量")
    plt.show()


if __name__ == '__main__':
    dataPath = "../../data/online_retail.xlsx"
    # getInfo(dataPath)
    data = preprocessing(dataPath)
    # status(data)