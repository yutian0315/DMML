import pandas as pd


# 根据路径加载数据集
def loadData(filePath, country="all"):
    data = pd.read_excel(filePath)

    if country != "all":
        data = data.loc[data["Country"] == country]
    data = data.loc[:, ["InvoiceNo", "StockCode"]]

    data["StockCode"] = data["StockCode"].apply(lambda x: "," + str(x))
    # 将数据按照发票编号进行分组
    data = data.groupby('InvoiceNo').sum().reset_index()
    # 将相同发票编号的产品代码放入同一个列表中
    data["StockCode"] = data["StockCode"].apply(lambda x: [x[1:]])
    stockCodeLists = list(data["StockCode"])

    # 将具有相同发票编号的商品代码字符串切分为列表
    for i in range(len(stockCodeLists)):
        stockCodeStr = stockCodeLists[i][0]
        stockCodeList = stockCodeStr.split(",")
        stockCodeLists[i] = stockCodeList

    return stockCodeLists


# 保存结果到 txt 文件
def saveRule(rule, path):
    with open(path, "w") as file:
        file.write("index  confidence   rules\n")
        index = 1
        for item in rule:
            line = " {:<4d}  {:.3f}        {}=>{}\n".format(index, item[2], str(list(item[0])), str(list(item[1])))
            index += 1
            file.write(line)
        file.close()
    print(f"Result saved, path is: {path}")


class Apriori:
    # 初始化函数，获取给定的数据集、支持度阈值、置信度阈值
    def __init__(self, data, supportThreshold, confidenceThreshold):
        self.data = data
        self.supportThreshold = supportThreshold
        self.confidenceThreshold = confidenceThreshold

    # 找到 1 项候选集 C1
    def createC1(self):
        c1 = []
        for row in self.data:
            for item in row:
                if [item] not in c1:
                    c1.append([item])
        c1.sort()
        return list(map(frozenset, c1))

    # 计算 1 项候选集的支持度，剔除小于最小支持度的项集
    def calculateSupport(self, dataset, candidates):
        supportCount = {}
        for transaction in dataset:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    supportCount[candidate] = supportCount.get(candidate, 0) + 1

        totalTransactions = float(len(dataset))
        frequentItemsets = []
        supportData = {}
        for item, count in supportCount.items():
            support = count / totalTransactions
            if support > self.supportThreshold:
                frequentItemsets.append(item)
                supportData[item] = support
        return frequentItemsets, supportData

    # 使用剪枝算法，生成 k 项候选集
    def generateCandidates(self, frequentItemsets, k):
        candidates = []
        numItems = len(frequentItemsets)
        for i in range(numItems):
            for j in range(i + 1, numItems):
                l1 = list(frequentItemsets[i])[:k - 2]
                l2 = list(frequentItemsets[j])[:k - 2]
                l1.sort()
                l2.sort()
                if l1 == l2:
                    candidate = frequentItemsets[i] | frequentItemsets[j]
                    if self.hasFrequentSubsets(candidate, frequentItemsets):
                        candidates.append(candidate)
        return candidates

    # 检查是否所有子集都是频繁项
    def hasFrequentSubsets(self, candidate, frequentItemsets):
        subsets = [frozenset(candidate - set([item])) for item in candidate]
        return all(subset in frequentItemsets for subset in subsets)

    # 查找频繁项集
    def getFrequentItemsets(self):
        c1 = self.createC1()
        dataset = list(map(set, self.data))
        l1, supportData = self.calculateSupport(dataset, c1)
        allFrequentItemsets = [l1]
        k = 2
        while len(allFrequentItemsets[k - 2]) > 0:
            ck = self.generateCandidates(allFrequentItemsets[k - 2], k)
            lk, supportK = self.calculateSupport(dataset, ck)
            supportData.update(supportK)
            allFrequentItemsets.append(lk)
            k += 1
        del allFrequentItemsets[-1]
        return allFrequentItemsets, supportData

    # 生成所有子集
    def getSubsets(self, frequentItemset, subsets):
        for i in range(len(frequentItemset)):
            subset = frozenset(set(frequentItemset) - set([frequentItemset[i]]))
            if subset not in subsets:
                subsets.append(subset)
                if len(subset) > 1:
                    self.getSubsets(list(subset), subsets)

    # 计算置信度，并剔除小于最小置信度的规则
    def calculateConfidence(self, frequentItemset, subsets, supportData, strongRules):
        for subset in subsets:
            confidence = supportData[frequentItemset] / supportData[frequentItemset - subset]
            lift = confidence / supportData[subset]
            if confidence >= self.confidenceThreshold and lift > 1:
                strongRules.append((frequentItemset - subset, subset, confidence))

    # 生成强关联规则
    def getRules(self, frequentItemsets, supportData):
        strongRules = []
        for k in range(1, len(frequentItemsets)):
            for frequentItemset in frequentItemsets[k]:
                subsets = []
                self.getSubsets(list(frequentItemset), subsets)
                self.calculateConfidence(frequentItemset, subsets, supportData, strongRules)
        return strongRules


if __name__ == "__main__":
    fileName = "online_retail.xlsx"
    filePath = "../../data/" + fileName
    savePath = "../../log/" + fileName.split(".")[0] + "_apriori.txt"

    stockCodeLists = loadData(filePath)
    apriori = Apriori(stockCodeLists, 0.01, 0.7)

    frequentItemsets, supportData = apriori.getFrequentItemsets()
    print("所有频繁项集：")
    for i, itemsets in enumerate(frequentItemsets):
        print(f"频繁 {i + 1} 项集：")
        print(f"个数：{len(itemsets)}")
        for itemset in itemsets:
            print(itemset)

    rules = apriori.getRules(frequentItemsets, supportData)
    print("符合要求的关联规则如下：")
    print(f"个数：{len(rules)}")
    saveRule(rules, savePath)
