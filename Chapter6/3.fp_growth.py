from tqdm import tqdm
from xlrd import open_workbook


def loadData(path):  # 根据路径加载数据集
    result = []  # 将数据保存到该数组
    if path.split(".")[-1] == "xls":  # 若路径为药方.xls
        workbook = open_workbook(path)
        sheet = workbook.sheet_by_index(0)  # 读取第一个 sheet
        for i in range(1, sheet.nrows):  # 忽视 header，从第二行开始读数据，第一列为处方 ID，第二列为药品清单
            temp = sheet.row_values(i)[1].split(";")[:-1]  # 取该行数据的第二列并以“;”分割为数组
            if len(temp) == 0:
                continue
            temp = [item.split(":")[0] for item in temp]  # 将药品后跟着的药品用量去掉
            temp = list(set(temp))  # 去重，排序
            temp.sort()
            result.append(temp)  # 将处理好的数据添加到数组
    return result


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


class Node:
    def __init__(self, nodeName, count, parentNode):
        self.name = nodeName
        self.count = count
        # 根据node_link可以找到整棵树中所有node_name一样的节点
        self.nodeLink = None
        # 父亲节点
        self.parent = parentNode
        # 子节点{节点名字:节点地址}
        self.children = {}


class FpGrowth():
    # 更新 headerTable 中的 node 节点形成的链表
    def updateHeader(self, node, targetNode):
        while node.nodeLink is not None:
            node = node.nodeLink
        node.nodeLink = targetNode

    # 更新 fpTree
    def updateFpTree(self, items, node, headerTable):
        if items[0] in node.children:
            # 判断 items 的第一个结点是否已作为子结点
            node.children[items[0]].count += 1
        else:
            # 创建新的分支
            node.children[items[0]] = Node(items[0], 1, node)
            # 更新相应频繁项集的链表，往后添加
            if headerTable[items[0]][1] is None:
                headerTable[items[0]][1] = node.children[items[0]]
            else:
                self.updateHeader(headerTable[items[0]][1], node.children[items[0]])
        # 递归
        if len(items) > 1:
            self.updateFpTree(items[1:], node.children[items[0]], headerTable)

    '''
    根据 data 创建 FP 树
    headerTable 结构为
    {"nodeName": [num, node], ...} 根据 node.nodeLink 可以找到整个树中的所有 nodeName
    '''
    def createFpTree(self, data, minSupport, showProgress=False):
        # 统计各项出现次数
        itemCount = {}
        # 第一次遍历，得到频繁 1 项集
        for transaction in data:
            for item in transaction:
                if item not in itemCount:
                    itemCount[item] = 1
                else:
                    itemCount[item] += 1

        headerTable = {}
        # 剔除不满足最小支持度的项
        for key in itemCount:
            if itemCount[key] >= minSupport:
                headerTable[key] = itemCount[key]

        # 满足最小支持度的频繁项集
        freqItemSet = set(headerTable.keys())
        if len(freqItemSet) == 0:
            return None, None

        for key in headerTable:
            headerTable[key] = [headerTable[key], None]  # element: [count, node]

        treeHeader = Node('headNode', 1, None)

        if showProgress:
            iterator = tqdm(data)
        else:
            iterator = data

        for transaction in iterator:  # 第二次遍历，建树
            localData = {}
            for item in transaction:
                # 过滤，只取该样本中满足最小支持度的频繁项
                if item in freqItemSet:
                    localData[item] = headerTable[item][0]  # element: count
            if len(localData) > 0:
                # 根据全局频数从大到小对单样本排序
                orderedItems = [v[0] for v in sorted(localData.items(), key=lambda x: x[1], reverse=True)]
                # 用过滤且排序后的样本更新树
                self.updateFpTree(orderedItems, treeHeader, headerTable)

        return treeHeader, headerTable

    '''
    递归将 node 的父节点添加到路径
    '''
    def findPath(self, node, nodePath):
        if node.parent is not None:
            nodePath.append(node.parent.name)
            self.findPath(node.parent, nodePath)

    '''
    根据节点名字，找出所有条件模式基
    '''
    def findCondPatternBase(self, nodeName, headerTable):
        treeNode = headerTable[nodeName][1]
        condPatternBase = {}  # 保存所有条件模式基
        while treeNode is not None:
            nodePath = []
            self.findPath(treeNode, nodePath)
            if len(nodePath) > 1:
                condPatternBase[frozenset(nodePath[:-1])] = treeNode.count
            treeNode = treeNode.nodeLink
        return condPatternBase

    def createCondFpTree(self, headerTable, minSupport, temp, freqItems, supportData):
        # 最开始的频繁项集是 headerTable 中的各元素
        freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
        for freq in freqs:  # 对每个频繁项
            freqSet = temp.copy()
            freqSet.add(freq)
            freqItems.add(frozenset(freqSet))
            # 检查该频繁项是否在 supportData 中
            if frozenset(freqSet) not in supportData:
                supportData[frozenset(freqSet)] = headerTable[freq][0]
            else:
                supportData[frozenset(freqSet)] += headerTable[freq][0]

            # 寻找到所有条件模式基
            condPatternBase = self.findCondPatternBase(freq, headerTable)
            # 将条件模式基字典转化为数组
            condPatternDataset = []
            for item in condPatternBase:
                itemTemp = list(item)
                itemTemp.sort()
                for _ in range(condPatternBase[item]):
                    condPatternDataset.append(itemTemp)
            # 创建条件模式树
            condTree, curHeadTable = self.createFpTree(condPatternDataset, minSupport)
            if curHeadTable is not None:
                # 递归挖掘条件 FP 树
                self.createCondFpTree(curHeadTable, minSupport, freqSet, freqItems, supportData)

    def generateL(self, dataSet, minSupport):
        freqItemSet = set()
        supportData = {}
        # 创建数据集的 FP 树
        treeHeader, headerTable = self.createFpTree(dataSet, minSupport, showProgress=True)
        # 创建各频繁一项的 FP 树，并挖掘频繁项并保存支持度计数
        self.createCondFpTree(headerTable, minSupport, set(), freqItemSet, supportData)

        maxL = 0
        # 将频繁项根据大小保存到指定的容器 L 中
        for item in freqItemSet:
            if len(item) > maxL:
                maxL = len(item)
        L = [set() for _ in range(maxL)]
        for item in freqItemSet:
            L[len(item) - 1].add(item)
        for i in range(len(L)):
            print(f"Frequent item {i + 1}: {len(L[i])}")
        return L, supportData

    def generateRule(self, data, minSupport, minConf):
        L, supportData = self.generateL(data, minSupport)
        ruleList = []
        subSetList = []
        for i in range(0, len(L)):
            for freqSet in L[i]:
                for subSet in subSetList:
                    if subSet.issubset(freqSet) and freqSet - subSet in supportData:
                        conf = supportData[freqSet] / supportData[freqSet - subSet]
                        bigRule = (freqSet - subSet, subSet, conf)
                        if conf >= minConf and bigRule not in ruleList:
                            ruleList.append(bigRule)
                subSetList.append(freqSet)
        ruleList = sorted(ruleList, key=lambda x: (x[2]), reverse=True)
        return ruleList


if __name__ == '__main__':
    fileName = "药方.xls"
    filePath = "../../data/" + fileName
    savePath = "../../log/" + fileName.split(".")[0] + "_fpgrowth.txt"

    minSupport = 500
    minConfidence = 0.9

    data = loadData(filePath)
    fpGrowth = FpGrowth()
    ruleList = fpGrowth.generateRule(data, minSupport, minConfidence)
    saveRule(ruleList, savePath)
