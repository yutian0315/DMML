

def loadData(path):
    ans = []
    with open(path, "r") as file:
        for line in file.readlines():
            if len(line) == 0:
                break
            line = line.strip('\n')
            items = list(set([int(i) for i in line.split(' ')]))
            items.sort()
            ans.append(items)
    return ans


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
        self.nodeLink = None  # 根据 nodeLink 可以找到整棵树中所有 nodeName 一样的节点
        self.parent = parentNode  # 父节点
        self.children = {}  # 子节点 {节点名字: 节点地址}


class FpGrowth:
    # 更新 headerTable 中的 node 节点形成的链表
    def updateHeader(self, node, targetNode):
        while node.nodeLink is not None:
            node = node.nodeLink
        node.nodeLink = targetNode

    # 更新 FP 树
    def updateFpTree(self, items, node, headerTable):
        if items[0] in node.children:
            node.children[items[0]].count += 1
        else:
            node.children[items[0]] = Node(items[0], 1, node)
            if headerTable[items[0]][1] is None:
                headerTable[items[0]][1] = node.children[items[0]]
            else:
                self.updateHeader(headerTable[items[0]][1], node.children[items[0]])
        if len(items) > 1:
            self.updateFpTree(items[1:], node.children[items[0]], headerTable)

    # 根据 data 创建 FP 树
    def createFpTree(self, data, minSupport):
        itemCount = {}
        for transaction in data:
            for item in transaction:
                itemCount[item] = itemCount.get(item, 0) + 1

        headerTable = {}
        for key, count in itemCount.items():
            if count >= minSupport:
                headerTable[key] = count

        freqItemSet = set(headerTable.keys())
        if not freqItemSet:
            return None, None

        for key in headerTable:
            headerTable[key] = [headerTable[key], None]

        treeHeader = Node('headNode', 1, None)
        for transaction in data:
            localData = {item: headerTable[item][0] for item in transaction if item in freqItemSet}
            if localData:
                orderedItems = [v[0] for v in sorted(localData.items(), key=lambda x: x[1], reverse=True)]
                self.updateFpTree(orderedItems, treeHeader, headerTable)

        return treeHeader, headerTable

    def findPath(self, node, nodePath):
        if node.parent is not None:
            nodePath.append(node.parent.name)
            self.findPath(node.parent, nodePath)

    def findCondPatternBase(self, nodeName, headerTable):
        treeNode = headerTable[nodeName][1]
        condPatternBase = {}
        while treeNode is not None:
            nodePath = []
            self.findPath(treeNode, nodePath)
            if len(nodePath) > 1:
                condPatternBase[frozenset(nodePath[:-1])] = treeNode.count
            treeNode = treeNode.nodeLink
        return condPatternBase

    def createCondFpTree(self, headerTable, minSupport, temp, freqItems, supportData):
        freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
        for freq in freqs:
            freqSet = temp.copy()
            freqSet.add(freq)
            freqItems.add(frozenset(freqSet))
            supportData[frozenset(freqSet)] = supportData.get(frozenset(freqSet), 0) + headerTable[freq][0]

            condPatternBase = self.findCondPatternBase(freq, headerTable)
            condPatternDataset = []
            for item, count in condPatternBase.items():
                itemList = sorted(item)
                condPatternDataset.extend([itemList] * count)

            condTree, curHeadTable = self.createFpTree(condPatternDataset, minSupport)
            if curHeadTable is not None:
                self.createCondFpTree(curHeadTable, minSupport, freqSet, freqItems, supportData)

    def generateL(self, dataset, minSupport):
        freqItemSet = set()
        supportData = {}
        treeHeader, headerTable = self.createFpTree(dataset, minSupport)
        if treeHeader and headerTable:
            self.createCondFpTree(headerTable, minSupport, set(), freqItemSet, supportData)

        maxL = max((len(item) for item in freqItemSet), default=0)
        L = [set() for _ in range(maxL)]
        for item in freqItemSet:
            L[len(item) - 1].add(item)

        for i, level in enumerate(L):
            print(f"Frequent item {i + 1}: {len(level)}")
        return L, supportData

    def generateR(self, dataset, minSupport, minConf):
        L, supportData = self.generateL(dataset, minSupport)
        ruleList = []
        subSetList = []

        for level in L:
            for freqSet in level:
                for subset in subSetList:
                    if subset.issubset(freqSet) and freqSet - subset in supportData:
                        conf = supportData[freqSet] / supportData[freqSet - subset]
                        if conf >= minConf:
                            ruleList.append((freqSet - subset, subset, conf))
                subSetList.append(freqSet)

        ruleList = sorted(ruleList, key=lambda x: x[2], reverse=True)
        return ruleList


if __name__ == '__main__':
    fileName = "tianchi.txt"
    filePath = "../../data/" + fileName
    savePath = "../../log/" + fileName.split(".")[0] + "_fpgrowth.txt"

    data = loadData(filePath)
    fpGrowth = FpGrowth()
    ruleList = fpGrowth.generateR(data, minSupport=15, minConf=0.7)
    saveRule(ruleList, savePath)
