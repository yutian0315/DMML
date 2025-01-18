import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torch
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tushare as ts
import numpy as np


class SingleStock(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)


def get_DataLoader(DATA_PATH, args): # 创建数据读取器 DataLoader

    stock_data = pd.read_csv(DATA_PATH)
    stock_data.drop('trade_date', axis=1, inplace=True)  # 删除列’trade_date‘
    close_max = stock_data['close'].max() #收盘价的最大值
    close_min = stock_data['close'].min() #收盘价的最小值
    df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max标准化

    # 定义X和Y，用于存储特征和标签
    # 根据前n天的数据，预测未来一天的收盘价(close)，
    sequence = args.sequence_length
    X, Y = [], []
    for i in range(df.shape[0] - sequence):
        # X存储前sequence天的数据
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
        # Y存储第sequence+1天的收盘价
        Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32))

    # 构建训练和测试数据集的batch
    total_len = len(Y)
    train_ratio = 0.95 # 定义训练数据占比
    trainx, trainy = X[:int(train_ratio * total_len)], Y[:int(train_ratio * total_len)]
    testx, testy = X[int(train_ratio * total_len):], Y[int(train_ratio * total_len):]
    
    # 创建数据读取器
    train_dataloder = DataLoader(dataset=SingleStock(trainx, trainy, transform=transforms.ToTensor()), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_dataloder = DataLoader(dataset=SingleStock(testx, testy, transform=transforms.ToTensor()), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    # 返回收盘价的最大值、最小值，训练和测试数据的DataLoader
    return close_max, close_min, train_dataloder, test_dataloder

if __name__ == "__main__":

    from config import OptInit
    import tushare as ts
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    opt = OptInit()
    opt.initialize()
    # get_DataLoader('ad470017ef0a1f4d9613fb95712158e61352b39a258ec63cdf9a17cd', opt.args)
    # 设置你的API密钥
    ts.set_token('ad470017ef0a1f4d9613fb95712158e61352b39a258ec63cdf9a17cd')
    pro = ts.pro_api()
    # 获取上证指数数据
    # 上证指数的ts_code是"000001.SH"
    data = pro.index_daily(ts_code='000001.SH', start_date='19980105', end_date='20240601')

    # 选择需要的列
    selected_columns = data[['trade_date', 'close', 'open', 'high', 'low', 'change', 'pct_chg', 'vol', 'amount']]

    # 输出数据
    print(selected_columns)

    # 反转行的顺序
    reversed_columns = selected_columns.sort_values(by='trade_date').reset_index(drop=True)
    print(reversed_columns)
    reversed_columns.to_csv('shanghai_index_data.csv', index=False, encoding='utf-8-sig')
    # # 1. 获取平安银行的历史数据
    # df = pro.daily(ts_code='601318.SH', start_date='20180101', end_date='20231231')
    # df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    # df['trade_date'] = pd.to_datetime(df['trade_date'])
    # df.set_index('trade_date', inplace=True)
    # df.sort_index(inplace=True)
    # print(df)

    # # 2. 数据预处理
    # # 计算涨跌情况
    # df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 明天涨则为1，跌则为0
    # df.dropna(inplace=True)
    # print(df)

