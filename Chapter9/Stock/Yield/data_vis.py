import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

# 设置你的Tushare API token
ts.set_token('ad470017ef0a1f4d9613fb95712158e61352b39a258ec63cdf9a17cd')
pro = ts.pro_api()

# 获取上证指数1998到2024年的基本盘数据
start_date = '19980101'
end_date = '20231231'

# 调用Tushare API获取数据
df = pro.index_daily(ts_code='000001.SH', start_date=start_date, end_date=end_date)

# 选择需要的字段
df = df[['trade_date', 'close', 'open', 'high', 'low', 'change', 'pct_chg', 'vol', 'amount']]

# 将数据的日期转换为datetime格式，并按日期排序
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(by='trade_date')

# 设置图形风格
plt.style.use('ggplot')

# 定义颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# 可视化每个字段
plt.figure(figsize=(14, 8))

# 关闭价格
plt.subplot(2, 2, 1)
plt.plot(df['trade_date'], df['close'], label='Close', color=colors[0])
plt.title('收盘价(Close Price)')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

# 开盘价格
plt.subplot(2, 2, 2)
plt.plot(df['trade_date'], df['open'], label='Open', color=colors[1])
plt.title('开盘价(Open Price)')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

# 最高价格
plt.subplot(2, 2, 3)
plt.plot(df['trade_date'], df['high'], label='High', color=colors[2])
plt.title('最高价格(High Price)')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

# 最低价格
plt.subplot(2, 2, 4)
plt.plot(df['trade_date'], df['low'], label='Low', color=colors[3])
plt.title('最低价格(Low Price)')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

plt.tight_layout()
plt.show()

# 交易量和成交额
plt.figure(figsize=(14, 8))

# 交易量
plt.subplot(2, 1, 1)
plt.plot(df['trade_date'], df['vol'], label='Volume', color=colors[4])
plt.title('成交量(Volume)')
plt.xlabel('日期')
plt.ylabel('成交量')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

# 成交额
plt.subplot(2, 1, 2)
plt.plot(df['trade_date'], df['amount'], label='Amount', color=colors[5])
plt.title('成交额(Amount)')
plt.xlabel('日期')
plt.ylabel('成交金额')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

plt.tight_layout()
plt.show()

# 涨跌幅和涨跌幅百分比
plt.figure(figsize=(14, 8))

# 涨跌幅
plt.subplot(2, 1, 1)
plt.plot(df['trade_date'], df['change'], label='Change', color=colors[6])
plt.title('涨跌幅度(Change)')
plt.xlabel('日期')
plt.ylabel('涨跌金额')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

# 涨跌幅百分比
plt.subplot(2, 1, 2)
plt.plot(df['trade_date'], df['pct_chg'], label='Percentage Change', color=colors[7])
plt.title('涨跌百分比(Percentage Change)')
plt.xlabel('日期')
plt.ylabel('百分比')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.minorticks_on()
plt.tick_params(axis='x', which='both', bottom=True, top=False)

plt.tight_layout()
plt.show()
