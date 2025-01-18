import os  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.dates as mdates

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 定义股票文件名、颜色列表和股票代码  
stocks = ['600519_SH.csv', '601398_SH.csv', '601288_SH.csv', '601857_SH.csv', '601988_SH.csv']  
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  
labels = ['600519.SH 贵州茅台', '601288.SH 农业银行', '601398.SH 工商银行', '601857.SH 中国石油', '601988.SH 中国银行']


# 设置样式为 ggplot  
plt.style.use('ggplot')

# 设置图形大小  
fig, axs = plt.subplots(len(stocks), 1, figsize=(10, 18), sharex=True)  

# 遍历股票文件名，读取每个股票的数据并绘制  
for i, (stock, color, label) in enumerate(zip(stocks, colors, labels)):  
    stock_path = os.path.join('data', stock)  
    df = pd.read_csv(stock_path)  
    
    # 确保DataFrame中有'close'这一列  
    if 'close' in df.columns and 'trade_date' in df.columns:  
        # 将 'trade_date' 转换为 datetime 格式并按日期排序  
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  
        df.sort_values(by='trade_date', inplace=True)  
        
        # 绘制收盘价线图  
        axs[i].plot(df['trade_date'], df['close'], label=label, color=color)  
        
        # 设置子图的标题和 y 轴标签  
        axs[i].set_title(f'Closing Price of {label}', fontsize=14)  
        axs[i].set_ylabel('Closing Price')  
        
        # 显示图例  
        axs[i].legend(loc='upper left')  
        
        # 设置灰色网格背景  
        axs[i].set_facecolor('#f0f0f0')  
        
        # 显示网格线  
        axs[i].grid(True)  

        # 设置 x 轴的年份刻度显示，只在最后一个子图上显示 x 轴标签  
        if i == len(stocks) - 1:  
            axs[i].set_xlabel('Date')
            axs[i].xaxis.set_major_locator(mdates.YearLocator())  # 每年一个主刻度
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 显示年份  
        else:
            axs[i].tick_params(labelbottom=False)  # 去掉其他子图的 x 轴标签

# 调整布局以适应所有子图的标题和标签  
plt.tight_layout(pad=3.0)  

# 显示图表  
plt.show()
