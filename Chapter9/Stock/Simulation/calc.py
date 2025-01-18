import numpy as np

# 示例数据
predicted_prices = [
    [100, 101, 102],  # 股票1的预测收盘价
    [100, 99, 98],    # 股票2的预测收盘价
    [100, 102, 105],  # 股票3的预测收盘价
    [100, 103, 107],  # 股票4的预测收盘价
    [100, 104, 100],  # 股票5的预测收盘价
]

actual_prices = [
    [100, 100, 101],  # 股票1的真实收盘价
    [100, 98, 97],    # 股票2的真实收盘价
    [100, 101, 106],  # 股票3的真实收盘价
    [100, 102, 108],  # 股票4的真实收盘价
    [100, 105, 99],   # 股票5的真实收盘价
]

threshold = 0.005  # 0.5%的阈值
total_returns_predicted = []
total_returns_actual = []

# 计算收益率
for i in range(len(predicted_prices)):
    predicted_returns = []
    actual_returns = []
    current_investment = 1  # 初始投资100%
    
    for j in range(len(predicted_prices[i]) - 1):
        # 计算预测收益率
        predicted_return = (predicted_prices[i][j + 1] - predicted_prices[i][j]) / predicted_prices[i][j]
        # 计算实际收益率
        actual_return = (actual_prices[i][j + 1] - actual_prices[i][j]) / actual_prices[i][j]
        
        # 判断是否满足买入条件
        if predicted_return > threshold:
            current_investment *= (1 + predicted_return)  # 买入并计算收益
        else:
            current_investment *= (1 + actual_return)  # 不买入，根据实际收益计算
        
        predicted_returns.append(predicted_return)
        actual_returns.append(actual_return)
    
    total_returns_predicted.append(current_investment - 1)  # 总收益率（预测）
    total_returns_actual.append(current_investment - 1)     # 总收益率（实际）

# 输出结果
for idx in range(len(predicted_prices)):
    print(f"股票{idx + 1}的总预测收益率: {total_returns_predicted[idx]:.2%}")
    print(f"股票{idx + 1}的总实际收益率: {total_returns_actual[idx]:.2%}")
