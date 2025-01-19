# config.py

import os

# 路径配置
DATA_PATH = "./data/modified_train_yaOffsB.csv"
MODEL_SAVE_PATH = "./model/my_model.h5"
LOG_FILE = "./logs/train.log"

# 训练配置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
VALIDATION_SPLIT = 0.1
CLASSIFICATION_LOSS_WEIGHT = 4.0
ENCODING_DIM = 16

# 预处理配置
NUMERIC_FEATURES = [
    'Estimated_Insects_Count',
    'Number_Doses_Week',
    'Number_Weeks_Quit',
    'Number_Weeks_Used'
]
CATEGORICAL_FEATURES = [
    'Crop_Type',
    'Soil_Type',
    'Pesticide_Use_Category',
    'Season'
]

# 特征扰动百分比
PERTURBATION_PERCENTAGES = [-0.2, -0.1, 0.0, 0.1, 0.2]

# 特征名称映射（中英文）
FEATURE_NAME_MAP = {
    'Estimated_Insects_Count': '估算虫害数量',
    'Number_Doses_Week': '每周施药次数',
    'Number_Weeks_Quit': '停止施药周数',
    'Number_Weeks_Used': '施药总周数'
}

# 确定是否使用 GPU
USE_GPU = True  # 如果不使用 GPU，可以设置为 False
