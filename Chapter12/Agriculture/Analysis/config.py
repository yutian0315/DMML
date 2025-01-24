# config.py

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

