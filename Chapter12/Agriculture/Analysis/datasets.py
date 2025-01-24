# datasets.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import config


def load_data():
    """加载原始数据"""
    data = pd.read_csv(config.DATA_PATH)
    return data


def preprocess_data(data):
    """数据预处理：填充、标准化、编码"""
    x = data.drop(['ID', 'Crop_Damage'], axis=1)
    y = data['Crop_Damage']

    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES

    # 数值管道：均值填充 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 分类管道：众数填充 + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 预处理
    x_processed = preprocessor.fit_transform(x)

    return x_processed, y, preprocessor


# 只填补缺失值，不进行独热编码
def preprocess_data_pure(data):
    """数据预处理：填充、标准化"""
    x = data.drop(['ID', 'Crop_Damage'], axis=1)
    y = data['Crop_Damage']

    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES

    # 数值管道：均值填充 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 分类管道：众数填充
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 预处理
    x_processed = preprocessor.fit_transform(x)

    return x_processed, y


def split_and_resample(x, y):
    """划分训练集和测试集，并使用 SMOTE 进行过采样"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)

    return x_train_resampled, x_test, y_train_resampled, y_test
