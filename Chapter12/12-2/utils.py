import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasets import load_data, preprocess_data_pure
import os
from datasets import load_data, preprocess_data, split_and_resample
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import config
from datasets import load_data
from models import build_autoencoder_with_classifier

# 设置绘图字体为中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def setup_logging():
    """配置日志记录，日志将同时输出到文件和控制台"""
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def save_model_to_file(model, path):
    """保存训练好的模型"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logging.info(f"Model saved to {path}")


def plot_precision_recall(thresholds, precision, recall):
    """绘制精准率和召回率随阈值变化的曲线并保存为 PDF"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label='精准率', linestyle='-', color='b')
    plt.plot(thresholds, recall, label='召回率', linestyle='--', color='r')
    plt.xlabel('阈值')
    plt.ylabel('得分')
    # 移除标题
    # plt.title('精准率和召回率随阈值变化的曲线')
    plt.legend()
    plt.grid(True)
    # 调整横坐标格式，确保清晰显示
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))  # 将阈值转化为百分比格式
    # 保存为 PDF 文件
    plt.savefig("precision_recall_curve.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_feature_sensitivity(metrics, numeric_features, feature_name_map):
    """绘制特征敏感度分析的各种曲线并保存为 PDF"""
    line_styles = ['-', '--', '-.', ':']
    if len(numeric_features) > len(line_styles):
        # 如果特征数量超过线条样式数量，则重复使用线条样式
        line_styles = line_styles * (len(numeric_features) // len(line_styles) + 1)

    # 绘制均值重构误差变化曲线
    plt.figure(figsize=(14, 6))
    for idx, feature in enumerate(numeric_features):
        plt.plot(metrics[feature]['percent'], metrics[feature]['mean_error'],
                 linestyle=line_styles[idx], label=feature_name_map[feature])
    plt.xlabel('扰动百分比 (%)')
    plt.ylabel('均值重构误差')
    plt.legend(title='特征')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.savefig("mean_reconstruction_error.pdf", format="pdf", bbox_inches="tight")  # 保存为 PDF
    plt.show()

    # 绘制重构误差方差变化曲线
    plt.figure(figsize=(14, 6))
    for idx, feature in enumerate(numeric_features):
        plt.plot(metrics[feature]['percent'], metrics[feature]['variance_error'],
                 linestyle=line_styles[idx], label=feature_name_map[feature])
    plt.xlabel('扰动百分比 (%)')
    plt.ylabel('重构误差方差')
    plt.legend(title='特征')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.savefig("variance_reconstruction_error.pdf", format="pdf", bbox_inches="tight")  # 保存为 PDF
    plt.show()

    # 绘制异常样本数量变化曲线
    plt.figure(figsize=(14, 6))
    for idx, feature in enumerate(numeric_features):
        plt.plot(metrics[feature]['percent'], metrics[feature]['num_anomalous'],
                 linestyle=line_styles[idx], label=feature_name_map[feature])
    plt.xlabel('扰动百分比 (%)')
    plt.ylabel('异常样本数量')
    plt.legend(title='特征')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.savefig("num_anomalous_samples.pdf", format="pdf", bbox_inches="tight")  # 保存为 PDF
    plt.show()

    # 绘制最大值和最小值变化曲线
    plt.figure(figsize=(14, 6))
    for idx, feature in enumerate(numeric_features):
        plt.plot(metrics[feature]['percent'], metrics[feature]['max_error'],
                 linestyle=line_styles[idx], label=f"{feature_name_map[feature]} - 最大值")
        plt.plot(metrics[feature]['percent'], metrics[feature]['min_error'],
                 linestyle=line_styles[idx], label=f"{feature_name_map[feature]} - 最小值")
    plt.xlabel('扰动百分比 (%)')
    plt.ylabel('重构误差 (最大值/最小值)')
    plt.legend(title='特征')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.savefig("max_min_reconstruction_error.pdf", format="pdf", bbox_inches="tight")  # 保存为 PDF
    plt.show()


def perform_feature_sensitivity_analysis(threshold):
    """执行特征敏感度分析"""
    # 获取所有特征（包括数值和类别特征）
    all_features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    feature_name_map = config.FEATURE_NAME_MAP
    perturbation_percentages = config.PERTURBATION_PERCENTAGES

    # 存储每个特征的变化数据
    feature_impact_auc_all = {feature: [] for feature in all_features}
    feature_impact_auprc_all = {feature: [] for feature in all_features}

    # 初始化 metrics
    metrics = {feature: {
        'percent': [],
        'mean_error': [],
        'variance_error': [],
        'num_anomalous': [],
        'max_error': [],
        'min_error': []
    } for feature in all_features}

    # 重复五次实验
    for _ in range(5):
        # 1. 划分数据集
        original_data = load_data()
        # 获取没有独热编码但是填补了缺失值的训练集和测试集
        X, y = preprocess_data_pure(original_data)

        X_train_old, X_test_old, y_train, y_test = split_and_resample(X, y)

        # 真正的训练集和测试集需要保证搞了独热编码
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
        ])

        # 组合预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        X_train_df = pd.DataFrame(X_train_old, columns=config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
        X_test_df = pd.DataFrame(X_test_old, columns=config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)

        # 预处理
        X_train = preprocessor.fit_transform(X_train_df)  # 这个是终极可以用的数据
        X_test = preprocessor.fit_transform(X_test_df)  # 这个是终极可以用的数据

        # 2. 训练一个新的模型，使用预处理后的数据
        logging.info("训练新模型...")
        model = build_autoencoder_with_classifier(input_dim=X_train.shape[1],
                                                  encoding_dim=config.ENCODING_DIM,
                                                  classification_loss_weight=config.CLASSIFICATION_LOSS_WEIGHT)

        # 原始数据训练模型
        model.fit(X_train, {'reconstruction': X_train, 'classification': y_train},
                  epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        # 3. 使用模型对测试集进行预测
        reconstruction_pred_original, classification_pred_original = model.predict(X_test)
        y_pred_original = (classification_pred_original.ravel() > threshold).astype(int)
        auc_original = roc_auc_score(y_test, classification_pred_original.ravel())
        auprc_original = average_precision_score(y_test, classification_pred_original.ravel())

        # 打印模型评估信息
        logging.info(f"Original AUC: {auc_original:.4f}")
        logging.info(f"Original AUPRC: {auprc_original:.4f}")

        # 对每个特征进行扰动并记录其影响
        for feature in all_features:
            # 扰动当前特征的值（在原始数据上打乱）
            X_test_perturbed = X_test_df.copy()
            X_test_perturbed[feature] = np.random.permutation(X_test_perturbed[feature].values)
            # 对扰动后的数据进行预处理（只在扰动分析时才预处理）
            X_test_processed_perturbed = preprocessor.transform(X_test_perturbed)
            # 使用模型对扰动后的数据进行预测
            reconstruction_pred_perturbed, classification_pred_perturbed = model.predict(X_test_processed_perturbed)
            # 计算打乱后的 AUC 和 AUPRC
            auc_perturbed = roc_auc_score(y_test, classification_pred_perturbed.ravel())
            auprc_perturbed = average_precision_score(y_test, classification_pred_perturbed.ravel())
            # 记录每次实验中的 AUC 和 AUPRC 变化
            feature_impact_auc_all[feature].append(auc_original - auc_perturbed)
            feature_impact_auprc_all[feature].append(auprc_original - auprc_perturbed)

    # 计算均值
    feature_impact_auc_mean = {feature: np.mean(impacts) for feature, impacts in feature_impact_auc_all.items()}
    feature_impact_auprc_mean = {feature: np.mean(impacts) for feature, impacts in feature_impact_auprc_all.items()}

    # 特征名称映射（中英文）
    FEATURE_NAME_MAP = {
        'Estimated_Insects_Count': '估算虫害数量',
        'Number_Doses_Week': '每周施药次数',
        'Number_Weeks_Quit': '停止施药周数',
        'Number_Weeks_Used': '施药总周数',
        'Crop_Type': '作物类型',
        'Soil_Type': '土壤类型',
        'Pesticide_Use_Category': '施药类别',
        'Season': '季节'
    }

    # 将 AUC 数据保存到文件
    with open('AUC.txt', 'w', encoding='utf-8') as auc_file:
        auc_file.write("特征\tAUC 变化\n")  # 文件头
        for feature, auc_change in feature_impact_auc_mean.items():
            # 将特征和它的 AUC 变化写入文件
            auc_file.write(f"{FEATURE_NAME_MAP.get(feature, feature)}\t{auc_change:.4f}\n")

    # 将 AUPRC 数据保存到文件
    with open('AUPRC.txt', 'w', encoding='utf-8') as auprc_file:
        auprc_file.write("特征\tAUPRC 变化\n")  # 文件头
        for feature, auprc_change in feature_impact_auprc_mean.items():
            # 将特征和它的 AUPRC 变化写入文件
            auprc_file.write(f"{FEATURE_NAME_MAP.get(feature, feature)}\t{auprc_change:.4f}\n")
